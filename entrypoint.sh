#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] ===== Arctic-TILT GPU container start ====="

# HF кеш — на /workspace (обычно это персистентный volume на RunPod)
export HF_HOME="${HF_HOME:-/workspace/cache/hf}"
mkdir -p "${HF_HOME}"

# Жёстко: работаем только с xformers
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_SKIP_PROFILE_RUN="${VLLM_SKIP_PROFILE_RUN:-1}"

echo "[entrypoint] HF_HOME=${HF_HOME}"
echo "[entrypoint] VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "[entrypoint] VLLM_SKIP_PROFILE_RUN=${VLLM_SKIP_PROFILE_RUN}"
echo "[entrypoint] Using python: $(command -v python)"

# -------------------------- Быстрый probe окружения ---------------------------
python << 'PY'
import os, sys, importlib

print("[probe] sys.version:", sys.version.replace("\n", " "))
print("[probe] sys.executable:", sys.executable)
print("[probe] HF_HOME:", os.getenv("HF_HOME"))
print("[probe] VLLM_ATTENTION_BACKEND:", os.getenv("VLLM_ATTENTION_BACKEND"))
print("[probe] VLLM_SKIP_PROFILE_RUN:", os.getenv("VLLM_SKIP_PROFILE_RUN"))

mods = ("torch", "vllm", "xformers", "fastapi", "uvicorn")
errors = {}

for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"[probe] {mod}: OK")
    except Exception as e:
        errors[mod] = e
        print(f"[probe] {mod}: FAIL -> {e!r}")

# xformers — критичен, без него дальше не идём
if "xformers" in errors:
    print("[probe] xformers is mandatory for Arctic-TILT (no SDPA fallback). Exiting.", file=sys.stderr)
    sys.exit(1)

# torch/vllm тоже критичны — если отвалились, нет смысла продолжать
for critical in ("torch", "vllm"):
    if critical in errors:
        print(f"[probe] {critical} failed to import, aborting startup.", file=sys.stderr)
        sys.exit(1)

PY

# Где лежит репозиторий на volume
APP_REPO_URL="${APP_REPO_URL:-https://github.com/STrachov/OCRlty.git}"
APP_SRC_ROOT="${APP_SRC_ROOT:-/workspace/src}"   # тут живёт git-копия
APP_FALLBACK_ROOT="/opt/app"                    # код, запечённый в образ
APP_ROOT="$APP_FALLBACK_ROOT"

echo "[entrypoint] ===== Code selection phase ====="
echo "[entrypoint] APP_REPO_URL=${APP_REPO_URL}"
echo "[entrypoint] APP_SRC_ROOT=${APP_SRC_ROOT}"
echo "[entrypoint] Fallback APP_ROOT=${APP_FALLBACK_ROOT}"

if command -v git >/dev/null 2>&1; then
  # Если репо уже есть на volume — пробуем обновить
  if [ -d "${APP_SRC_ROOT}/.git" ]; then
    echo "[entrypoint] Found existing git repo at ${APP_SRC_ROOT}, running git pull..."
    (
      cd "${APP_SRC_ROOT}" && \
      git pull --ff-only || echo "[entrypoint] git pull failed, keeping existing code"
    )
    APP_ROOT="${APP_SRC_ROOT}"
  else
    # Если директории нет или она пустая — клонируем
    if [ ! -d "${APP_SRC_ROOT}" ] || [ -z "$(ls -A "${APP_SRC_ROOT}" 2>/dev/null || true)" ]; then
      echo "[entrypoint] Cloning repo into ${APP_SRC_ROOT}..."
      mkdir -p "${APP_SRC_ROOT}"
      if git clone --depth 1 "${APP_REPO_URL}" "${APP_SRC_ROOT}"; then
        echo "[entrypoint] Clone OK."
        APP_ROOT="${APP_SRC_ROOT}"
      else
        echo "[entrypoint] git clone FAILED, using baked code at ${APP_FALLBACK_ROOT}"
        APP_ROOT="${APP_FALLBACK_ROOT}"
      fi
    else
      echo "[entrypoint] ${APP_SRC_ROOT} not empty and not a git repo, using baked code."
      APP_ROOT="${APP_FALLBACK_ROOT}"
    fi
  fi
else
  echo "[entrypoint] git not found, using baked code at ${APP_FALLBACK_ROOT}"
  APP_ROOT="${APP_FALLBACK_ROOT}"
fi

echo "[entrypoint] Final APP_ROOT=${APP_ROOT}"

# -------------------------- Debug sleep (optional) -----------------------------
if [ "${SLEEP_ON_START:-0}" = "1" ]; then
  echo "[entrypoint] SLEEP_ON_START=1 → sleeping indefinitely for debug..."
  exec tail -f /dev/null
fi

# ---------------------------- Порты и ENV -------------------------------------

API_PORT="${API_PORT:-8000}"
TILT_PORT="${TILT_PORT:-${VLLM_PORT:-8001}}"

echo "[entrypoint] API_PORT=${API_PORT}"
echo "[entrypoint] TILT_PORT=${TILT_PORT}"

# URL TILT-бэкенда для основного API (если не задан снаружи)
TILT_BASE_URL="http://127.0.0.1:${TILT_PORT}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-${TILT_BASE_URL}/v1}"

echo "[entrypoint] VLLM_BASE_URL=${VLLM_BASE_URL}"

# ---------------------------- Старт процессов ---------------------------------

UVICORN_EXTRA_ARGS=""
# Для dev-подов можно включать хот-релоад (подхватывает изменения после git pull)
# if [ "${UVICORN_RELOAD:-0}" = "1" ]; then
#   UVICORN_EXTRA_ARGS="--reload"
#   echo "[entrypoint] Uvicorn reload mode ENABLED"
# fi
UVICORN_EXTRA_ARGS="--reload"

# Ограничиваем потоки для OCR (Paddle / OpenCV)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

cd "${APP_ROOT}"

# --- 1) GPU: tilt_api (Arctic-TILT + vLLM) на :TILT_PORT ---
echo "[entrypoint] Starting tilt_api (GPU) from ${APP_ROOT} on 0.0.0.0:${TILT_PORT}"

python -m uvicorn apps.tilt_api:app \
    --app-dir "${APP_ROOT}" \
    --host 0.0.0.0 \
    --port "${TILT_PORT}" \
    ${UVICORN_EXTRA_ARGS} &
TILT_PID=$!

# Ждём готовности tilt_api
echo "[entrypoint] Waiting for tilt_api at ${TILT_BASE_URL}/v1/health ..."
TILT_READY=0
for i in $(seq 1 60); do
  if curl -fsS "${TILT_BASE_URL}/v1/health" >/dev/null 2>&1; then
    echo "[entrypoint] tilt_api is UP (attempt ${i})"
    TILT_READY=1
    break
  fi
  echo "[entrypoint] tilt_api not ready yet, retry ${i}/60..."
  sleep 2
done

if [ "${TILT_READY}" -ne 1 ]; then
  echo "[entrypoint] ERROR: tilt_api failed to become ready, exiting."
  # Пытаемся аккуратно завершить tilt_api
  if kill "${TILT_PID}" 2>/dev/null; then
    wait "${TILT_PID}" || true
  fi
  exit 1
fi

# --- 2) CPU: основной API (OCR + TILT client) на :API_PORT ---
echo "[entrypoint] Starting main API (OCR+TILT) on 0.0.0.0:${API_PORT}"

exec python -m uvicorn apps.api_gpu.main:app \
    --app-dir "${APP_ROOT}" \
    --host 0.0.0.0 \
    --port "${API_PORT}" \
    ${UVICORN_EXTRA_ARGS}

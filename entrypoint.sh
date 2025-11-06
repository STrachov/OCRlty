#!/usr/bin/env bash
set -euo pipefail

: "${SRC_DIR:=/workspace/src}"
: "${VENV_DIR:=/workspace/venv}"
: "${HF_HOME:=/workspace/cache/hf}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8001}"
: "${VENV_RESET:=}"   # 1|true|yes => пересоздать venv

export HF_HOME
mkdir -p "$HF_HOME"

APP_FILE="$SRC_DIR/apps/tilt_api.py"
if [[ ! -f "$APP_FILE" ]]; then
  echo "[entrypoint] ERROR: $APP_FILE is missing in image."
  exit 1
fi

# --- reset venv по флагу ---
case "${VENV_RESET,,}" in
  1|true|yes)
    echo "[entrypoint] VENV_RESET is set → recreating venv at $VENV_DIR"
    rm -rf "$VENV_DIR"
  ;;
esac
# --- создаём/активируем venv (БЕЗ --system-site-packages!) ---
if [[ ! -d "$VENV_DIR/bin" ]]; then
  python3.10 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- кладём наш sitecustomize в site-packages venv ---
SITE_SRC="${SRC_DIR}/sitecustomize.py"
SITE_DST="$VENV_DIR/lib/python3.10/site-packages/sitecustomize.py"
if [[ -f "$SITE_SRC" ]]; then
  mkdir -p "$(dirname "$SITE_DST")"
  cp -f "$SITE_SRC" "$SITE_DST"
fi

# На всякий случай ставим наш src в самый ПЕРВЫЙ элемент sys.path
export PYTHONPATH="$SRC_DIR:$VENV_DIR/lib/python3.10/site-packages:${PYTHONPATH:-}"

# --- проверка: покажи, ОТКУДА импортировался sitecustomize ---
python - <<'PY'
import sys, importlib
print("[probe] sys.executable:", sys.executable)
m = importlib.import_module("sitecustomize")
print("[probe] sitecustomize loaded from:", getattr(m, "__file__", "<builtin>"))
PY

# нормализуем странный ввод из UI: VLLM_PLUGINS='""' -> empty
if [[ "${VLLM_PLUGINS:-}" == '""' ]]; then export VLLM_PLUGINS=; fi
echo "[entrypoint] VLLM_PLUGINS=$(printf %q "${VLLM_PLUGINS:-}")"
echo "[entrypoint] VLLM_SKIP_PROFILE_RUN=${VLLM_SKIP_PROFILE_RUN:-}"

# быстрый импорт-чек vLLM
python - <<'PY'
import sys
try:
    import vllm
    print(f"[entrypoint] vLLM OK:", getattr(vllm,"__version__","unknown"))
except Exception as e:
    print(f"[entrypoint][FATAL] vLLM not importable:", e)
    sys.exit(2)
PY

export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"
exec python -m uvicorn apps.tilt_api:app --host "$HOST" --port "$PORT"

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

# --- создаём/активируем venv (по умолчанию переиспользуем) ---
if [[ ! -d "$VENV_DIR/bin" ]]; then
  python3.10 -m venv "$VENV_DIR" --system-site-packages
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

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

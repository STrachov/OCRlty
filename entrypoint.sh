#!/usr/bin/env bash
set -euo pipefail

: "${SRC_DIR:=/workspace/src}"
: "${VENV_DIR:=/workspace/venv}"
: "${HF_HOME:=/workspace/cache/hf}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8001}"
# Если нужно пересоздать venv, передай VENV_RESET=1 (или true/yes)
: "${VENV_RESET:=}"

export HF_HOME
mkdir -p "$HF_HOME"

APP_FILE="$SRC_DIR/apps/tilt_api.py"
if [[ ! -f "$APP_FILE" ]]; then
  echo "[entrypoint] ERROR: $APP_FILE is missing in image."
  exit 1
fi

# --- ТОЛЬКО ПО ЗАПРОСУ пересоздаём venv ---
case "${VENV_RESET,,}" in
  1|true|yes)
    echo "[entrypoint] VENV_RESET is set → recreating venv at $VENV_DIR"
    rm -rf "$VENV_DIR"
  ;;
  *)
    : # ничего не делаем
  ;;
esac

# Создаём venv, если его нет (по умолчанию используем существующий как есть)
if [[ ! -d "$VENV_DIR/bin" ]]; then
  python3.10 -m venv "$VENV_DIR" --system-site-packages
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Опциональная проверка vLLM (оставил как раньше; можно убрать, если не нужна)
python - <<'PY'
import sys
try:
    import vllm
    print(f"[entrypoint] vLLM OK: {getattr(vllm, '__version__', 'unknown')}")
except Exception as e:
    print(f"[entrypoint][FATAL] vLLM not importable: {e}")
    sys.exit(2)
PY

export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"
exec python -m uvicorn apps.tilt_api:app --host "$HOST" --port "$PORT"

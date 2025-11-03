#!/usr/bin/env bash
set -euo pipefail

: "${SRC_DIR:=/workspace/src}"
: "${VENV_DIR:=/workspace/venv}"
: "${HF_HOME:=/workspace/cache/hf}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8001}"

export HF_HOME
mkdir -p "$HF_HOME"

APP_FILE="$SRC_DIR/apps/tilt_api.py"
if [[ ! -f "$APP_FILE" ]]; then
  echo "[entrypoint] ERROR: $APP_FILE is missing in image."
  exit 1
fi

# Лёгкий venv, чтобы не трогать системные пакеты (torch/vllm уже в образе)
if [[ ! -d "$VENV_DIR/bin" ]]; then
  python3.10 -m venv "$VENV_DIR" --system-site-packages
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"

# Sanity: vLLM есть?
python - <<'PY'
import sys
try:
    import vllm
    print(f"[entrypoint] vLLM OK: {getattr(vllm, '__version__', 'unknown')}")
except Exception as e:
    print(f"[entrypoint][FATAL] vLLM not importable: {e}")
    sys.exit(2)
PY

# (не ставим reqs на рантайме — всё уже запечено в образ)
exec python -m uvicorn apps.tilt_api:app --host "$HOST" --port "$PORT"

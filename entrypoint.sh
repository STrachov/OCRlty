#!/usr/bin/env bash
set -euo pipefail

# Дефолты (перекрывай через ENV при запуске)
: "${SRC_DIR:=/workspace/src}"
: "${VENV_DIR:=/workspace/venv}"
: "${HF_HOME:=/workspace/cache/hf}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8001}"

export HF_HOME
mkdir -p "$HF_HOME" "$VENV_DIR"

# Проверки наличия исходников
APP_FILE="$SRC_DIR/apps/tilt_api.py"
if [[ ! -f "$APP_FILE" ]]; then
  echo "[entrypoint] ERROR: $APP_FILE is missing in image. Did you COPY the repo into /workspace/src?"
  exit 1
fi

# VENV с доступом к системным пакетам (torch/vllm уже в образе)
if [[ ! -d "$VENV_DIR/bin" ]]; then
  python3.10 -m venv "$VENV_DIR" --system-site-packages
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# PYTHONPATH на всякий случай (если нет __init__.py в apps/)
export PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}"

# Проверка доступности vLLM из venv (должен видеть системный пакет)
python - <<'PY'
import sys
try:
    import vllm
    print(f"[entrypoint] vLLM OK: {getattr(vllm, '__version__', 'unknown')}")
except Exception as e:
    print(f"[entrypoint][FATAL] vLLM not importable in venv (expected via system-site-packages): {e}")
    sys.exit(2)
PY

# Sanity: согласованность HF-библиотек (как в твоём скрипте)
python - <<'PY'
import sys
def parse(v: str):
    ps=[]
    for p in v.split('.'):
        num=''.join(ch for ch in p if ch.isdigit())
        ps.append(int(num) if num else 0)
    return tuple(ps or [0])

try:
    import transformers, tokenizers
    tv = parse(getattr(transformers, "__version__", "0"))
    kv = parse(getattr(tokenizers, "__version__", "0"))
    if tv >= (4,57,0) and not ((0,22,0) <= kv <= (0,23,0)):
        print(f"[entrypoint][FATAL] transformers {transformers.__version__} incompatible with tokenizers {tokenizers.__version__}", flush=True)
        sys.exit(2)
    print(f"[entrypoint] HF OK: transformers={transformers.__version__} tokenizers={tokenizers.__version__}")
except Exception as e:
    print(f"[entrypoint][WARN] HF pair check skipped: {e}", flush=True)
PY

# Запуск FastAPI (Arctic-TILT на vLLM внутри процесса)
exec python -m uvicorn apps.tilt_api:app --host "$HOST" --port "$PORT"

#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# ЕДИНСТВЕННЫЙ источник дефолтов (перекрывай через ENV при запуске)
# ----------------------------
: "${SRC_DIR:=/workspace/src}"
: "${VENV_DIR:=/workspace/venv}"
: "${WHEELHOUSE:=/workspace/wheelhouse}"  
: "${HF_HOME:=/workspace/cache/hf}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8001}"
# 0 = строго оффлайн; 1 = разрешить онлайн-фолбэк для requirements
: "${ALLOW_ONLINE_FALLBACK:=0}"

export HF_HOME

# ----------------------------
# Жёсткие проверки наличия исходников/требований
# ----------------------------
if [[ ! -d "$SRC_DIR" ]]; then
  echo "[entrypoint] ERROR: source directory $SRC_DIR is missing (mount your repo)."
  exit 1
fi
REQ_FILE="$SRC_DIR/requirements-gpu.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "[entrypoint] ERROR: $REQ_FILE is missing in repo."
  exit 1
fi
APP_FILE="$SRC_DIR/apps/tilt_api.py"
if [[ ! -f "$APP_FILE" ]]; then
  echo "[entrypoint] ERROR: $APP_FILE is missing in repo."
  exit 1
fi

mkdir -p "$WHEELHOUSE" "$HF_HOME"

# ----------------------------
# VENV: reuse системных пакетов (Torch/CUDA + vLLM уже в образе)
# ----------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  python3.10 -m venv "$VENV_DIR" --system-site-packages
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Проверим, что системный vLLM действительно виден в venv
python - <<'PY'
import sys
try:
    import vllm
    print(f"[entrypoint] vLLM OK: {getattr(vllm, '__version__', 'unknown')}")
except Exception as e:
    print(f"[entrypoint][FATAL] vLLM not importable in venv (expected system-site-packages). {e}")
    sys.exit(2)
PY

# ----------------------------
# Установка зависимостей: offline-first из wheelhouse → (опц.) онлайн-фолбэк
# ----------------------------
# pip обновляем только если разрешён онлайновый фолбэк
if [[ "${ALLOW_ONLINE_FALLBACK}" == "1" ]]; then
  python -m pip install -U --no-cache-dir pip || true
fi

if ! python -m pip install --no-cache-dir --no-index --find-links "$WHEELHOUSE" -r "$REQ_FILE"; then
  if [[ "${ALLOW_ONLINE_FALLBACK}" == "1" ]]; then
    python -m pip install --no-cache-dir -r "$REQ_FILE"
  else
    echo "[entrypoint] ERROR: offline install failed and ALLOW_ONLINE_FALLBACK=0"
    exit 1
  fi
fi

# ----------------------------
# Sanity-check: совместимость transformers ↔ tokenizers (без внешних зависимостей)
# ----------------------------
python - <<'PY'
import sys
def parse(v: str):
    parts=[]
    for p in v.split('.'):
        num=''.join(ch for ch in p if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts or [0])

try:
    import transformers, tokenizers
    tv = parse(getattr(transformers, "__version__", "0"))
    kv = parse(getattr(tokenizers, "__version__", "0"))
    # Для transformers >= 4.57.0 требуем 0.22.0 <= tokenizers <= 0.23.0
    if tv >= (4,57,0) and not ((0,22,0) <= kv <= (0,23,0)):
        print(f"[entrypoint][FATAL] transformers {transformers.__version__} incompatible with tokenizers {tokenizers.__version__}", flush=True)
        sys.exit(2)
    print(f"[entrypoint] HF OK: transformers={transformers.__version__} tokenizers={tokenizers.__version__}")
except Exception as e:
    print(f"[entrypoint][WARN] HF pair check skipped: {e}", flush=True)
PY

# ----------------------------
# Запуск API
# ----------------------------
exec python -m uvicorn apps.tilt_api:app --host "$HOST" --port "$PORT"

#!/usr/bin/env bash
set -euo pipefail

: "${SRC_DIR:=/workspace/src}"
: "${VENV_DIR:=/workspace/venv}"
: "${WHEELHOUSE:=/workspace/wheelhouse}"
: "${HF_HOME:=/workspace/cache/hf}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8001}"

export HF_HOME

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
# VENV: reuse системных пакетов (Torch/CUDA из образа)
# ----------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  python3.10 -m venv "$VENV_DIR" --system-site-packages
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ----------------------------
# pip: строго оффлайн по wheelhouse (без сети)
# (не обновляем pip, чтобы не упасть оффлайн)
# ----------------------------
export PIP_NO_INDEX=1
export PIP_FIND_LINKS="$WHEELHOUSE"

# vLLM: должен быть в системе ИЛИ колесом в wheelhouse
if ! python - <<'PY'
try:
    import vllm  # noqa
    import sys; sys.exit(0)
except Exception:
    import sys; sys.exit(1)
PY
then
  shopt -s nullglob
  VLLM_WHEELS=("$WHEELHOUSE"/vllm-*.whl)
  if (( ${#VLLM_WHEELS[@]} == 0 )); then
    echo "[entrypoint] ERROR: vLLM not found system-wide AND no wheel in $WHEELHOUSE."
    echo "Place vllm-*.whl (e.g. 0.8.3 cp310 cu124) into $WHEELHOUSE or bake vLLM into base image."
    exit 1
  fi
  python -m pip install --no-cache-dir "${VLLM_WHEELS[@]}"
fi

# Устанавливаем требования строго из wheelhouse
python -m pip install --no-cache-dir -r "$REQ_FILE"

# Быстрый sanity-check совместимости transformers ↔ tokenizers
python - <<'PY'
import sys
try:
    import transformers, tokenizers
    def parse(v):
        parts = []
        for p in v.split('.'):
            if p.isdigit():
                parts.append(int(p))
            else:
                # отбрасываем суффиксы типа rc/post
                num = ''.join(ch for ch in p if ch.isdigit())
                parts.append(int(num) if num else 0)
        return tuple(parts or [0])
    tv = parse(getattr(transformers, "__version__", "0"))
    kv = parse(getattr(tokenizers, "__version__", "0"))
    # Для transformers >= 4.57.0 — требуем 0.22.0 <= tokenizers <= 0.23.0
    if tv >= (4,57,0) and not ((0,22,0) <= kv <= (0,23,0)):
        print(f"[entrypoint][FATAL] transformers {transformers.__version__} incompatible with tokenizers {tokenizers.__version__}", flush=True)
        sys.exit(2)
except Exception as e:
    print(f"[entrypoint][WARN] HF pair check skipped: {e}", flush=True)
PY

# ----------------------------
# Старт приложения
# ----------------------------
exec python -m uvicorn apps.tilt_api:app --host "$HOST" --port "$PORT"

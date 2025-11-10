#!/usr/bin/env bash
set -Eeuo pipefail

echo "[entrypoint] Python: $(python -V)"
echo "[entrypoint] Torch:   $(python -c 'import torch,sys;print(torch.__version__)' 2>/dev/null || echo 'not importable')"
echo "[entrypoint] CUDA:    $(python -c 'import torch;print(torch.version.cuda)' 2>/dev/null || echo 'n/a')"
echo "[entrypoint] vLLM:    $(python -c 'import vllm,sys;print(vllm.__version__)' 2>/dev/null || echo 'not importable')"

# Жёстко форсим безопасные для нас режимы
export VLLM_USE_FLASH_ATTENTION=${VLLM_USE_FLASH_ATTENTION:-0}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}
export VLLM_SKIP_PROFILE_RUN=${VLLM_SKIP_PROFILE_RUN:-1}
export HF_HOME=${HF_HOME:-/workspace/cache/hf}

echo "[entrypoint] VLLM_USE_FLASH_ATTENTION=${VLLM_USE_FLASH_ATTENTION}"
echo "[entrypoint] VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "[entrypoint] VLLM_SKIP_PROFILE_RUN=${VLLM_SKIP_PROFILE_RUN}"
echo "[entrypoint] HF_HOME=${HF_HOME}"

# Проверка загрузки sitecustomize и пути
python - <<'PY'
import sys, os
print(f"[probe] sys.executable: {sys.executable}")
try:
    import sitecustomize as sc
    print(f"[probe] sitecustomize loaded from: {sc.__file__}")
except Exception as e:
    print(f"[probe][WARN] sitecustomize not loaded: {e}")
PY

# Мини-проверки на импорт
python - <<'PY'
mods = ["vllm","transformers","fastapi","paddleocr","paddle","numpy","pandas"]
for m in mods:
    try:
        __import__(m)
        print(f"[probe] import ok: {m}")
    except Exception as e:
        print(f"[probe][WARN] import fail: {m}: {e}")
PY

# На всякий — добавим src в PYTHONPATH (ожидаем apps/tilt_api.py внутри /workspace/src)
export PYTHONPATH="/workspace/src:${PYTHONPATH:-}"

APP_MODULE="apps.tilt_api:app"
if [[ ! -f "/workspace/src/apps/tilt_api.py" ]]; then
  echo "[entrypoint][FATAL] /workspace/src/apps/tilt_api.py not found."
  echo "                 Проверьте структуру репозитория. Должно быть: src/apps/tilt_api.py"
  exit 2
fi

HOST="${UVICORN_HOST:-0.0.0.0}"
PORT="${UVICORN_PORT:-8001}"

echo "[entrypoint] Starting uvicorn ${APP_MODULE} on ${HOST}:${PORT}"
exec python -m uvicorn "${APP_MODULE}" --host "${HOST}" --port "${PORT}" --timeout-keep-alive 75

#!/usr/bin/env bash
set -euo pipefail

# каталоги под volume/кэши (RunPod монтирует /workspace)
mkdir -p /workspace/venv /workspace/.cache/pip /workspace/cache/hf

# всегда используем venv на volume, но с доступом к пакетам из /opt/venv (vLLM, torch)
if [[ ! -x /workspace/venv/bin/python ]]; then
  python3.10 -m venv /workspace/venv
  /workspace/venv/bin/python -m pip install --upgrade pip wheel
  /workspace/venv/bin/python -m pip install -r /opt/app/requirements-gpu.txt
fi

# 2) подмешиваем пакеты из /opt/venv (там vLLM + torch)
export PYTHONPATH="/opt/venv/lib/python3.10/site-packages:${PYTHONPATH:-}"
# 3) используем именно /workspace/venv/python
export PATH="/workspace/venv/bin:${PATH}"

# (необязательно, но полезно один раз увидеть в логах)
/workspace/venv/bin/python - <<'PY'
import vllm, torch; print("vLLM:", vllm.__version__, "torch:", torch.__version__)
PY


export PATH="/workspace/venv/bin:${PATH}"
export HF_HOME=/workspace/cache/hf
export TRANSFORMERS_CACHE=/workspace/cache/hf/hub
export PIP_CACHE_DIR=/workspace/.cache/pip
export VLLM_SKIP_PROFILE_RUN=${VLLM_SKIP_PROFILE_RUN:-1}
export VLLM_PLUGINS=${VLLM_PLUGINS:-}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-SDPA}

# Запуск API. App-директория: /opt/app, модуль: apps.tilt_api:app
exec /workspace/venv/bin/python -m uvicorn apps.tilt_api:app \
  --app-dir /opt/app \
  --host 0.0.0.0 \
  --port 8001

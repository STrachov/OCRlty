FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/hf \
    # Жёстко глушим FA/плагины и профиль ран
    VLLM_ATTENTION_BACKEND=TORCH_SDPA \
    VLLM_USE_FLASH_ATTENTION=0 \
    VLLM_PLUGINS="" \
    VLLM_DISABLE_PLUGINS=1 \
    VLLM_SKIP_PROFILE_RUN=1 \
    PYTHONPATH=/workspace/src

# ТВОЙ wheel (не меняем!)
ARG VLLM_WHEEL_URL="https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-linux_x86_64.whl"

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip \
      git curl ca-certificates tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# исходники твоего API
COPY ./src /workspace/src
# (если есть) твои зависимости, НО: тут не должно быть vllm/flash-attn/xformers/paddlex
# COPY ./requirements-gpu.txt /workspace/requirements-gpu.txt

# Обновим инструменты сборки
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Torch 2.6.0 (CUDA 12.4)
RUN python3.10 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple \
    torch==2.6.0 torchvision==0.21.0

# Базовые пакеты API
RUN python3.10 -m pip install fastapi uvicorn

RUN python3.10 -m pip install --no-cache-dir paddlepaddle-gpu==2.6.1 \
 -i https://www.paddlepaddle.org.cn/whl/cu124

# Ставим vLLM ИЗ ТВОЕГО URL (без всяких шаблонов)
RUN echo "Downloading vLLM wheel from: ${VLLM_WHEEL_URL}" \
 && curl -fL "${VLLM_WHEEL_URL}" -o /tmp/vllm.whl \
 && python3.10 -m pip install /tmp/vllm.whl \
 && rm -f /tmp/vllm.whl


# На всякий: выпилим потенциальные конфликтёры, если затянулись зависимостями
RUN (python3.10 -m pip uninstall -y flash-attn || true) \
 && (python3.10 -m pip uninstall -y xformers || true) \
 && (python3.10 -m pip uninstall -y paddlex || true)

# (опционально) если используешь requirements-gpu.txt:
# RUN python3.10 -m pip install -r /workspace/requirements-gpu.txt

# Мини-проверка на этапе билда
RUN python3.10 - <<'PY'
import vllm, torch
print("vLLM __version__:", getattr(vllm, "__version__", "?"))
print("Torch version:", torch.__version__)
PY

# entrypoint
COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8001
ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/entrypoint.sh"]

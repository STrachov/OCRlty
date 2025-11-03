# CUDA 12.4 runtime, Ubuntu 22.04
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Базовые утилиты + Python 3.10 + tini (для PID 1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git curl ca-certificates tini \
 && rm -rf /var/lib/apt/lists/* \
 && python3.10 -m pip install -U pip

# ---- Torch (CUDA 12.4) ----
RUN python3.10 -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0

# ---- Библиотеки рантайма (минимум для tilt_api) ----
# совместимая пара HF: transformers 4.57.1 + tokenizers 0.23.0
RUN python3.10 -m pip install --no-cache-dir \
    fastapi>=0.120,<1.0 uvicorn[standard]>=0.38 httpx>=0.28 pydantic>=2.5 loguru>=0.7 \
    transformers==4.57.1 tokenizers==0.23.0 \
    huggingface_hub>=0.23,<1.0 sentencepiece>=0.1.99 tiktoken>=0.6

# ---- vLLM колесо (linux_x86_64) ----
ARG VLLM_WHL_URL="https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-linux_x86_64.whl"
ARG VLLM_WHL_SHA256="c0f53b29a7c2b79a86d45fed8770b4164b46dfe5cda5bc4cd375bb86f3335811"
RUN set -eux; \
    F="$(basename "$VLLM_WHL_URL")"; \
    curl -fL "$VLLM_WHL_URL" -o "/tmp/${F}"; \
    echo "${VLLM_WHL_SHA256}  /tmp/${F}" | sha256sum -c -; \
    python3.10 -m pip install --no-deps --no-cache-dir "/tmp/${F}"; \
    rm -f "/tmp/${F}"

# ---- Код приложения ----
WORKDIR /workspace
RUN mkdir -p /workspace/src /workspace/cache/hf
COPY . /workspace/src

# ENV по умолчанию
ENV HF_HOME=/workspace/cache/hf \
    HOST=0.0.0.0 \
    PORT=8001 \
    PYTHONUNBUFFERED=1

# Entrypoint: tini как PID 1 (subreaper)
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 8001

# tini будет PID 1 и передаст сигнализацию в дочерний процесс
ENTRYPOINT ["/usr/bin/tini","-s","--"]
CMD ["/usr/local/bin/entrypoint.sh"]

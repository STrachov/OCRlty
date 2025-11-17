# CUDA 12.4 + cuDNN runtime (Ubuntu 22.04)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/hf \
    TRANSFORMERS_CACHE=/workspace/cache/hf/hub \
    PIP_CACHE_DIR=/workspace/.cache/pip \
    PIP_NO_CACHE_DIR=0 \
    # vLLM окружение
    VLLM_SKIP_PROFILE_RUN=1 \
    VLLM_PLUGINS="" \
    VLLM_ATTENTION_BACKEND=SDPA

# Базовые утилиты + Python 3.10 + tini
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-distutils python3.10-minimal \
      ca-certificates curl git tini && \
    rm -rf /var/lib/apt/lists/*

# pip под Python 3.10 и удобный алиас "python"
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    ln -sf /usr/bin/python3.10 /usr/local/bin/python && \
    python -V && pip -V

# === /opt/venv: "запечённый" venv с Torch+vLLM ===
RUN python -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --upgrade pip wheel

# Torch 2.6.0 (CUDA 12.4) строго из официального индекса
RUN /opt/venv/bin/python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchvision==0.21.0


# Аргументы для твоего колеса vLLM
ARG VLLM_WHEEL_NAME="vllm-0.8.3-cp310-cp310-linux_x86_64.whl"
ARG VLLM_WHEEL_URL="https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/${VLLM_WHEEL_NAME}"
ARG VLLM_WHEEL_SHA256="c0f53b29a7c2b79a86d45fed8770b4164b46dfe5cda5bc4cd375bb86f3335811"

# Ставим vLLM из твоего .whl в /opt/venv (без зависимостей — Torch уже стоит)
RUN curl -L -o "/tmp/${VLLM_WHEEL_NAME}" "${VLLM_WHEEL_URL}" && \
    echo "${VLLM_WHEEL_SHA256}  /tmp/${VLLM_WHEEL_NAME}" | sha256sum -c - && \
    /opt/venv/bin/python -m pip install --no-cache-dir --no-deps -U "/tmp/${VLLM_WHEEL_NAME}" && \
    rm -f "/tmp/${VLLM_WHEEL_NAME}"

# Проект (весь репозиторий) — в /opt/app
WORKDIR /opt/app
COPY . /opt/app

# entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8001
# tini как PID 1 (с -s чтобы не ругался про subreaper)
ENTRYPOINT ["/usr/bin/tini","-s","--","/usr/local/bin/entrypoint.sh"]

# CUDA 12.4 + cuDNN runtime (Ubuntu 22.04)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/hf \
    # vLLM окружение
    VLLM_SKIP_PROFILE_RUN=1 \
    VLLM_PLUGINS="" \
    VLLM_ATTENTION_BACKEND=SDPA \
    # чтобы Python автоматически нашёл /workspace/sitecustomize.py
    PYTHONPATH=/workspace

# Базовые утилиты + Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-distutils python3.10-minimal \
      curl ca-certificates git tini && \
    rm -rf /var/lib/apt/lists/*

# pip под Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Проверка
RUN python3.10 - <<'PY'
import sys,platform
assert sys.version_info[:2]==(3,10), sys.version
print("OK: Python", sys.version, "arch:", platform.machine())
PY

# Создаём venv и используем ТОЛЬКО его далее
RUN python3.10 -m venv /workspace/venv && /workspace/venv/bin/python -m pip install --upgrade pip wheel

ENV PATH="/workspace/venv/bin:${PATH}"

# Устанавливаем Torch 2.6.0 + cu124 под cp310
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 

# ---- vLLM из твоего колеса (cp310) с проверкой sha256 ----
ARG VLLM_WHEEL_NAME="vllm-0.8.3-cp310-cp310-linux_x86_64.whl"
ARG VLLM_WHEEL_URL="https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/${VLLM_WHEEL_NAME}"
ARG VLLM_WHEEL_SHA256="c0f53b29a7c2b79a86d45fed8770b4164b46dfe5cda5bc4cd375bb86f3335811"
 
RUN set -euo; \
    echo "[fetch] ${VLLM_WHEEL_URL}"; \
    curl -L -o "/tmp/${VLLM_WHEEL_NAME}" "${VLLM_WHEEL_URL}"; \
    echo "${VLLM_WHEEL_SHA256}  /tmp/${VLLM_WHEEL_NAME}" | sha256sum -c -; \
    python -m pip install --no-cache-dir --no-deps -U "/tmp/${VLLM_WHEEL_NAME}"; \
    rm -f "/tmp/${VLLM_WHEEL_NAME}"

# Остальные зависимости проекта (без torch/vllm)
WORKDIR /workspace
COPY requirements-gpu.txt /workspace/requirements-gpu.txt
RUN python -m pip install --no-cache-dir -r /workspace/requirements-gpu.txt

# sitecustomize: включится автоматически при старте любого python-процесса
COPY sitecustomize.py /workspace/sitecustomize.py

COPY . /workspace/src

# entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8001
ENTRYPOINT ["/usr/bin/tini","--","/usr/local/bin/entrypoint.sh"]

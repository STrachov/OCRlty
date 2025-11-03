FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1 PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1

# Python 3.10 + базовые утилиты + tini
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    git curl ca-certificates tini bash \
 && rm -rf /var/lib/apt/lists/* \
 && python3.10 -m pip install -U pip

# Torch 2.6.0 (CUDA 12.4) — отдельно, чтобы не зависеть от reqs
RUN python3.10 -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0

# Ставим ЗАВИСИМОСТИ ПРИЛОЖЕНИЯ из requirements-gpu.txt
WORKDIR /workspace
COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN python3.10 -m pip install -r /tmp/requirements-gpu.txt

# vLLM колёсo (linux_x86_64) из твоего релиза
ARG VLLM_WHL_URL="https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-linux_x86_64.whl"
ARG VLLM_WHL_SHA256="c0f53b29a7c2b79a86d45fed8770b4164b46dfe5cda5bc4cd375bb86f3335811"
RUN set -eux; F="$(basename "$VLLM_WHL_URL")"; \
    curl -fL "$VLLM_WHL_URL" -o "/tmp/${F}"; \
    echo "${VLLM_WHL_SHA256}  /tmp/${F}" | sha256sum -c -; \
    python3.10 -m pip install --no-deps "/tmp/${F}"; \
    rm -f "/tmp/${F}"

# Код приложения
RUN mkdir -p /workspace/src /workspace/cache/hf
COPY . /workspace/src

ENV HF_HOME=/workspace/cache/hf HOST=0.0.0.0 PORT=8001
EXPOSE 8001

# Tini = PID 1 (чтобы не было WARN про subreaper)
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/bin/tini","-s","--"]
CMD ["/usr/local/bin/entrypoint.sh"]
# syntax=docker/dockerfile:1.6

########### builder ###########
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda CMAKE_BUILD_PARALLEL_LEVEL=1 MAX_JOBS=1 NINJA_FLAGS="-j1" \
    TORCH_CUDA_ARCH_LIST="8.6"

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip \
      git ca-certificates curl build-essential cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install -U pip wheel setuptools setuptools-scm jinja2 packaging

# >>> Только torch (без torchvision/torchaudio) + фикс typing_extensions
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip install --no-cache-dir typing_extensions==4.12.2 && \
    python3.10 -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0

WORKDIR /opt
RUN git clone --depth=1 --branch v0.8.3 https://github.com/Snowflake-Labs/arctic-tilt.git arctic-tilt

# Собрать wheel их форка vLLM (на нашем torch 2.6.0), без изоляции/доп.зависимостей
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip wheel --no-deps --no-build-isolation -w /wheels /opt/arctic-tilt


########### runtime ###########
FROM python:3.10-slim AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/hf VLLM_GPU_UTIL=0.80

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# >>> Только torch (без torchvision/torchaudio) + фикс typing_extensions
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir typing_extensions==4.12.2 && \
    pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir einops sentencepiece "httpx>=0.27"

COPY --from=builder /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/wheels vllm

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=8s --start-period=40s --retries=20 \
  CMD curl -fsS http://127.0.0.1:8001/v1/models || exit 1

CMD ["/usr/local/bin/entrypoint.sh"]

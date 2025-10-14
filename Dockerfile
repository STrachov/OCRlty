# syntax=docker/dockerfile:1.6

########################################
# Stage 0: builder с CUDA toolkit (nvcc)
########################################
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # ограничим параллелизм (меньше RAM/диск)
    CMAKE_BUILD_PARALLEL_LEVEL=1 \
    MAX_JOBS=1 \
    NINJA_FLAGS="-j1" \
    CUDA_HOME=/usr/local/cuda \
    # для RTX A5000 (SM 8.6); поменяй под свой GPU при желании
    TORCH_CUDA_ARCH_LIST="8.6"

# Python 3.10 + инструменты
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip \
      git ca-certificates curl build-essential cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install -U pip wheel setuptools setuptools-scm jinja2 packaging

# Torch 2.6.0 (cu124) — сборка форка vLLM должна идти против ЭТОГО торча
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Клонируем Arctic-TILT и СБОРКА колесом (без изоляции, без депсов)
WORKDIR /opt
RUN git clone --depth=1 --branch v0.8.3 https://github.com/Snowflake-Labs/arctic-tilt.git arctic-tilt
# Соберём колёса в /wheels (получится в первую очередь vllm-*.whl их форка)
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip wheel --no-deps --no-build-isolation -w /wheels /opt/arctic-tilt

########################################
# Stage 1: runtime (лёгкий)
########################################
FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/hf \
    VLLM_GPU_UTIL=0.80

# Базовые утилиты
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Torch 2.6.0 (cu124) — тот же рантайм, что и в builder
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Лёгкие зависимости рантайма
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install einops sentencepiece "httpx>=0.27"

# Ставим собранный wheel из builder-стейджа (их форк vLLM с поддержкой TILT)
COPY --from=builder /wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-index --find-links=/wheels vllm

# entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=8s --start-period=40s --retries=20 \
  CMD curl -fsS http://127.0.0.1:8001/v1/models || exit 1

CMD ["/usr/local/bin/entrypoint.sh"]

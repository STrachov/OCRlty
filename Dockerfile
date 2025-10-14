# syntax=docker/dockerfile:1.6

########### Stage 0: builder (CUDA toolkit + nvcc) ###########
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    # ограничим параллелизм сборки (меньше RAM/диск)
    CMAKE_BUILD_PARALLEL_LEVEL=1 \
    MAX_JOBS=1 \
    NINJA_FLAGS="-j1" \
    # под RTX A5000 (SM 8.6); при другой карте можно сменить/убрать
    TORCH_CUDA_ARCH_LIST="8.6" \
    # фикс для setuptools_scm при shallow clone
    SETUPTOOLS_SCM_PRETEND_VERSION=0.8.3 \
    # чтобы компоновщик видел cudart из toolkita
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Python 3.10 + инструменты (без cmake из apt!)
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip \
      git ca-certificates curl build-essential ninja-build pkg-config kmod \
    && rm -rf /var/lib/apt/lists/*

# Базовые питон-инструменты + новый CMake и NumPy
RUN python3.10 -m pip install -U pip wheel setuptools setuptools-scm jinja2 packaging \
 && python3.10 -m pip install "cmake>=3.29" "numpy<2.3"

# Torch 2.6.0 (CUDA 12.4). Не подменяем основной индекс, используем extra-index.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip install --no-cache-dir typing_extensions==4.12.2 && \
    python3.10 -m pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0

# Клонируем Arctic-TILT (v0.8.3) и собираем wheel их форка vLLM
WORKDIR /opt
RUN git clone --depth=1 --branch v0.8.3 https://github.com/Snowflake-Labs/arctic-tilt.git arctic-tilt \
 && cd arctic-tilt && (git fetch --unshallow || true) && cd ..

# Собираем колёса в /wheels (главное — vllm-*.whl), без изоляции/доп. deps
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.10 -m pip wheel --no-deps --no-build-isolation -w /wheels /opt/arctic-tilt


########### Stage 1: runtime (лёгкий) ###########
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

# Точно такой же torch 2.6.0 (cu124) — без torchvision/torchaudio
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir typing_extensions==4.12.2 && \
    pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0

# Лёгкие зависимости рантайма
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir einops sentencepiece "httpx>=0.27"

# Ставим собранный wheel их форка vLLM из builder-стейджа
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

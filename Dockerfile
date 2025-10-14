# syntax=docker/dockerfile:1.6
FROM python:3.10-slim

# --- Настройки окружения и сборки ---
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/hf \
    VLLM_GPU_UTIL=0.80 \
    # Ограничим параллелизм сборки, чтобы не съедать всю RAM
    CMAKE_BUILD_PARALLEL_LEVEL=1 \
    MAX_JOBS=1 \
    NINJA_FLAGS="-j1" \
    # Для RTX A5000 (SM 8.6). Если GPU другой, поменяйте значение.
    TORCH_CUDA_ARCH_LIST="8.6"

# --- Базовые утилиты ---
RUN apt-get update && apt-get install -y --no-install-recommends \
      git curl ca-certificates build-essential cmake ninja-build pkg-config \
    && rm -rf /var/lib/apt/lists/*

# --- Базовые python-инструменты ---
RUN python -m pip install -U pip wheel setuptools setuptools-scm jinja2 packaging

# --- Torch 2.6.0 + CUDA 12.4 (официальный индекс PyTorch) ---
# При необходимости зафиксируйте и torchvision/torchaudio под 2.6.0
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# --- (Опционально) ускорение внимания; если не найдётся колесо — не критично ---
# RUN --mount=type=cache,target=/root/.cache/pip pip install "xformers>=0.0.27" || true

# --- Минимальные зависимости рантайма (бинарными колёсами) ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install einops sentencepiece "httpx>=0.27"

# --- Клонируем Arctic-TILT и ставим ИХ форк vLLM ---
# ВАЖНО: --no-build-isolation, чтобы сборка шла на уже установленном torch==2.6.0
RUN git clone --depth=1 --branch v0.8.3 \
      https://github.com/Snowflake-Labs/arctic-tilt.git /opt/arctic-tilt && \
    pip install --no-deps --no-build-isolation /opt/arctic-tilt

# --- Здоровье/порты/запуск ---
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=8s --start-period=40s --retries=20 \
  CMD curl -fsS http://127.0.0.1:8001/v1/models || exit 1

# Запускаем OpenAI-совместимый сервер vLLM (TILT)
# Токен HF передайте как ENV на площадке (HUGGING_FACE_HUB_TOKEN), секреты внутрь образа не кладём.
CMD ["bash", "-lc", "\
python -m vllm.entrypoints.openai.api_server \
 --host 0.0.0.0 --port 8001 \
 --model Snowflake/snowflake-arctic-tilt-v1.3 \
 --dtype float16 \
 --max-model-len 4096 \
 --gpu-memory-utilization ${VLLM_GPU_UTIL} \
 --download-dir ${HF_HOME} \
 --trust-remote-code \
"]

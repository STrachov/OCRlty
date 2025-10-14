# syntax=docker/dockerfile:1.6
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ONLY_BINARY=":all:" \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/hf \
    VLLM_GPU_UTIL=0.80

# Системные утилиты
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Базовые python-инструменты
RUN python -m pip install -U pip wheel setuptools

# (Опционально) ускоритель внимания; можно удалить, если не нужен
# RUN pip install xformers || true

# Тяжёлые зависимости только из бинарных колёс
RUN pip install --only-binary=:all: \
      "vllm==0.8.3" \
      einops sentencepiece httpx>=0.27

# Arctic-TILT (официальная реализация на vLLM 0.8.3)
RUN git clone --depth=1 --branch v0.8.3 \
      https://github.com/Snowflake-Labs/arctic-tilt.git /opt/arctic-tilt && \
    pip install --no-deps /opt/arctic-tilt

# Небольшой entrypoint-скрипт, чтобы подставлять ENV
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=20 \
  CMD curl -fsS http://127.0.0.1:8001/v1/models || exit 1

CMD ["/usr/local/bin/entrypoint.sh"]

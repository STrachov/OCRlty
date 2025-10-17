FROM python:3.10-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/workspace/cache/hf VLLM_GPU_UTIL=0.80

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*

# Torch 2.6.0 (CUDA 12.4) и deps
RUN pip install --no-cache-dir typing_extensions==4.12.2 && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 && \
    pip install --no-cache-dir einops sentencepiece "httpx>=0.27"

# Тянем готовый wheel
ARG VLLM_WHL_URL
ARG VLLM_WHL_SHA256
ADD ${VLLM_WHL_URL} /wheels/vllm.whl
RUN echo "${VLLM_WHL_SHA256}  /wheels/vllm.whl" | sha256sum -c - && \
    pip install --no-cache-dir /wheels/vllm.whl

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=8s --start-period=40s --retries=20 \
  CMD curl -fsS http://127.0.0.1:8001/v1/models || exit 1
CMD ["/usr/local/bin/entrypoint.sh"]

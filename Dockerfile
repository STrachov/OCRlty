FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip git curl && \
    rm -rf /var/lib/apt/lists/* && python3.10 -m pip install -U pip

# Torch/cu124 + xformers
RUN python3.10 -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
    python3.10 -m pip install xformers==0.0.29.post2

# vLLM из вашего релиза (колесо уже есть)
ARG VLLM_WHL_URL=https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-linux_x86_64.whl
ARG VLLM_WHL_NAME=vllm-0.8.3-cp310-cp310-linux_x86_64.whl
ARG VLLM_WHL_SHA256=sha256:c0f53b29a7c2b79a86d45fed8770b4164b46dfe5cda5bc4cd375bb86f3335811
ADD ${VLLM_WHL_URL} /wheels/${VLLM_WHL_NAME}
RUN python3.10 -m pip install /wheels/${VLLM_WHL_NAME} && rm -f /wheels/${VLLM_WHL_NAME}

# утилиты
RUN python3.10 -m pip install uvicorn fastapi httpx loguru

ENV VLLM_ATTENTION_BACKEND=XFORMERS \
    VLLM_USE_FLASH_ATTENTION=0 \
    PYTHONUNBUFFERED=1

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
EXPOSE 8001
CMD ["/usr/local/bin/entrypoint.sh"]
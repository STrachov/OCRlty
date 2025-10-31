FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ARG VLLM_WHEEL_URL="https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-manylinux_2_34_x86_64.whl"


RUN set -eux; \
    FILENAME="$(basename "$VLLM_WHEEL_URL")"; \
    curl -fL "$VLLM_WHEEL_URL" -o "/tmp/${FILENAME}"; \
    PIP_NO_INDEX=1 python -m pip install --no-deps --no-cache-dir "/tmp/${FILENAME}"; \
    rm -f "/tmp/${FILENAME}"


COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8001

# tini как PID 1 и subreaper, чтобы не было ворнингов о зомби
ENTRYPOINT ["/usr/bin/tini","-s","--","/usr/local/bin/entrypoint.sh"]

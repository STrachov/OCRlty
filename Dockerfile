FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

ARG WHEEL_URL="https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-linux_x86_64.whl"


RUN set -eux; \
  curl -fL \
    ${WHEEL_URL} \
    -o /tmp/vllm.whl; \
  PIP_NO_INDEX=1 pip install --no-deps --no-cache-dir /tmp/vllm.whl; \
  rm -f /tmp/vllm.whl

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8001

# tini как PID 1 и subreaper, чтобы не было ворнингов о зомби
ENTRYPOINT ["/usr/bin/tini","-s","--","/usr/local/bin/entrypoint.sh"]

#!/usr/bin/env bash
set -euo pipefail

: "${VLLM_GPU_UTIL:=0.80}"
: "${HF_HOME:=/workspace/cache/hf}"

python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8001 \
  --model Snowflake/snowflake-arctic-tilt-v1.3 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
  --download-dir "${HF_HOME}" \
  --trust-remote-code

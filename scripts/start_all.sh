#!/usr/bin/env bash
set -e

export HF_HOME=${HF_HOME:-/workspace/nv/cache/hf}
mkdir -p "$HF_HOME" /workspace/nv/logs

python -m vllm.entrypoints.openai.api_server \
  --model Snowflake/snowflake-arctic-tilt-v1.3 \
  --host 0.0.0.0 --port 8001 \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  > /workspace/nv/logs/vllm.log 2>&1 &

for i in {1..60}; do
  curl -fsS http://127.0.0.1:8001/v1/models >/dev/null && break
  sleep 2
done

export VLLM_BASE_URL=http://127.0.0.1:8001/v1
exec uvicorn apps.api_gpu.main:app --host 0.0.0.0 --port 8000 --workers 1

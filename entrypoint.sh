#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ID:=Snowflake/snowflake-arctic-tilt-v1.3}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8001}"
: "${HF_HOME:=/workspace/cache/hf}"
: "${VLLM_GPU_UTIL:=0.80}"
: "${TP_SIZE:=1}"

echo "[entrypoint] MODEL_ID=$MODEL_ID  HOST=$HOST PORT=$PORT TP=$TP_SIZE"

# важно для TILT/форка — trust-remote-code
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$VLLM_GPU_UTIL" \
  --trust-remote-code

#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ID:=Snowflake/snowflake-arctic-tilt-v1.3}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8001}"
: "${TP_SIZE:=1}"
: "${VLLM_GPU_UTIL:=0.80}"

# на всякий случай — большой shm
mount -o remount,size=${SHM_SIZE:-4g} /dev/shm 2>/dev/null || true

# режим отладки: только открыть терминал
if [[ "${SLEEP_ON_START:-0}" == "1" ]]; then
  echo "[entrypoint] SLEEP_ON_START=1 -> tail -f /dev/null"
  exec tail -f /dev/null
fi

echo "[entrypoint] MODEL_ID=$MODEL_ID HOST=$HOST PORT=$PORT TP=$TP_SIZE"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$VLLM_GPU_UTIL" \
  --trust-remote-code \
  --log-level "${VLLM_LOGGING_LEVEL:-INFO}"

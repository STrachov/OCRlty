#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ID:=Snowflake/snowflake-arctic-tilt-v1.3}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8001}"
: "${TP_SIZE:=1}"
: "${VLLM_GPU_UTIL:=0.80}"
: "${SHM_MIN_MB:=2048}"   # ремоунт /dev/shm, только если меньше 2 ГБ

# Не уменьшаем большой shm: ремоунтим только если маленький
cur_kb=$(df -k /dev/shm | awk 'NR==2{print $2}')
if [ "${cur_kb:-0}" -lt $((SHM_MIN_MB*1024)) ]; then
  mount -o remount,size=${SHM_SIZE:-4g} /dev/shm 2>/dev/null || true
fi

# Режим отладки: позволить зайти в веб-терминал без запуска сервера
if [[ "${SLEEP_ON_START:-0}" == "1" ]]; then
  echo "[entrypoint] SLEEP_ON_START=1 -> tail -f /dev/null"
  exec tail -f /dev/null
fi

echo "[entrypoint] MODEL_ID=$MODEL_ID HOST=$HOST PORT=$PORT TP=$TP_SIZE"

# Короткая сводка окружения
python - <<'PY'
import torch, vllm, os
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("vllm version", getattr(vllm, "__version__", "unknown"))
print("HF token set:", bool(os.getenv("HUGGING_FACE_HUB_TOKEN")))
PY

# Старт API (без неподдерживаемых флагов)
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --gpu-memory-utilization "$VLLM_GPU_UTIL" \
  --trust-remote-code

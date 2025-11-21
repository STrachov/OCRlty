#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] ===== Arctic-TILT GPU container start ====="

# HF кеш — на /workspace (обычно это персистентный volume на RunPod)
export HF_HOME="${HF_HOME:-/workspace/cache/hf}"
mkdir -p "${HF_HOME}"

# Жёстко: работаем только с xformers
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export VLLM_SKIP_PROFILE_RUN="${VLLM_SKIP_PROFILE_RUN:-1}"

echo "[entrypoint] HF_HOME=${HF_HOME}"
echo "[entrypoint] VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
echo "[entrypoint] VLLM_SKIP_PROFILE_RUN=${VLLM_SKIP_PROFILE_RUN}"
echo "[entrypoint] Using python: $(command -v python)"

# -------------------------- Быстрый probe окружения ---------------------------
python << 'PY'
import os, sys, importlib

print("[probe] sys.version:", sys.version.replace("\n", " "))
print("[probe] sys.executable:", sys.executable)
print("[probe] HF_HOME:", os.getenv("HF_HOME"))
print("[probe] VLLM_ATTENTION_BACKEND:", os.getenv("VLLM_ATTENTION_BACKEND"))
print("[probe] VLLM_SKIP_PROFILE_RUN:", os.getenv("VLLM_SKIP_PROFILE_RUN"))

mods = ("torch", "vllm", "xformers", "fastapi", "uvicorn")
errors = {}

for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"[probe] {mod}: OK")
    except Exception as e:
        errors[mod] = e
        print(f"[probe] {mod}: FAIL -> {e!r}")

# xformers — критичен, без него дальше не идём
if "xformers" in errors:
    print("[probe] xformers is mandatory for Arctic-TILT (no SDPA fallback). Exiting.", file=sys.stderr)
    sys.exit(1)

# torch/vllm тоже критичны — если отвалились, нет смысла продолжать
for critical in ("torch", "vllm"):
    if critical in errors:
        print(f"[probe] {critical} failed to import, aborting startup.", file=sys.stderr)
        sys.exit(1)

PY

# ------------------------------ Старт uvicorn ---------------------------------
echo "[entrypoint] Starting uvicorn apps.tilt_api:app on 0.0.0.0:8001"
exec python -m uvicorn apps.tilt_api:app \
    --app-dir /opt/app \
    --host 0.0.0.0 \
    --port 8001

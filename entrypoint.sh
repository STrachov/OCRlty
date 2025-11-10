#!/usr/bin/env bash
set -euo pipefail

# Пробные выводы окружения
echo "[probe] python: $(python -V)"
echo "[probe] pip: $(pip -V)"
python - <<'PY'
import os, sys
print("[probe] sys.executable:", sys.executable)
print("[probe] sys.path[0:3]:", sys.path[:3])
try:
    import sitecustomize as sc
    print("[probe] sitecustomize loaded from:", getattr(sc, "__file__", "<unknown>"))
except Exception as e:
    print("[probe] sitecustomize import error:", e)
print("[probe] VLLM_SKIP_PROFILE_RUN =", os.getenv("VLLM_SKIP_PROFILE_RUN"))
print("[probe] VLLM_ATTENTION_BACKEND =", os.getenv("VLLM_ATTENTION_BACKEND"))
PY

# Небольшая самопроверка import vllm
python - <<'PY'
try:
    import vllm
    print("[probe] vllm version OK")
except Exception as e:
    import traceback; traceback.print_exc()
    raise SystemExit("[FATAL] vLLM not importable")
PY

# Запуск твоего API
# при необходимости поменяй модуль/путь
exec python -m uvicorn src.apps.tilt_api:app --host 0.0.0.0 --port 8001

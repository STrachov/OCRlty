#!/usr/bin/env bash
set -euo pipefail

# Всегда предпочитаем venv-питон
# VENV_PY="/workspace/venv/bin/python"
# if [[ -x "$VENV_PY" ]]; then
#   export PATH="/workspace/venv/bin:${PATH}"
#   PY_BIN="$VENV_PY"
# else
#   # Фоллбек — но в норме до него не дойдём
#   PY_BIN="$(command -v python || true)"
#   [[ -n "${PY_BIN}" ]] || PY_BIN="$(command -v python3 || true)"
# fi
export PATH=/workspace/venv/bin:$PATH
echo "[probe] python: $(${PY_BIN:-python} -V 2>/dev/null || echo 'not found')"
echo "[probe] pip: $(pip -V || true)"

# Подхватим sitecustomize и покажем откуда он грузится
echo "[probe] sitecustomize:"
${PY_BIN} - <<'PY' || true
import sitecustomize, sys
print(getattr(sitecustomize, "__file__", "??"), file=sys.stdout)
PY

# Логируем критичные переменные среды
echo "[entrypoint] VLLM_PLUGINS='${VLLM_PLUGINS:-}'"
echo "[entrypoint] VLLM_SKIP_PROFILE_RUN=${VLLM_SKIP_PROFILE_RUN:-}"
echo "[entrypoint] VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-}"

# 1) Если том перекрыл /workspace – положим туда код из образа один раз
if [ ! -d /workspace/src ]; then
  echo "[seed] /workspace/src missing -> copy from /app"
  mkdir -p /workspace/src
  cp -a /app/. /workspace/src/
fi


# Запуск API
cd /workspace/src
exec ${PY_BIN} -m uvicorn apps.tilt_api:app --host 0.0.0.0 --port 8001

#!/usr/bin/env bash
set -euo pipefail

# 0) Базовые переменные окружения
export HF_HOME="${HF_HOME:-/workspace/cache/hf}"
# TRANSFORMERS_CACHE устарел — уберём, чтобы не было warning
unset TRANSFORMERS_CACHE || true

# 1) Всегда используем venv на /workspace (персистентный volume на RunPod)
if [[ ! -x /workspace/venv/bin/python ]]; then
  python3.10 -m venv /workspace/venv
  /workspace/venv/bin/python -m pip install -U pip wheel
  
fi

# 2) Ставим runtime-зависимости проекта в /workspace/venv
REQ_FILE="/opt/app/requirements-gpu.txt"
/workspace/venv/bin/python -m pip install --no-cache-dir -r "$REQ_FILE"

# 3) vLLM + torch лежат в /opt/venv → сделаем их видимыми для /workspace/venv
# ВАЖНО: добавляем в КОНЕЦ PYTHONPATH, чтобы сначала брались пакеты из /workspace/venv
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}/opt/venv/lib/python3.10/site-packages"
export PATH="/workspace/venv/bin:${PATH}"

# Форсим SDPA и отключаем xformers
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-SDPA}"
export VLLM_USE_XFORMERS=0

# Подкладываем shim, чтобы vllm мог импортировать vllm.attention.backends.xformers
# и не требовал пакет xformers.* (C++/CUDA расширения), которых нет под torch 2.6+cu124
if [[ ! -f /opt/app/_shims/vllm/attention/backends/xformers.py ]]; then
  mkdir -p /opt/app/_shims/vllm/attention/backends
  cat > /opt/app/_shims/vllm/attention/backends/xformers.py <<'PY'
class XFormersMetadata:
    """Stub used when xformers is unavailable. vLLM will not use it with SDPA."""
    def __init__(self, *args, **kwargs):
        self.available = False
    def __bool__(self):
        return False
PY
fi
# Очень важно: этот путь должен стоять ПЕРЕД путями пакета vllm
export PYTHONPATH="/opt/app/_shims:${PYTHONPATH}"


# 4) Быстрый догон для пропущенных runtime-deps vLLM, если образ собирался с --no-deps
# (pip мгновенно скажет "already satisfied", если уже установлено)
python - <<'PY'
import sys, subprocess
def ensure(mod, spec=None):
    try:
        __import__(mod)
    except Exception:
        pkg = spec or mod
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", pkg])
for mod, spec in [
    ("cachetools","cachetools>=5,<6"),
    # ("einops","einops>=0.7"),  # раскомментируй, если дальше упадёт на einops
]:
    ensure(mod, spec)
PY

# 5) Лог для проверки
python - <<'PY'
import sys, importlib
print("[probe] Python:", sys.version)
for m in ("uvicorn","vllm","torch","cachetools"):
    try:
        importlib.import_module(m)
        print(f"[probe] {m}: OK")
    except Exception as e:
        print(f"[probe] {m}: FAIL -> {e}")
        raise
PY

# 6) Старт API (код лежит под /opt/app, модуль apps.tilt_api:app)
exec python -m uvicorn apps.tilt_api:app --app-dir /opt/app --host 0.0.0.0 --port 8001

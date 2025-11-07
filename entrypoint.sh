#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="/workspace/venv"

echo "[entrypoint] Recreating venv at $VENV_DIR"
rm -rf "$VENV_DIR" || true
python3.10 -m venv --system-site-packages "$VENV_DIR"

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# Гарантируем доступ к твоему коду
export PYTHONPATH="/workspace/src:${PYTHONPATH:-}"

# Создадим .pth + модуль патча, который:
#  1) форсит env для SDPA/disable FA
#  2) глушит TILTModelRunner.profile_run(), чтобы не падало на flash_attn в профайле
python - <<'PY'
import os, sys, site, textwrap, importlib, subprocess

sp = site.getsitepackages()[0]
pth_path = os.path.join(sp, "tilt_startup_patch.pth")
mod_path = os.path.join(sp, "tilt_startup_patch.py")

patch_code = textwrap.dedent(r"""
import os
# Жёсткие флажки до любых импортов vllm
os.environ.setdefault("VLLM_SKIP_PROFILE_RUN", "1")
os.environ.setdefault("VLLM_DISABLE_PLUGINS", "1")
os.environ.setdefault("VLLM_PLUGINS", "")
os.environ.setdefault("VLLM_USE_FLASH_ATTENTION", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")

# После импорта vllm — отключим profile_run у TILT
def _patch_tilt_profile_run():
    try:
        import importlib
        mdl = importlib.import_module("vllm.worker.tilt_model_runner")
        cls = getattr(mdl, "TILTModelRunner", None)
        if cls and hasattr(cls, "profile_run"):
            setattr(cls, "profile_run", lambda self: None)
            print("[tilt_patch] TILTModelRunner.profile_run disabled")
    except Exception as e:
        print("[tilt_patch] fail to patch profile_run:", repr(e))

# Если vllm уже импортировали — патчим сейчас, иначе повесим ленивый хук
try:
    import vllm  # noqa
    _patch_tilt_profile_run()
except Exception:
    import importlib, sys
    class _Hook(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "vllm.worker.tilt_model_runner":
                # когда модуль реально подгрузится — после него выполним патч
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                if spec and spec.loader:
                    orig_exec = spec.loader.exec_module
                    def _wrapped_exec(m):
                        orig_exec(m)
                        # модуль загружен — патчим класс
                        _patch_tilt_profile_run()
                    spec.loader.exec_module = _wrapped_exec
                return spec
            return None
    sys.meta_path.insert(0, _Hook())
""")

os.makedirs(sp, exist_ok=True)
with open(mod_path, "w", encoding="utf-8") as f:
    f.write(patch_code)
with open(pth_path, "w", encoding="utf-8") as f:
    f.write("import tilt_startup_patch\n")

print("[probe] venv site-packages:", sp)
print("[probe] wrote:", pth_path, "and", mod_path)

# Диагностика окружения/пакетов
def show(name: str):
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "show", name], text=True)
        print(out.strip())
    except Exception as e:
        print(f"{name}: <not installed> ({e})")

print("[probe] sys.executable:", sys.executable)
print("[probe] PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("----- pip show vllm -----"); show("vllm")
print("----- pip show torch -----"); show("torch")

# Проверим, что vllm импортируется и патч применяется уже сейчас
try:
    import vllm  # noqa
    print("[probe] vLLM import OK")
    # примем патч немедленно (если модуль уже поднимался ранее)
    try:
        import vllm.worker.tilt_model_runner as tmr
        if hasattr(tmr, "TILTModelRunner") and hasattr(tmr.TILTModelRunner, "profile_run"):
            setattr(tmr.TILTModelRunner, "profile_run", lambda self: None)
            print("[probe] TILTModelRunner.profile_run disabled (eager)")
    except Exception as e:
        print("[probe] eager patch failed:", repr(e))
except Exception as e:
    print("[entrypoint][FATAL] vLLM not importable:", e)
    raise
PY

echo "[entrypoint] vLLM OK — starting API…"
exec python -m uvicorn tilt_api:app --host 0.0.0.0 --port 8001

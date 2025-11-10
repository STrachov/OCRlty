"""
sitecustomize.py
Автоматически подхватывается Python'ом при старте (благодаря PYTHONPATH=/workspace).
Делает два дела:
1) Форсит бэкенд внимания vLLM = SDPA (без Flash-Attn/xformers)
2) Выключает profile_run() у TILT-модели (патчит vLLM), если VLLM_SKIP_PROFILE_RUN=1
"""

import os
import sys
import builtins
import importlib

# Безопасные дефолты
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "SDPA")
os.environ.setdefault("VLLM_SKIP_PROFILE_RUN", "1")

def _patch_tilt_profile_run():
    """Меняем TILTModelRunner.profile_run на no-op при включённом флаге."""
    if os.getenv("VLLM_SKIP_PROFILE_RUN") not in ("1", "true", "True", "YES", "yes"):
        return
    try:
        import vllm.worker.tilt_model_runner as tmr
        if hasattr(tmr, "TILTModelRunner"):
            def _no_profile(self):
                try:
                    from loguru import logger
                    logger.info("sitecustomize: TILT profile_run() skipped")
                except Exception:
                    sys.stderr.write("sitecustomize: TILT profile_run() skipped\n")
                return None
            tmr.TILTModelRunner.profile_run = _no_profile
            sys.stderr.write("sitecustomize: patched TILTModelRunner.profile_run -> no-op\n")
    except Exception as e:
        # молча — патч сработает через import hook ниже
        pass

# Пытаемся пропатчить сразу (если vllm уже доступен)
_patch_tilt_profile_run()

# А также ставим простой import-hook, чтобы патч применился при ПЕРВОМ импорте vllm
_orig_import = builtins.__import__
def _hooked_import(name, *args, **kwargs):
    mod = _orig_import(name, *args, **kwargs)
    if name.startswith("vllm"):
        _patch_tilt_profile_run()
    return mod
builtins.__import__ = _hooked_import

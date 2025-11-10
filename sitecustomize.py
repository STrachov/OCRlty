"""
sitecustomize.py
Автозагружается Python’ом до всего остального.
Здесь мы:
  1) форсим безопасные для vLLM режимы (SDPA, без flash-attn),
  2) отключаем TILT profile_run монкипатчем,
  3) печатаем диагностическую строку чтобы видеть, что файл реально подхватился.
"""
import os
import sys

print(f"[sitecustomize] Loaded sitecustomize from: {__file__}")

# 1) среда для внимания и профайлинга
os.environ.setdefault("VLLM_USE_FLASH_ATTENTION", "0")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TORCH_SDPA")
os.environ.setdefault("VLLM_SKIP_PROFILE_RUN", "1")
# На всякий случай глушим любые плагины
os.environ.setdefault("VLLM_PLUGINS", "")

# 2) монкипатч: отключить profile_run у TILT, если он есть
try:
    # импортируем по месту, чтобы не грузить лишнее, но достаточно рано
    import importlib
    tmr = importlib.import_module("vllm.worker.tilt_model_runner")  # type: ignore
    ModelRunner = getattr(tmr, "ModelRunner", None)
    if ModelRunner and hasattr(ModelRunner, "profile_run"):
        def _no_profile_run(self, *args, **kwargs):
            try:
                # Логируем только один раз, без зависимостей от loguru
                sys.stderr.write("[sitecustomize] TILT profile_run is DISABLED by sitecustomize.\n")
            except Exception:
                pass
            return None
        setattr(ModelRunner, "profile_run", _no_profile_run)
        print("[sitecustomize] Patched vllm.worker.tilt_model_runner.ModelRunner.profile_run -> no-op")
except Exception as e:
    print(f"[sitecustomize][WARN] Could not patch TILT profile_run: {e}")

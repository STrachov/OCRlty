# auto-generated: disable TILT warmup when asked
import os, importlib, sys

# RunPod UI иногда сохраняет пустое значение как '""'
if os.environ.get("VLLM_PLUGINS", "") == '""':
    os.environ["VLLM_PLUGINS"] = ""

patched = False
if os.environ.get("VLLM_SKIP_PROFILE_RUN", "").lower() in ("1", "true", "yes", "on"):
    try:
        tmr = importlib.import_module("vllm.worker.tilt_model_runner")
        def _noop(self):  # no-op вместо profile_run
            return
        # Найдём все классы модуля, у которых есть profile_run, и перепишем метод
        for _, obj in vars(tmr).items():
            if isinstance(obj, type) and hasattr(obj, "profile_run"):
                try:
                    setattr(obj, "profile_run", _noop)
                    patched = True
                except Exception:
                    pass
    except Exception as e:
        print("[sitecustomize] patch error:", repr(e), file=sys.stderr)

print(f"[sitecustomize] VLLM_SKIP_PROFILE_RUN={os.environ.get('VLLM_SKIP_PROFILE_RUN')} patched={patched}")

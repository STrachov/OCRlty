"""
sitecustomize.py

Автоматически подхватывается Python'ом при старте, если лежит на sys.path.
У нас:
- Dockerfile копирует его в /opt/app/sitecustomize.py
- WORKDIR=/opt/app и PYTHONPATH включает /opt/app

Единственная задача:
- если VLLM_SKIP_PROFILE_RUN = 1/true/yes -> подменить
  TILTModelRunner.profile_run() на no-op, чтобы не делать тяжёлый прогрев.
"""

import os
import sys


def _patch_tilt_profile_run() -> None:
    val = os.getenv("VLLM_SKIP_PROFILE_RUN")
    if val not in ("1", "true", "True", "yes", "YES"):
        return

    try:
        from vllm.worker import tilt_model_runner as tmr  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"sitecustomize: cannot import vllm.worker.tilt_model_runner: {exc}\n")
        return

    runner_cls = getattr(tmr, "TILTModelRunner", None)
    if runner_cls is None:
        sys.stderr.write("sitecustomize: TILTModelRunner not found on tilt_model_runner\n")
        return

    def _no_profile(self) -> None:  # noqa: ANN001
        # Логируем, что прогрев пропущен
        try:
            from loguru import logger  # type: ignore[import]
            logger.info("sitecustomize: TILT profile_run() skipped")
        except Exception:
            sys.stderr.write("sitecustomize: TILT profile_run() skipped\n")
        return None

    runner_cls.profile_run = _no_profile  # type: ignore[assignment]
    sys.stderr.write("sitecustomize: patched TILTModelRunner.profile_run -> no-op\n")


try:
    _patch_tilt_profile_run()
except Exception as exc:  # noqa: BLE001
    sys.stderr.write(f"sitecustomize: failed to patch TILTModelRunner.profile_run: {exc}\n")

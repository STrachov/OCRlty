"""
sitecustomize для Arctic-TILT контейнера.

Цель:
- если установлена VLLM_SKIP_PROFILE_RUN=1, отключить profile_run()
  для TILT-моделей в vLLM, чтобы не падать на кривом тестовом прогоне.

Работает для:
- старого TILTModelRunner (если появится)
- EncDecModelRunner с architecture == "TiltModel"
"""

import os
import sys


def _patch_vllm_profile_run() -> None:
    flag = os.getenv("VLLM_SKIP_PROFILE_RUN")
    if flag not in ("1", "true", "True", "yes", "YES"):
        return

    # 1) Патч для потенциального TILTModelRunner (task="tilt_generate")
    try:
        from vllm.worker import tilt_model_runner as tmr  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"sitecustomize: cannot import vllm.worker.tilt_model_runner: {exc}\n")
    else:
        runner_cls = getattr(tmr, "TILTModelRunner", None)
        if runner_cls is not None:
            def _no_profile_cls(self) -> None:  # noqa: ANN001, D401
                """No-op TILTModelRunner.profile_run()."""
                sys.stderr.write("sitecustomize: TILTModelRunner.profile_run() skipped\n")
                return None

            try:
                runner_cls.profile_run = _no_profile_cls  # type: ignore[assignment]
                sys.stderr.write(
                    "sitecustomize: patched TILTModelRunner.profile_run -> no-op\n"
                )
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(
                    f"sitecustomize: failed to patch TILTModelRunner.profile_run: {exc}\n"
                )
        else:
            sys.stderr.write("sitecustomize: TILTModelRunner not found on tilt_model_runner\n")

    # 2) Патч EncDecModelRunner.profile_run для TiltModel (task='generate' с TiltModel)
    try:
        from vllm.worker import enc_dec_model_runner as edm  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"sitecustomize: cannot import vllm.worker.enc_dec_model_runner: {exc}\n")
        return

    encdec_cls = getattr(edm, "EncDecModelRunner", None)
    if encdec_cls is None:
        sys.stderr.write(
            "sitecustomize: EncDecModelRunner not found on enc_dec_model_runner\n"
        )
        return

    original_profile_run = getattr(encdec_cls, "profile_run", None)
    if original_profile_run is None:
        sys.stderr.write(
            "sitecustomize: EncDecModelRunner.profile_run not found, nothing to patch\n"
        )
        return

    def _patched_profile_run(self, *args, **kwargs):  # noqa: ANN001, D401
        """Wrapper над EncDecModelRunner.profile_run для TiltModel.

        Если модель имеет architecture == 'TiltModel', просто пропускаем
        profile_run, чтобы не падать на некорректных тестовых вводах.
        Для всех остальных моделей вызываем оригинальную реализацию.
        """
        arch = None
        try:
            model_config = getattr(self, "model_config", None)
            arch = getattr(model_config, "architecture", None)
        except Exception:  # noqa: BLE001
            # Если не смогли достать архитектуру, лучше не ломать поведение.
            pass

        if arch == "TiltModel":
            sys.stderr.write(
                "sitecustomize: EncDecModelRunner.profile_run() skipped for TiltModel\n"
            )
            return None

        # Для всех прочих моделей работаем как обычно.
        return original_profile_run(self, *args, **kwargs)

    try:
        encdec_cls.profile_run = _patched_profile_run  # type: ignore[assignment]
        sys.stderr.write(
            "sitecustomize: patched EncDecModelRunner.profile_run for TiltModel\n"
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            f"sitecustomize: failed to patch EncDecModelRunner.profile_run: {exc}\n"
        )


try:
    _patch_vllm_profile_run()
except Exception as exc:  # noqa: BLE001
    sys.stderr.write(f"sitecustomize: unexpected error during patching: {exc}\n")

"""
sitecustomize для Arctic-TILT контейнера.

Цель:
- если установлена VLLM_SKIP_PROFILE_RUN=1, отключить profile_run()
  для энкодер-декодерных моделей в vLLM (в т.ч. TiltModel),
  чтобы не падать на некорректном тестовом прогоне.

Подход:
- пробуем пропатчить TILTModelRunner (если есть),
- плюс ИЩЕМ в vllm.worker.enc_dec_model_runner любой класс
  с методом profile_run и execute_model и глушим profile_run
  вообще (глобально).
"""

import os
import sys
import inspect


def _patch_vllm_profile_run() -> None:
    flag = os.getenv("VLLM_SKIP_PROFILE_RUN")
    if flag not in ("1", "true", "True", "yes", "YES"):
        return

    # 1) Патч для потенциального TILTModelRunner (старые варианты vLLM)
    try:
        from vllm.worker import tilt_model_runner as tmr  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"sitecustomize: cannot import vllm.worker.tilt_model_runner: {exc}\n")
    else:
        runner_cls = getattr(tmr, "TILTModelRunner", None)
        if runner_cls is not None:
            def _no_profile_tilt(self) -> None:  # noqa: ANN001, D401
                """No-op TILTModelRunner.profile_run()."""
                sys.stderr.write("sitecustomize: TILTModelRunner.profile_run() skipped\n")
                return None

            try:
                runner_cls.profile_run = _no_profile_tilt  # type: ignore[assignment]
                sys.stderr.write(
                    "sitecustomize: patched TILTModelRunner.profile_run -> no-op\n"
                )
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(
                    f"sitecustomize: failed to patch TILTModelRunner.profile_run: {exc}\n"
                )
        else:
            sys.stderr.write("sitecustomize: TILTModelRunner not found on tilt_model_runner\n")

    # 2) Глобальный патч profile_run для энкодер-декодерного раннера
    try:
        from vllm.worker import enc_dec_model_runner as edm  # type: ignore[import]
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"sitecustomize: cannot import vllm.worker.enc_dec_model_runner: {exc}\n")
        return

    candidates = []

    for name, obj in vars(edm).items():
        if not inspect.isclass(obj):
            continue
        if hasattr(obj, "profile_run") and hasattr(obj, "execute_model"):
            candidates.append((name, obj))

    if not candidates:
        sys.stderr.write(
            "sitecustomize: no suitable classes with profile_run/execute_model "
            "found in enc_dec_model_runner\n"
        )
        return

    def _patched_profile_run(self, *args, **kwargs):  # noqa: ANN001, D401
        """Глобально выключенный profile_run для enc-dec моделей.
        Для TiltModel это обязательно, иначе падает на forward().
        Для остальных — просто пропускаем профилирование (ok для нашего случая).
        """
        sys.stderr.write(
            "sitecustomize: EncDec profile_run() skipped (global patch)\n"
        )
        return None

    for name, cls in candidates:
        try:
            original = getattr(cls, "profile_run", None)
            if original is None:
                continue
            cls.profile_run = _patched_profile_run  # type: ignore[assignment]
            sys.stderr.write(
                f"sitecustomize: patched {name}.profile_run -> no-op\n"
            )
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(
                f"sitecustomize: failed to patch {name}.profile_run: {exc}\n"
            )


try:
    _patch_vllm_profile_run()
except Exception as exc:  # noqa: BLE001
    sys.stderr.write(f"sitecustomize: unexpected error during patching: {exc}\n")

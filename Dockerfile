FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# ---- Torch cu124 (через официальный индекс) ----
RUN python3.10 -m pip install --upgrade --no-cache-dir pip \
    && python3.10 -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0+cu124" "triton==3.2.0"

# ---- Прикладные зависимости проекта ----
COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN python3.10 -m pip install --no-cache-dir -r /tmp/requirements-gpu.txt

# ---- vLLM wheel (ставим с deps) ----
ARG VLLM_WHL_URL="https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-linux_x86_64.whl"
ARG VLLM_WHL_SHA256="c0f53b29a7c2b79a86d45fed8770b4164b46dfe5cda5bc4cd375bb86f3335811"
RUN set -eux; \
    F="$(basename "$VLLM_WHL_URL")"; \
    curl -fL "$VLLM_WHL_URL" -o "/tmp/${F}"; \
    echo "${VLLM_WHL_SHA256}  /tmp/${F}" | sha256sum -c -; \
    python3.10 -m pip install --no-cache-dir "/tmp/${F}"; \
    rm -f "/tmp/${F}"

# ---- sitecustomize: фикс profile_run для TILT при флаге ----
RUN python3.10 - <<'PY'
from pathlib import Path
p = Path('/usr/local/lib/python3.10/dist-packages/sitecustomize.py')
p.write_text('''# auto-generated
import os, importlib
# Если в UI ввели две кавычки, нормализуем в пустую строку:
if os.environ.get("VLLM_PLUGINS","") == '""':
    os.environ["VLLM_PLUGINS"] = ""
# По флагу вырубаем "тёплый" прогон TILT (vLLM 0.8.3)
if os.environ.get("VLLM_SKIP_PROFILE_RUN","") in ("1","true","yes","on"):
    try:
        tmr = importlib.import_module("vllm.worker.tilt_model_runner")
        def _no_profile(self):
            print("[sitecustomize] Skipping TILT profile_run (VLLM_SKIP_PROFILE_RUN=1)")
            return
        setattr(tmr.TiltModelRunner, "profile_run", _no_profile)
    except Exception as e:
        print("[sitecustomize] Could not patch TiltModelRunner.profile_run:", e)
''')
print("Created", p)
PY

# ---- sanity-check, чтобы падать на этапе сборки, а не в поде ----
RUN python3.10 - <<'PY'
from packaging.version import Version
import transformers, tokenizers, vllm, msgspec, cachetools, torch
print("transformers", transformers.__version__,
        "tokenizers", tokenizers.__version__,
        "vllm", getattr(vllm,"__version__","?"),
        "torch", torch.__version__,
        "msgspec", msgspec.__version__,
        "cachetools", cachetools.__version__)
assert Version("0.22") <= Version(tokenizers.__version__) < Version("0.24")
PY

# ---- Код приложения ----
RUN mkdir -p /workspace/src /workspace/cache/hf
COPY . /workspace/src

# ---- ENV/порт ----
ENV HF_HOME=/workspace/cache/hf \
    HOST=0.0.0.0 \
    PORT=8001 \
    VLLM_PLUGINS=

EXPOSE 8001

# ---- Entrypoint: tini как PID1 + наш скрипт ----
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/bin/tini","-s","--"]
CMD ["/usr/local/bin/entrypoint.sh"]

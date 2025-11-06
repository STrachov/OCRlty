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

COPY sitecustomize.py /workspace/src/sitecustomize.py

# Гарантируем, что /workspace/src виден Python'у до старта uvicorn
ENV PYTHONPATH=/workspace/src:$PYTHONPATH

# (опционально) sanity-check прямо на сборке, что sitecustomize исполнился и патчится
# Здесь выставим флаг только на время проверки
RUN VLLM_SKIP_PROFILE_RUN=1 python3.10 - <<'PY'
import importlib
import vllm
import vllm.worker.tilt_model_runner as tmr
patched_any = []
for _, obj in vars(tmr).items():
    if isinstance(obj, type) and hasattr(obj, "profile_run"):
        fn = getattr(obj, "profile_run")
        # наш _noop без констант и с одним аргументом self — простейшая проверка
        patched_any.append(getattr(fn, "__code__", None) and fn.__code__.co_argcount == 1)
print("[build-check] vLLM:", getattr(vllm, "__version__", "?"), "tilt patched any:", any(patched_any))
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

ENV VLLM_ATTENTION_BACKEND=TORCH \
    VLLM_USE_FLASH_ATTENTION=0 \
    VLLM_USE_TRITON_FLASH_ATTN=0 

EXPOSE 8001

# ---- Entrypoint: tini как PID1 + наш скрипт ----
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/bin/tini","-s","--"]
CMD ["/usr/local/bin/entrypoint.sh"]

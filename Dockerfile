# Базовый образ: CUDA 12.4 + cuDNN, Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/cache/hf \
    PIP_NO_CACHE_DIR=1 \
    # мы хотим всегда xformers
    VLLM_ATTENTION_BACKEND=XFORMERS \
    VLLM_PLUGINS="" \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    # profile_run у TILT отключаем через sitecustomize
    VLLM_SKIP_PROFILE_RUN=1

# ----------------------- 1. Базовая система + Python 3.10 ---------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        git \
        curl \
        ca-certificates \
        tini \
        build-essential \
        pkg-config \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Каталоги под кеши HF и pip (RunPod volume обычно монтится в /workspace)
RUN mkdir -p /workspace/cache/hf /workspace/.cache/pip

# ----------------------------- 2. Код проекта ---------------------------------
WORKDIR /opt/app
COPY . /opt/app

# sitecustomize лежит в корне проекта, Python увидит его по PYTHONPATH/WORKDIR
# (если в репо он уже есть — COPY выше его тоже привёз)

# ---------------------- 3. venv + установка зависимостей ----------------------
# Весь питоновский стек ставим ЗДЕСЬ, а не в entrypoint — чтобы поды стартовали быстро
RUN python3.10 -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip wheel \
 # ставим TORCH / TORCHVISION / TORCHAUDIO
 && /opt/venv/bin/pip install \
      torch==2.6.0 \
      torchvision==0.21.0 \
      #torchaudio==2.6.0 \
      --index-url https://download.pytorch.org/whl/cu124 \
 # requirements-gpu.txt: torch 2.6.0+cu124, HF, FastAPI, opencv и прочий стек
 && /opt/venv/bin/pip install -r requirements-gpu.txt \
 # vLLM и xformers — готовые колёса из GitHub Release
 && /opt/venv/bin/pip install --no-deps \
      https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-linux_x86_64.whl \
      https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/xformers-0.0.29.post2-cp310-cp310-manylinux_2_28_x86_64.whl

RUN /opt/venv/bin/pip uninstall -y paddlex || true

# В рантайме всегда используем /opt/venv/bin/python и видим /opt/app/sitecustomize.py
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/app:${PYTHONPATH:-}

# ----------------------------- 4. entrypoint ----------------------------------
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8001
# tini как PID 1 (с -s чтобы не ругался про subreaper)
ENTRYPOINT ["/usr/bin/tini","-s","--","/usr/local/bin/entrypoint.sh"]

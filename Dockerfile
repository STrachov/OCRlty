FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Базовые переменные окружения
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/cache/hf \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8001 \
    # Жёстко запрещаем flash-attn и форсим SDPA
    VLLM_USE_FLASH_ATTENTION=0 \
    VLLM_ATTENTION_BACKEND=TORCH_SDPA \
    # Пробуем отключить прогон профайла у TILT (а также делаем монкипатчем в sitecustomize)
    VLLM_SKIP_PROFILE_RUN=1

# Системные зависимости (минимум, без графических тулов)
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      tini git ca-certificates curl \
      libglib2.0-0 libgl1 libxrender1 libxext6 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Рабочие каталоги
RUN mkdir -p /workspace/src /workspace/cache/hf
WORKDIR /workspace

# Ставим/обновляем pip заранее
RUN python -m pip install --upgrade pip setuptools wheel

# Подкладываем sitecustomize.py именно в conda-путь Python этого образа
# (в этих образах Python живёт в /opt/conda/lib/python3.10/)
COPY sitecustomize.py /opt/conda/lib/python3.10/sitecustomize.py

# Ставим зависимости проекта (кроме vLLM/flash-attn/xformers/paddlex — их здесь нет)
COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-gpu.txt

# ⚠️ Ставим твой vLLM ТОЧНО из твоего релиза и БЕЗ зависимостей
# Torch уже есть в базе → исключаем любые перетяжки
RUN python -m pip install --no-cache-dir --no-deps -U \
  https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-linux_x86_64.whl

# Кладем весь репозиторий (чтобы не промахнуться с путём src)
# Если репо тяжёлое — можете сузить до COPY ./src и нужных файлов
COPY . /workspace

# Делаем entrypoint исполняемым
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8001

# init-процесс для корректных сигналов
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/usr/local/bin/entrypoint.sh"]

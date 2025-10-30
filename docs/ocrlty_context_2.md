# OCRlty — Контекст и план (GPU: Arctic‑TILT на vLLM 0.8.3, CUDA 12.4, Torch 2.6.0)

## TL;DR
- Переходим на **кастомный сервер `tilt_api.py`** (Uvicorn, порт **:8001**) вместо стандартного `vllm.entrypoints.openai.api_server` — это устраняет ошибки `TiltModel.forward(... encoder_chunk_ids ...)` и позволяет явно указать **задачу `tilt_generate`**.
- **Бэкенд внимания:** используем **XFormers**. В рантайме выставляем:
  - `VLLM_ATTENTION_BACKEND=XFORMERS`
  - `VLLM_USE_FLASH_ATTENTION=0`
- **Сборка:** vLLM **0.8.3** собран отдельно в wheel (CPython 3.10, Linux x86_64), выкладывается в **GitHub Releases**. В прод‑образе этот wheel подтягивается по URL + проверка **SHA256**.
- **База образа:** CUDA **12.4** (runtime), Python **3.10**, **Torch 2.6.0+cu124**, **xformers 0.0.29.post2**. 
- **Prod на RunPod:** один контейнер, где поднимаем **`tilt_api.py` (:8001)**. (Второй API с OCR на CPU можно добавить позже, но TILT уже работает стабильно отдельно.)
- **SHM:** в RunPod оставляем дефолтный **/dev/shm ≈ 24ГБ** (не занижаем, это важно для FA/xformers). 
- **Отладка старта:** переменная `SLEEP_ON_START=1` переведёт контейнер в режим `tail -f /dev/null`, чтобы подключиться и проверить окружение.

---

## 1) Архитектура (минимально‑жизнеспособная для TILT)
**Один под RunPod → один контейнер → один процесс:**
- **`tilt_api.py` (Uvicorn, :8001)** — поднимает vLLM `LLM()` с `task=tilt_generate`, dtype=float16, TP=1; даёт OpenAI‑совместимый endpoint `/v1/chat/completions`.

> Позже можно добавить второй процесс (FastAPI OCR на :8000) и сделать полноценный pipeline OCR → TILT. Текущая версия документа фокусируется на стабильном запуске **TILT**.

---

## 2) Версии/стек
- **CUDA:** 12.4 (runtime)
- **Python:** 3.10
- **PyTorch:** 2.6.0+cu124 (`--extra-index-url https://download.pytorch.org/whl/cu124`)
- **xformers:** 0.0.29.post2 (совместима с Torch 2.6.0 cu124)
- **vLLM:** 0.8.3 (wheel, собранный из форка Snowflake‑Labs/arctic‑tilt v0.8.3)
- **HF экосистема:** `huggingface_hub`, `tokenizers`, `sentencepiece` и т.д.

---

## 3) Колесо vLLM 0.8.3 (форк Arctic‑TILT) — сборка и выкладка
- Сборка делалась на выделенной сервере (16 ядер, 64 ГБ RAM, быстрый NVMe). Итоговый wheel ≈ **400–420 МБ**. Процесс сборки подробно описан в файле vllm_wheel_creation.txt
- Колесо кладём в **GitHub Releases** своего репозитория (тег `tilt-vllm-cu124-py310-torch26`).

  
---

## 4) GitHub Actions — заметки по сборке
- Использовать **buildx** и **QEMU** не требуется (linux/amd64).
- Чистить диск в CI перед установкой Torch (удаление `pip`‑кэшей, `rm -rf /usr/local/cuda-*/compat/*` не трогаем).
- Сохранять артефактом **wheel** vLLM и выкладывать в **Releases** (имя файла без пробелов, полное имя колеса).
- Здесь впервые задаются:
  IMAGE: ghcr.io/strachov/arctic-tilt:cu124-py310-torch26-v0.8.3
  VLLM_WHL_URL: https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-linux_x86_64.whl
  VLLM_WHL_SHA256
---

## 5) RunPod — параметры пода
  **Image:** `ghcr.io/strachov/arctic-tilt:cu124-py310-torch26-v0.8.3`

  **Ports:**
  - 8001,8000

  **Env:**
  ```
  #опционально (все основные переменные окружения и их значения, используемые в коде)
  #VLLM_WHL_URL=https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-linux_x86_64.whl
  #VLLM_WHL_NAME=vllm-0.8.3-cp310-cp310-linux_x86_64.whl
  #VLLM_WHL_SHA256=sha256:c0f53b29a7c2b79a86d45fed8770b4164b46dfe5cda5bc4cd375bb86f3335811

  #VLLM_ATTENTION_BACKEND=XFORMERS
  #VLLM_USE_FLASH_ATTENTION=0

  #GIT_URL=https://github.com/STrachov/OCRlty.git
  #GIT_BRANCH=main
  #HF_HOME=/workspace/cache/hf
  #PORT_VLLM=8001
  #PIP_FIND_LINKS=/workspace/wheelhouse
  #PIP_NO_INDEX:=1

  #TILT_MODEL=Snowflake/snowflake-arctic-tilt-v1.3
  #TILT_DTYPE=float16
  #TILT_TP=1 (TP_SIZE)
  #TILT_MAX_MODEL_LEN=16384 (MAX_MODEL_LEN)

  ```

  **Volumes (Network Volume):**
  - `/workspace/cache/hf` — кэши моделей HF (ускоряет рестарты)

  **Command (вариант А — обычный старт):**
  ```
  оставляем пустым
  ```
  **Command (вариант B — отладка):**
  ```
  tail -f /dev/null
  ---

## 6) Dockerfile (prod)

  ```dockerfile
  FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

  ARG DEBIAN_FRONTEND=noninteractive
  RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip git curl && \
      rm -rf /var/lib/apt/lists/* && python3.10 -m pip install -U pip

  # Torch/cu124 + xformers
  RUN python3.10 -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 && \
      python3.10 -m pip install xformers==0.0.29.post2

  # vLLM из вашего релиза (колесо уже есть)
  ARG VLLM_WHL_URL=https://github.com/STrachov/OCRlty/releases/download/tilt-vllm-cu124-py310-torch26/vllm-0.8.3-cp310-cp310-linux_x86_64.whl
  ARG VLLM_WHL_NAME=vllm-0.8.3-cp310-cp310-linux_x86_64.whl
  ARG VLLM_WHL_SHA256=sha256:c0f53b29a7c2b79a86d45fed8770b4164b46dfe5cda5bc4cd375bb86f3335811
  ADD ${VLLM_WHL_URL} /wheels/${VLLM_WHL_NAME}
  RUN python3.10 -m pip install /wheels/${VLLM_WHL_NAME} && rm -f /wheels/${VLLM_WHL_NAME}

  # утилиты
  RUN python3.10 -m pip install uvicorn fastapi httpx loguru

  ENV VLLM_ATTENTION_BACKEND=XFORMERS \
      VLLM_USE_FLASH_ATTENTION=0 \
      PYTHONUNBUFFERED=1

  COPY entrypoint.sh /usr/local/bin/entrypoint.sh
  RUN chmod +x /usr/local/bin/entrypoint.sh
  EXPOSE 8001
  CMD ["/usr/local/bin/entrypoint.sh"]

  ```

  **Важно:**
  - Файл с колесом нужно называть полным именем wheel (не `vllm.whl`), иначе pip ругается: *"not a valid wheel filename"*.
  - чистим кэши и не используем `--cache-dir` → экономим диск в CI.

  ---

## 7) entrypoint.sh
  #!/usr/bin/env bash
  set -euo pipefail

  : "${GIT_URL:=https://github.com/STrachov/OCRlty.git}"
  : "${GIT_BRANCH:=main}"
  : "${HF_HOME:=/workspace/cache/hf}"
  : "${PORT_VLLM:=8001}"
  : "${PIP_FIND_LINKS:=/workspace/wheelhouse}"
  : "${PIP_NO_INDEX:=1}"

  mkdir -p /workspace/src /workspace/wheelhouse "$HF_HOME"

  #код (clone/pull)
  if [ ! -d /workspace/src/.git ]; then
    git clone --branch "$GIT_BRANCH" --depth 1 "$GIT_URL" /workspace/src
  else
    git -C /workspace/src fetch origin "$GIT_BRANCH" --depth 1
    git -C /workspace/src checkout "$GIT_BRANCH"
    git -C /workspace/src reset --hard "origin/$GIT_BRANCH"
  fi

  #окружение (персистентный venv + системные site-packages)
  if [ ! -d /workspace/venv ]; then
    python3.10 -m venv /workspace/venv --system-site-packages
    . /workspace/venv/bin/activate
    export PIP_NO_INDEX PIP_FIND_LINKS
    # лёгкие зависимости приложения
    if [ -f /workspace/src/requirements-gpu.txt ]; then
      pip install -U pip && pip install -r /workspace/src/requirements-gpu.txt
    fi
  else
    . /workspace/venv/bin/activate
  fi

  #запуск TILT-сервера (пример: apps/tilt_api.py)
  cd /workspace/src
  python -m uvicorn apps.tilt_api:app --host 0.0.0.0 --port "$PORT_VLLM"

## 8) `tilt_api.py` (финальная версия)




## 9) Траблшутинг
- **`address already in use :8001`** — остановить прежний сервер:
  ```bash
  pkill -f "uvicorn tilt_api:app" || true
  ```
- **`NotImplementedError: XFormers and Flash‑Attention …`** — убедиться, что:
  ```bash
  python - <<'PY'
  import xformers, xformers.ops as xo; import torch
  print('xformers', xformers.__version__, 'torch', torch.__version__)
  print('has MEA:', hasattr(xo, 'memory_efficient_attention_forward'))
  PY
  ```
  И что выставлены `VLLM_ATTENTION_BACKEND=XFORMERS`, `VLLM_USE_FLASH_ATTENTION=0`.
- **`TiltModel.forward() missing ...`** — это признак старта через `vllm.entrypoints.openai.api_server` без `tilt_generate`. Использовать **`tilt_api.py`** (см. выше).
- **GH Actions: `No space left on device`** — отключить `pip`‑кэш, чистить `apt`‑листы, собирать в multi‑stage, удалять временные файлы.

---

## 10) Дальше (опционально)
- Добавить второй процесс (FastAPI OCR CPU :8000) и объединить в один контейнер через `scripts/entrypoint.sh` (vLLM → wait → API).
- Ограничить OCR‑параллелизм семафором, держать один `httpx.AsyncClient` к :8001.
- Вынести общие конфиги в `configs/default.yaml` и переменные в `.env`.
- Интегрировать e2e‑тесты против :8001 (минимум smoke на `/v1/chat/completions`).

# Контекст проекта (обновлён: Docker Desktop + WSL2)

**Цель:** R&D → сервис OCR/KIE для чеков и инвойсов (Upwork-ориентированный dev-API без клиентской части: только backend/API).

---

## Поддерживаемые окружения

- **Локально (Windows 10 + Docker Desktop + WSL2, CPU):** разработка, мок-инференс (без GPU).
- **Облако (RunPod, A5000 GPU):** реальный сервинг Arctic‑TILT через vLLM + FastAPI.

---

## Быстрый старт на Windows 10 (WSL2 + Docker Desktop)

1) **Включи WSL2** (PowerShell администратора):
   ```powershell
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   wsl --set-default-version 2
   ```
2) **Поставь дистрибутив** (рекомендуем: Ubuntu 22.04) из Microsoft Store. Проверка:
   ```powershell
   wsl -l -v
   ```
   Убедись, что твой дистрибутив в режиме **VERSION 2**.
3) **Установи Docker Desktop** и включи:
   - *Settings → General*: Use the WSL 2 based engine.
   - *Settings → Resources → WSL Integration*: включи интеграцию для твоего Ubuntu.
   - (опц.) *Settings → Resources → File Sharing*: расшарь проектный диск (C:).
4) **(Опц. GPU в WSL2)** Для локальной GPU-нагрузки нужны: NVIDIA драйвер для Windows (WDDM), включение *Settings → Resources → GPU*. В нашем случае локальная GPU слабая — **не используем**.

Проверка из WSL2-терминала (Ubuntu):
```bash
docker version
docker context ls
```

---

## Структура репозитория

```
your-project/
├─ apps/
│  └─ api-gpu/             # FastAPI: /v1/health, /v1/extract (оркестратор)
├─ configs/                # default.yaml (tilt, normalizer, locales, flags)
├─ docs/                   # OpenAPI, Postman, диаграммы, SLA
├─ lib/
│  ├─ pipelines/
│  │  ├─ extract.py        # preproc → OCR → (TILT via vLLM) → rules → schema
│  │  └─ tilt_client.py    # клиент OpenAI-совм. vLLM (mock/real)
│  └─ post/                
│     └─ rules.py          # правила, нормализации, валидации
├─ notebooks/
│  └─ gpu/                 # R&D ноутбуки
├─ samples/                # тестовые файлы/манифесты
├─ scripts/
│  ├─ start_all.sh         # (GPU) vLLM (TILT) → ожидание → FastAPI
│  ├─ mock_vllm.py         # (CPU) локальный mock OpenAI API
│  └─ bootstrap_*.sh       # утилиты установки/проверки
├─ requirements-cpu.txt    # локальная разработка без GPU
├─ requirements-gpu.txt    # RunPod (A5000) + vLLM + Arctic-TILT
├─ Dockerfile.dev          # CPU dev-образ (WSL2/Docker Desktop)
├─ docker-compose.dev.yml  # локально: mock-vllm + api
├─ Dockerfile.gpu          # GPU образ для RunPod (или Linux GPU-хоста)
├─ pyproject.toml
├─ .gitattributes
├─ .gitignore
└─ README.md
```

---

## Зависимости (Py 3.11)

### CPU (локалка, без vLLM/Arctic‑TILT)
`requirements-cpu.txt`:
```text
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0

huggingface_hub>=0.23
tokenizers>=0.15,<0.20
sentencepiece>=0.1.99
tiktoken>=0.6

# OCR (PaddleOCR 3.x)
paddlepaddle==2.6.1
paddleocr>=3.1,<4.0
#onnxruntime==1.18.1 #при тестах на Windows скорее всего из-за него появлялась ошибка: Error #15: Initializing libiomp5md.dll, but found libomp140.x86_64.dll already initialized.
 

# Images / Geometry
opencv-python-headless>=4.9,<5.0
Pillow>=10.2
shapely>=2.0
pyclipper>=1.3
rapidfuzz>=3.0

# PDF / IO
pypdfium2>=4.20
pdfminer.six>=20221105

# API / utils
fastapi>=0.110
uvicorn[standard]>=0.27
httpx>=0.27
python-multipart>=0.0.9
pydantic>=2.5
loguru>=0.7

numpy>=1.25,<3
```

### GPU (RunPod, vLLM + Arctic‑TILT)
`requirements-gpu.txt`:
```text
--extra-index-url https://download.pytorch.org/whl/cu124
torch==2.4.0+cu124
torchvision==0.19.0+cu124
torchaudio==2.4.0+cu124

vllm==0.8.3
-e git+https://github.com/Snowflake-Labs/arctic-tilt@v0.8.3#egg=arctic-tilt

huggingface_hub>=0.23
tokenizers>=0.15,<0.20
sentencepiece>=0.1.99
tiktoken>=0.6

# OCR (PaddleOCR 3.x)
paddlepaddle-gpu==2.6.1  # если недоступно колесо под вашу CUDA — замените на ==2.6.0
paddleocr>=3.1,<4.0
# onnxruntime-gpu чаще всего не нужен; ставьте только при необходимости
# onnxruntime-gpu==1.18.1

opencv-python>=4.9,<5.0
Pillow>=10.2
shapely>=2.0
pyclipper>=1.3
rapidfuzz>=3.0

pypdfium2>=4.20
pdfminer.six>=20221105

fastapi>=0.110
uvicorn[standard]>=0.27
httpx>=0.27
python-multipart>=0.0.9
pydantic>=2.5
loguru>=0.7

numpy>=1.25,<3
```

---

## Docker для локальной разработки (CPU)

### `.gitattributes` (чтобы shell-скрипты не ломались на Windows)
```gitattributes
*.sh text eol=lf
```

> После коммита: `git update-index --chmod=+x scripts/*.sh` (сделать скрипты исполняемыми).

### `Dockerfile.dev`
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-cpu.txt ./
RUN pip install --upgrade pip && pip install -r requirements-cpu.txt

COPY . .
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn","apps.api_gpu.main:app","--host","0.0.0.0","--port","8000","--reload"]
```

### `docker-compose.dev.yml`
```yaml
version: "3.9"

services:
  mock-vllm:
    build:
      context: .
      dockerfile: Dockerfile.dev
    command: uvicorn scripts.mock_vllm:app --host 0.0.0.0 --port 8001
    ports:
      - "8001:8001"
    volumes:
      - .:/app
    environment:
      - PYTHONUNBUFFERED=1

  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    depends_on:
      - mock-vllm
    environment:
      - VLLM_BASE_URL=http://mock-vllm:8001/v1
      - MOCK_VLLM=1
      - PYTHONUNBUFFERED=1
    ports:
      - "8000:8000"
    volumes:
      - .:/app
```

Запуск:
```bash
docker compose -f docker-compose.dev.yml up --build
```
Проверка:
```bash
curl http://localhost:8000/v1/health
curl -F "file=@samples/your_doc.jpg" http://localhost:8000/v1/extract
```

---

## Docker для GPU/RunPod

### `Dockerfile.gpu`
```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-gpu.txt ./
RUN pip install --upgrade pip && pip install -r requirements-gpu.txt

COPY . .
RUN pip install -e .

COPY scripts/start_all.sh scripts/start_all.sh
RUN chmod +x scripts/start_all.sh
EXPOSE 8000 8001
ENTRYPOINT ["bash","scripts/start_all.sh"]
```

### `scripts/start_all.sh`
```bash
#!/usr/bin/env bash
set -e

export HF_HOME=${HF_HOME:-/workspace/nv/cache/hf}
mkdir -p "$HF_HOME" /workspace/nv/logs

python -m vllm.entrypoints.openai.api_server \
  --model Snowflake/snowflake-arctic-tilt-v1.3 \
  --host 0.0.0.0 --port 8001 \
  --dtype bfloat16 --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  > /workspace/nv/logs/vllm.log 2>&1 &

for i in {1..60}; do
  curl -fsS http://127.0.0.1:8001/v1/models >/dev/null && break
  sleep 2
done

export VLLM_BASE_URL=http://127.0.0.1:8001/v1
exec uvicorn apps.api_gpu.main:app --host 0.0.0.0 --port 8000 --workers 1
```

---

## Конфиги и ENV

`configs/default.yaml`:
```yaml
tilt:
  base_url: ${VLLM_BASE_URL}
  model: Snowflake/snowflake-arctic-tilt-v1.3
  timeout_s: 10.0

normalizer:
  enabled: false
```


---

## Пайплайн и роли

- **mock-vllm (локально):** имитирует OpenAI API vLLM.
- **api:** файл → preproc/OCR → messages → (mock)vLLM → правила/валидации → JSON.

---

## Чек-лист готовности

- [ ] `docker compose -f docker-compose.dev.yml up --build` поднимает **mock-vllm** и **api** на Windows (WSL2).
- [ ] `GET /v1/health` и `POST /v1/extract` работают локально.
- [ ] На RunPod GPU образ стартует `start_all.sh`, vLLM отвечает на `/v1/models`, API — на `/v1/health`.
- [ ] requirements для CPU и GPU зафиксированы под Py 3.11.
- [ ] Документация (OpenAPI/Postman) лежит в `docs/`.

# OCRlty — Контекст и план (CPU-OCR + GPU-TILT в одном поде RunPod)
**TL;DR**: На проде запускаем один под на RunPod, где в одном контейнере крутятся **vLLM + Arctic-TILT (GPU)** и **FastAPI + PaddleOCR 3.x (CPU)**. OCR выполняется на CPU с ограниченным параллелизмом; TILT вызывается асинхронно через OpenAI-совместимый vLLM (`/v1/chat/completions`). Для локалки — Docker Desktop + WSL2, мок vLLM.
---
## 1) Цели и границы (без клиентской части)
- **Цель**: сервис Dev-API, принимающий PDF/JPG/PNG счетов/инвойсов и возвращающий нормализованный JSON (merchant, date, currency, subtotal, tax_amount, total).
- **Границы**: только API и pipeline (OCR → KIE). Клиентской UI-части нет.
- **Модели**:
  - OCR: **PaddleOCR 3.x** (CPU), `paddlepaddle==3.1.0`.
  - KIE: **Snowflake/snowflake-arctic-tilt-v1.3** через **vLLM 0.8.3** (GPU).
---
## 2) Архитектура (прод)
**Один под RunPod** (один контейнер, два процесса):
[Container]
├─ vLLM + Arctic-TILT (GPU) :8001
└─ FastAPI + PaddleOCR (CPU) :8000

Потоки выполнения запроса:
1) API читает файл, при необходимости рендерит PDF → PNG (pypdfium2).
2) **PaddleOCR.predict** (CPU) → детект и распознавание строк (параллелим ограниченно).
3) Формируем **OCR Document** (`pages[].{width,height,spans[{bbox,text}]}`).
4) Шлём в vLLM `/v1/chat/completions` с `messages=[{"type":"input_ocr_document",...}]`.
5) Парсим JSON, применяем правила нормализации, отдаём ответ.

---

## 3) Репозиторий (структура)
├─ apps/
│ └─ api_gpu/ # FastAPI + inference + OpenAPI
├─ configs/ # yaml/json конфиги моделей/правил/локалей
├─ docs/ # OpenAPI-export, примеры, диаграммы, SLA
├─ lib/
│ ├─ pipelines/
│ │ ├─ tilt_client.py # клиент к vLLM (OpenAI chat)
│ │ ├─ tilt_prep.py # OCR→OCR Document→messages
│ │ └─ extract.py # оркестрация стадий (OCR→KIE)
│ └─ post/rules.py # нормализация/валидации (даты/валюты/сумм)
├─ notebooks/
│ └─ gpu/ # R&D
├─ samples/ # эталоны и входные файлы
├─ scripts/ # build, warmup, eval, bootstrap, entrypoints
│  ├─ start_all.sh         # (GPU) vLLM (TILT) → ожидание → FastAPI
│  ├─ mock_vllm.py         # (CPU) локальный mock OpenAI API
│  └─ bootstrap_*.sh       # утилиты установки/проверки
├─ tests/ # unit + integ + e2e
├─ pyproject.toml
├─ requirements-cpu.txt
├─ requirements-gpu.txt
├─ docker-compose.dev.yml
├─ Dockerfile.dev
├─ Dockerfile.prod
└─ README.md
> Важно: директория **`apps/api_gpu/`** (подчёркивание, не дефис) и файлы `__init__.py` в `apps/` и `apps/api_gpu/` для корректного импорта.

---

## 4) Версии и зависимости
### 4.1 CPU-стек (API + OCR) → `requirements-cpu.txt`
- **PaddleOCR 3.x**: `paddleocr>=3.2,<4.0`
- **PaddlePaddle (CPU)**: `paddlepaddle==3.1.0`
- (опц.) **ONNX Runtime**: `onnxruntime==1.18.1` (ускорение через HPI)
- PDF/входы: `pypdfium2>=4.20`, `pdfminer.six>=20221105`
- Изображения/геометрия: `opencv-python-headless>=4.9,<5.0`, `shapely>=2.0`, `pyclipper>=1.3`, `rapidfuzz>=3.0`
- API: `fastapi>=0.110`, `uvicorn[standard]>=0.27`, `pydantic>=2.5`, `httpx>=0.27`, `loguru>=0.7`
- Базовые: `numpy>=1.25,<3`

### 4.2 GPU-стек (vLLM + TILT) → `requirements-gpu.txt`
- **vLLM**: `vllm==0.8.3`
- HF-экосистема: `huggingface_hub>=0.23`, `tokenizers>=0.15,<0.20`, `tiktoken>=0.6`, `sentencepiece>=0.1.99`
- (опц.) Torch под CUDA образ (если требуется конкретной сборкой).

> В dev-образах добавляем системные либы для OpenCV: `libgl1 libglib2.0-0 libsm6 libxext6 libxrender1`.
---
## 5) Локальная разработка (Docker Desktop + WSL2)
### 5.1 Dev-compose
Два сервиса: **`mock-vllm`** (мок OpenAI API) и **`api`** (FastAPI+OCR).
- `api` поднимается на `:8000`, `--reload`, `PYTHONPATH=/app`.
- Исключаем из reloader проблемные директории (`--reload-exclude /app/wheelhouse-gpu/*`).

Команды:
```bash
docker compose -f docker-compose.dev.yml up -d --build
docker compose -f docker-compose.dev.yml logs -f api
docker compose -f docker-compose.dev.yml logs -f mock-vllm
Тесты:

bash
Копировать код
# Windows PowerShell
curl.exe http://localhost:8000/v1/health
curl.exe -F "file=@tests/test_ocr.png" http://localhost:8000/v1/extract
5.2 PaddleOCR smoke-тест
tests/test_paddle_only.py (3.x, CPU, вызов predict()), запуск:
docker compose -f docker-compose.dev.yml exec api python tests/test_paddle_only.py tests/test_ocr.png en

6) Прод (RunPod): один под, два процесса
6.1 ENV (ключевые)
# API
API_PORT=8000
MAX_UPLOAD_MB=20
ALLOWED_CONTENT_TYPES=application/pdf,image/jpeg,image/png
OCR_LANG=en
OCR_DEVICE=cpu
OCR_CONCURRENCY=3
OMP_NUM_THREADS=1
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_API_KEY=secret-or-dummy
RULES_ENABLED=1

# vLLM
TILT_MODEL=Snowflake/snowflake-arctic-tilt-v1.3
VLLM_PORT=8001
VLLM_GPU_UTIL=0.90

6.2 entrypoint (упрощённо)
scripts/entrypoint.sh запускает vLLM, ждёт /v1/models, затем поднимает Uvicorn:
python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port "$VLLM_PORT" \
  --model "$TILT_MODEL" --dtype auto --gpu-memory-utilization "$VLLM_GPU_UTIL" &
# wait-for vLLM ... затем
uvicorn apps.api_gpu.main:app --host 0.0.0.0 --port "$API_PORT" --workers 1
6.3 Тюнинг
Асинхронный клиент к vLLM (один httpx.AsyncClient на приложение, keep-alive).

OCR в пуле потоков: anyio.to_thread.run_sync(ocr.predict, path) + asyncio.Semaphore(OCR_CONCURRENCY).

Ограничить шум тредов: OMP_NUM_THREADS=1–2.

Хэш-кэш на входы (повторные файлы → быстрый ответ).

Хелсчеки: /v1/health (API), /v1/models (vLLM).

7) API контракты
7.1 GET /v1/health
Статус, версия, доступность vLLM.

7.2 POST /v1/extract (single file)
Form-data file: PDF/JPG/PNG (лимиты по типу и размеру).
Результат:
{
  "data": { "merchant": "...", "date": "YYYY-MM-DD", "currency": "ISO", "subtotal": 0.0, "tax_amount": 0.0, "total": 0.0 },
  "meta": { "request_id": "...", "model_version": "...", "ruleset_version": "..." }
}
7.3 POST /v1/extract-batch (несколько одностраничных)
Form-data files: список файлов.
Внутри: OCR по каждому файлу (параллельно с ограничением), затем параллельные вызовы vLLM.
Возвращает список результатов по файлам.

8) Пайплайн OCR → TILT (минимум кода)
PaddleOCR.predict(path) → берём rec_texts и rec_polys.
Строим OCR Document (AABB из полигонов; ширина/высота страницы).
messages = [{"role":"user","content":[{"type":"input_ocr_document","document": ...}]}] (+ system-prompt с требуемой схемой JSON).

POST /v1/chat/completions на vLLM, достаём choices[0].message.content.

json.loads + postprocess_rules() (валидаторы дат/валют/сумм).

9) Наблюдаемость и качество
Логи с request_id, latency по стадиям (OCR, vLLM, total).
Метрики по ошибкам (400/413/415/502/503/500).
Эталонные samples/ и e2e-тесты (PNG/PDF, кириллица/латиница, валюты).
SLA/TTR: ретраи vLLM на 502/503; circuit-breaker при деградации.

10) Дорожная карта (коротко)
✅ Локальная связка: mock-vLLM + API (Docker Desktop).
✅ PaddleOCR 3.x (CPU) и тесты.
🔜 /v1/extract-batch, асинхронный клиент vLLM, семафор OCR.
🔜 Prod-образ (entrypoint.sh) для RunPod (один под).
🔜 Документация docs/ (OpenAPI, примеры запросов/ответов, SLA).
⏭️ HPI/ONNX Runtime для OCR (ускорение CPU, опционально).
⏭️ Очереди задач (Redis/RQ) при росте объёмов и SLA по прогрессу.

11) Заметки по Windows/PowerShell
Используй curl.exe вместо curl -F (PS алиас на Invoke-WebRequest).
Если uvicorn reloader падает на bind-mount директории, добавь --reload-exclude и/или --reload-dir.
Для OpenCV headless нужны системные либы (в образе dev): libgl1 libglib2.0-0 libsm6 libxext6 libxrender1.

12) Пример системного промпта для TILT
“Extract merchant, date (YYYY-MM-DD), currency (ISO 4217), subtotal, tax_amount, total from the OCR document. Return strict valid JSON with exactly these keys and numeric values as floats. If a value is missing, set it to null.”

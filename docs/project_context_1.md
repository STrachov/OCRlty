# Контекст проекта (обновлён)

**Цель:** R&D → сервис OCR/KIE для чеков и инвойсов (Upwork-ориентированный dev-API без клиентской части).

---

## Архитектура монорепо

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
│  └─ post/                # правила, нормализации, валидации
├─ notebooks/
│  └─ gpu/                 # R&D ноутбуки
├─ samples/                # тестовые файлы/манифесты
├─ scripts/
│  ├─ start_all.sh         # vLLM (TILT) → ожидание → FastAPI
│  ├─ mock_vllm.py         # локальный mock OpenAI API (CPU)
│  └─ bootstrap_*.sh       # утилиты установки/проверки
├─ requirements-cpu.txt    # локальная разработка без GPU
├─ requirements-gpu.txt    # RunPod (A5000) + vLLM + Arctic-TILT
├─ pyproject.toml
├─ .gitignore
└─ README.md
```

---

## Ключевые технические решения

### 1) Сервинг модели: **vLLM как OpenAI-совместимый сервер**
- Модель: `Snowflake/snowflake-arctic-tilt-v1.3`.
- Сервис поднимается как: `python -m vllm.entrypoints.openai.api_server …`.
- **TiltPreprocessor** берётся из репозитория Snowflake-Labs (namespace-пакет `vllm.*`):
  - в `requirements-gpu.txt`:  
    `vllm==0.8.3` →  
    `-e git+https://github.com/Snowflake-Labs/arctic-tilt@v0.8.3#egg=arctic-tilt`.

### 2) Orchestrator: **FastAPI**
- Принимает PDF/JPG → preproc/OCR → формирует messages (через `TiltPreprocessor`) → `/v1/chat/completions` в vLLM → post-rules → **стабильный JSON**.
- Единственная публичная точка доступа; vLLM не экспонируется наружу.

### 3) Локальная разработка **без GPU**
- **vLLM не поддерживает CPU-инференс** → используем `scripts/mock_vllm.py` (OpenAI-совместимый мок на `:8001`).
- В `lib/pipelines/tilt_client.py` два режима:
  - `MOCK_VLLM=1` → лёгкий dev-клиент (без импорта Arctic-TILT), ходит в mock.
  - по умолчанию → «боевой» клиент (лениво импортирует `TiltPreprocessor`).

### 4) Зависимости (Py 3.11)
- **CPU** (`requirements-cpu.txt`): Torch CPU; PaddleOCR **3.x**; без vLLM/Arctic-TILT.
- **GPU** (`requirements-gpu.txt`): Torch CUDA 12.4; `vllm==0.8.3`; **editable** Arctic-TILT `@v0.8.3`; PaddleOCR **3.x**.
- PaddlePaddle GPU версия:
  - предпочтительно `paddlepaddle-gpu==2.6.1` (если есть колесо под вашу CUDA),
  - иначе `==2.6.0`.

### 5) Развёртывание на RunPod (экономичный старт)
- **Один Pod (рекомендовано на старте)**: vLLM + FastAPI в одном контейнере.
- **Network Volume (NV)**: `HF_HOME`/кэши/логи на NV → быстрый рестарт.
- **Spot A5000** + чекпоинты при обучении (когда дойдём до fine-tune).

---

## Конфиги и переменные окружения

`configs/default.yaml` (минимум):
```yaml
tilt:
  base_url: ${VLLM_BASE_URL}         # напр. http://127.0.0.1:8001/v1
  model: Snowflake/snowflake-arctic-tilt-v1.3
  timeout_s: 10.0

normalizer:
  enabled: false                      # доп. LLM-логика выключена по умолчанию
```

ENV (пример):
```
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_API_KEY=dummy
MOCK_VLLM=1            # ← только локально
HF_HOME=/workspace/nv/cache/hf
TRANSFORMERS_CACHE=/workspace/nv/cache/hf
```

---

## Стартовые сценарии

### Локально (CPU, мок)
1) `pip install -r requirements-cpu.txt && pip install -e .`
2) `uvicorn scripts.mock_vllm:app --host 127.0.0.1 --port 8001`
3) `export VLLM_BASE_URL=http://127.0.0.1:8001/v1 && export MOCK_VLLM=1`
4) `uvicorn apps.api_gpu.main:app --host 127.0.0.1 --port 8000`
5) Проверка:
   - `GET http://127.0.0.1:8000/v1/health`
   - `POST /v1/extract` (файл) → JSON.

### RunPod (GPU, боевой)
- **Start Command** (в одном Pod):
  ```bash
  bash -lc '
  export HF_HOME=/workspace/nv/cache/hf; mkdir -p "$HF_HOME" /workspace/nv/logs
  # vLLM (TILT) в фоне
  python -m vllm.entrypoints.openai.api_server \
    --model Snowflake/snowflake-arctic-tilt-v1.3 \
    --host 0.0.0.0 --port 8001 \
    --dtype bfloat16 --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    > /workspace/nv/logs/vllm.log 2>&1 &

  # ожидание готовности
  for i in {1..60}; do curl -fsS http://127.0.0.1:8001/v1/models >/dev/null && break; sleep 2; done

  # FastAPI
  export VLLM_BASE_URL=http://127.0.0.1:8001/v1
  uvicorn apps.api_gpu.main:app --host 0.0.0.0 --port 8000 --workers 1
  '
  ```
- Открываем наружу **только 8000**; порт 8001 не экспонируем.

---

## Данные, обучение и оценка

- **Тренировка (receipts):** `CORD-v2` — строго `train/val/test`, финальный отчёт по `test`.
- **Cross-domain eval:** `SROIE` (общие поля), **DocILE — только оценка** (некоммерческая лицензия).
- **Инвойсы (прод):** коммерчески чистые данные (клиентские/синтетика) + табличные/layout наборы.
- **Метрики:** F1 по ключевым полям (`merchant/date/total/tax/currency`) + latency p50/p95; для инвойсов — LIR (строки позиций).

---

## R&D ноутбуки (неделя 1, укороченный набор)

1) `00_env-io-warmup.ipynb` — проверка окружения, прогрев, сбор манифестов из `samples/`.
2) `01_preproc+ocr_ablation.ipynb` — DPI/deskew/denoise, CPU-OCR sanity.
3) `02_kie_min+rules_validation.ipynb` — Arctic-TILT без дообучения + правила.
4) `04_e2e_smoke+latency_caching.ipynb` — сквозной прогон и латентность.
5) `05_eval_report_week1.ipynb` — отчёт: CORD-test + SROIE mini-eval.

Все вызовы — через `lib.*`; после проверки функции выносятся из ноутбуков в `lib/`.

---

## API-контракт (первый релиз)

- `GET /v1/health` → `status`, `model_version`, `ruleset_version`, признак reachability vLLM.
- `POST /v1/extract` (multipart file) → **стабильный JSON** по Pydantic-схеме (поля + meta).
- Документация: `openapi.json`, 2–3 `curl` примера, Postman-коллекция.

---

## Затраты и инфраструктура

- **RunPod A5000 Spot** для инференса/кратких джоб — экономично (NV-кэш ускоряет рестарт).
- Для обучения >12–24 ч или демо со строгим SLA — можно On-Demand.
- Чекпоинты при обучении каждые 10–15 минут на NV; `--resume_from last.ckpt`.

---

## Лицензии и риски

- **DocILE** — только для исследовательских целей (eval-only). Не использовать в тренинге без отдельной коммерческой лицензии.
- Прод-обучение — только на совместимых наборах (CORD-v2 CC-BY, собственные/синтетика, CDLA-совместимые для layout/table).

---

## Быстрый чек-лист готовности

- [ ] Локально (CPU): работает `mock_vllm` и `api` → энд-ту-энд JSON.
- [ ] RunPod (GPU): поднят vLLM (TILT), `api` ходит на `localhost:8001`.
- [ ] Стабильные `requirements-cpu.txt` и `requirements-gpu.txt` (Py 3.11).
- [ ] `tilt_client.py` поддерживает `MOCK_VLLM` и боевой режим.
- [ ] `openapi.json` и примеры вызовов в `docs/`.
- [ ] Мини-отчёт по CORD-v2 (test) + SROIE mini-eval.

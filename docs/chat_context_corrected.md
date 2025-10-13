# Контекст проекта (API‑only, без клиентского UI)

**Цель:** R&D → продакшн **OCR/KIE сервиса для счетов и инвойсов** с ориентацией на спрос клиентов Upwork. Поставка — **только Dev‑API** для интеграторов (без отдельного веб‑интерфейса).

## Область и критерии успеха
- **Область:** извлечение структурированных данных из инвойсов/чеков (KIE) + базовый нормализатор, батч‑обработка, обратная связь (feedback) и активное дообучение.
- **Критерии приёмки:**
  - SLA по отклику для одиночного документа: _p50_ ≤ 1.5s (GPU), _p95_ ≤ 4s при A4/PDF или JPG ≤ 3MB.
  - Точность по ключевым полям инвойса (supplier_name, invoice_no, dates, totals, currency): **≥ 0.93 F1** на эталонном сете; для строк (line_items) — **≥ 0.90 F1** по ячейкам.
  - Репродуцируемая **оценка качества** (`scripts/eval.py`) и отчёт в артефактах CI.
  - **API‑контракт (OpenAPI)** и **Postman‑коллекция** — можно интегрировать «как есть».

## Архитектура кода (monorepo)
- `apps/api-gpu/` — FastAPI сервис инференса (GPU, RunPod).
- `apps/api-cpu/` — при необходимости лёгкий CPU‑нормализатор/валидация (опционально).
- `lib/` — общая библиотека: OCR, KIE (например, Arctic‑TILT), правила постпроцесса, валидации, мапперы в ERP.
- `notebooks/gpu/` — R&D‑ноутбуки (прототипирование).
- `scripts/` — bootstrap, прогрев весов, сборка образа, запуск eval.
- `configs/` — конфиги моделей, подключаемые правила/языки/локали.
- `tests/` — unit/integ/e2e + снапшоты JSON.
- Корневые: `pyproject.toml`, `requirements-gpu.txt`, `README.md`.

**Импорты:** единый пакет (`pyproject.toml` + `pip install -e .`) → импорт `lib.*` доступен отовсюду.

## Технологический стек и версии
- PyTorch **2.4** / CUDA **12.4**, PaddleOCR **3.x**, FastAPI, Pydantic v2, Uvicorn, uvloop.
- vLLM **0.8.3** (опционально) — для LLM‑нормализации/объяснений (вынос в отдельный процесс/порт).
- Трёхуровневый постпроцесс: правила (regex/грамматики/локали) → валидации (дат/валют, суммы) → нормализатор (LLM‑reasoner, при необходимости).

## Развёртывание и окружение (RunPod)
- Модель запуска: **Spot Pod + Network Volume (NV)**.
- **Холодный старт ≤ 5 минут** за счёт:
  - Предсобранного Docker‑образа c pinned зависимостями (Torch/Paddle/Tesseract/Poppler при необходимости).
  - Локального **wheelhouse** только для тяжёлых пакетов (бэкап‑вариант при обновлении образа).
  - Кэшей HF/Paddle/PIP на NV; отдельный `venv-gpu` на NV (bootstrap ≤ 60s).
  - **Lazy‑прогрева весов** при старте (`scripts/warmup.py`) + здравый `GET /v1/health`.
  - Автостоп/terminate между спринтами, старт по требованию.
- Конфиги через `.env`/`configs/*.yaml`: язык, локали, валюты, правила VAT/GST, включение/выключение LLM‑нормализатора.

## Поток данных (высокоуровневый)
`PDF/JPG` → preproc (deskew, split, dpi) → **OCR** → KIE (layout/seq) → постпроцесс (правила, валидации, нормализация) → **JSON‑схема** → (опц.) мапперы в **CSV/XLSX/ERP** → `/feedback` → накопление датасета → `/retrain` (активное обучение).

## Целевые вертикали и схемы полей (MVP)
### Инвойсы (B2B)
```json
{
  "document_type": "invoice",
  "supplier_name": "...",
  "supplier_vat": "...",
  "invoice_no": "...",
  "issue_date": "YYYY-MM-DD",
  "due_date": "YYYY-MM-DD",
  "currency": "EUR",
  "subtotal": 0.0,
  "tax_amount": 0.0,
  "total": 0.0,
  "po_number": "...",
  "line_items": [{"desc": "...", "qty": 1, "unit_price": 0.0, "amount": 0.0}]
}
```
**Валидации:** `abs(subtotal + tax_amount - total) ≤ ε`; даты по локали; ISO‑4217; `invoice_no` дедуп.

### Розничные чеки
```json
{
  "document_type": "receipt",
  "merchant": "...",
  "date": "YYYY-MM-DD",
  "payment_method": "cash|card|...",
  "currency": "EUR",
  "total": 0.0,
  "tax": 0.0,
  "items": [{"name": "...", "qty": 1, "price": 0.0}]
}
```

## API‑контракт (обязательный минимум)
- `POST /v1/extract` — файл/URL → **JSON** + `bbox`, `confidence`, `model_version`, `ruleset_version`.
- `POST /v1/batch` — список файлов → `job_id` (асинхронно, частичный прогресс).
- `GET /v1/jobs/{id}` — статус, прогресс, частичные результаты.
- `POST /v1/feedback` — приём правок (дифф/исправленный JSON), persistent в `datasets/client_*`.
- `POST /v1/retrain` — ручной триггер дообучения (асинхронный; отдаёт метрики «до/после»).
- `GET /v1/health` — версии, состояние моделей, latency прогрева.

Артефакты: **OpenAPI (YAML/JSON)** в `apps/api-gpu/openapi.*`, **Postman‑коллекция** в `docs/postman_collection.json`, примеры `curl` в `README.md`.

## Качество и оценка
- `scripts/eval.py` — отчёт CSV/JSON по метрикам (по полям и по строкам), агрегации по документам/вендорам.
- `samples/` — 10–20 эталонных инвойсов/чеков с разметкой и ожидаемым JSON.
- В каждом ответе — **confidence + источники**: `source_text`, `source_bbox`, `rule_applied`.
- CI (pytest + eval) → артефакт‑отчёт, контроль регрессий.

## Безопасность и комплаенс
- **On‑prem/air‑gap режим** (`NO_EXTERNAL=1`) — никакого исходящего трафика.
- **TTL‑удаление** входных файлов/текста (например, 60 минут, настраивается).
- Маскирование PII/финданных в логах (IBAN/карты/адреса).
- **Audit‑лог**: `request_id`, хэш файла, версии моделей, latency, узкие места.

## Репозиторий (структура)
```
your-project/
├─ apps/
│  └─ api-gpu/          # FastAPI + inference + OpenAPI
├─ configs/             # yaml/ini/json конфиги моделей/правил/локалей
├─ docs/                # OpenAPI-export, Postman, диаграммы, SLA/политики
├─ lib/
│  └─ pipelines/        # OCR, KIE, постпроцесс, мапперы ERP
├─ notebooks/
│  └─ gpu/              # R&D
├─ samples/             # эталоны и входные файлы
├─ scripts/             # build, warmup, eval, bootstrap
├─ tests/               # unit + integ + e2e
├─ pyproject.toml
├─ requirements-gpu.txt
├─ .gitignore
└─ README.md
```

## Дорожная карта (6 недель, без UI)
1. **Неделя 1:** каркас монорепо, Docker‑образ, прогрев, `GET /v1/health`, `POST /v1/extract` (PDF/JPG→JSON, без line_items).
2. **Неделя 2:** line_items для инвойсов, валидации валют/дат/сумм, первые метрики и `scripts/eval.py`.
3. **Неделя 3:** `POST /v1/batch` + `GET /v1/jobs/{id}`, `samples/` и эталоны, отчёт CI.
4. **Неделя 4:** `/feedback` + накопление датасета клиента; `/retrain` (минимально работающее дообучение/обновление правил).
5. **Неделя 5:** мапперы в CSV/XLSX и ERP (SAP/Xero/QuickBooks), Postman‑коллекция, SLA/политики (TTL, on‑prem).
6. **Неделя 6:** оптимизации производительности, multilingual (опц.), финальная документация и демонстрационные сценарии (curl/скрипт).

## Не‑цели (на этапе MVP)
- Отдельный веб‑интерфейс/админка. 
- Сложные ML‑объяснения (explainability) — только базовые трассировки (bbox/score/rule).
- Дизайн сложных RBAC/мульти‑тенант — пока единая зона данных (разделение по клиентским префиксам в сторадже).

---
_Обновлено: 2025-09-19._

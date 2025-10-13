# Контекст проекта

**Цель:** R&D с последующей разработкой OCR/IE сервиса для счетов и инвойсов (ориентация на спрос у клиентов Upwork).

**Архитектура кода (monorepo):**
- `apps/api-gpu/` — FastAPI-сервис для инференса (GPU, напр. RunPod).
- `apps/api-cpu/` — FastAPI-сервис для клиентской части (CPU-хостинг), позже.
- `notebooks/gpu/` — R&D-ноутбуки под GPU.
- `lib/` — общая библиотека (OCR, TILT, пайплайны, утилиты).

**Импорты:** `pyproject.toml` (setuptools) + `pip install -e .` → импорт `lib.*` работает отовсюду.

**Зависимости:** `requirements-gpu.txt` (PyTorch 2.4 / CUDA 12.4, **vLLM 0.8.3**, PaddleOCR 3.x, FastAPI и утилиты).  
Используются локальные wheels (wheelhouse) для быстрой повторной установки на Pod.

**RunPod:** Spot + Network Volume.  
- `venv`: `/workspace/venv-gpu`  
- Кэши HF/Paddle/PIP: `/workspace/.cache`  
- Wheelhouse (опц.): `/workspace/wheelhouse-gpu`

**Монтирование:** Pod → `/workspace`; (на будущее) Serverless → `/runpod-volume`.

**Регион:** датацентр с наибольшей доступностью A5000 (medium/high). Если в текущем DC нет S3-доступа к NV — ок, работаем через запущенный Pod.

**Практика работы:** «спринты» + **Terminate** между ними; цель — минимизировать бюджет на сессию (желательно до ~$1–2).

---

## Верхнеуровневый план

1. **Каркас репо:** `lib/apps/scripts/configs/notebooks`, `pyproject.toml`, заглушка пайплайна.  
2. **OCR на CPU (минимум):** базовый текст + bbox, первичный постпроцесс.  
3. **GPU-среда на RunPod:** Pod+NV, единый `venv-gpu`, скачаны веса.  
Чтобы обеспечить быстрый старт/минимальную стоимость (критерий - минимальное время от старта до “готов к работе”, желательно < 5 минут):
	- Terminate между спринтами,
	- автостоп (таймер),
	- кэши HF/Paddle/PIP на NV,
	- venv на NV + базовый bootstrap,
	- минимальный wheelhouse только для самых тяжёлых пакетов под текущий шаблон.
   _Критерий: холодный старт до «готов к работе» < 5 минут._
4. **Подключение Arctic-TILT (минимум):** 2–3 поля (напр. vendor, total, date).  
5. **Контур оценки:** простые метрики/эталоны, `scripts/eval.py`.  
6. **Пост-обработка / нормализация:** рост метрик на тех же данных.  
7. **Dev-API (для себя) на GPU.**  
8. **Полный wheelhouse** на NV, финальный bootstrap (холодный старт ≤ 5 мин).  
9. **Мини-демо + Backlog** следующих итераций.

##Лёгкий bootstrap.sh, который ничего не ставит, а только проверяет:
#!/usr/bin/env bash
set -e
export RUNPOD_VOLUME_ROOT=/workspace
source /workspace/venv-gpu/bin/activate
python -c "import torch,sys; print('Py',sys.version.split()[0],'Torch',torch.__version__)"
echo "✅ Ready"

---

## Рабочий процесс

**Кратко:** Local (Cursor / Win10) ↔ GitHub ↔ RunPod

**Подробно:**
**Local (Cursor / Windows 10):**
- Пишем код в `lib/...`, минимальные CPU-тесты и форматирование.
- `py -3.11 -m venv .venv` → `.\.venv\Scripts\Activate.ps1` → `pip install -e .` → `pip install -r requirements-gpu.txt` (GPU-блок можно временно убрать локально).
- Коммитим только код/конфиги; **не** коммитим данные/веса/venv.

**GitHub:**
- Центр синхронизации кода.
- (Позже) CI на линт/pytest; сборка Docker-образа для прод.

**RunPod (GPU + NV):**
- Только то, что требует GPU: vLLM/TILT, бенчи, профилинг.
- Тянем последний коммит нужной ветки:
  ```bash
  git clone --depth 1 --single-branch --branch main https://github.com/<USER>/<REPO>.git


##Текущая структура проекта
your-project/
├─ apps/
│  └─ api-gpu/
├─ configs/
├─ lib/
│  └─ pipelines/
├─ notebooks/
│  └─ gpu/
├─ scripts/
├─ pyproject.toml
├─ requirements-gpu.txt
├─ .gitignore
└─ README.md


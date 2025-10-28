# Dev-Guide: Локальный ПК ↔ GitHub ↔ RunPod (git-mode + персистентный NV)

Этот документ объясняет, **что именно делать** в рабочей схеме разработки, где код живёт в GitHub, а на RunPod используется **git-mode** с **персистентным виртуальным окружением (venv)** и кэшами на **Network Volume** (NV). Предполагается, что у вас уже есть **готовое колесо vLLM** и/или base‑image с предустановленными `torch + xformers + vLLM`.

---

## 1) Как это устроено (коротко)
- Под на **RunPod** стартует из **base‑image**, где уже стоят тяжёлые зависимости: `torch + xformers + vLLM` (ваше колесо).
- На смонтированном **NV** (`/workspace`) хранятся:
  - `/workspace/src` — ваш код (клонируется/обновляется из GitHub на старте);
  - `/workspace/venv` — **персистентный venv**;
  - `/workspace/wheelhouse` — локальные `.whl` (включая ваш `vllm` на случай офлайн‑установок);
  - `/workspace/cache/hf` — кэш весов/токенайзеров Hugging Face.
- Стартовый скрипт `entrypoint.sh` при каждом рестарте делает `git pull` и **поддерживает синхронность venv с `requirements-app.txt`** за счёт хеша файла.

Результат: повторные старты занимают **секунды** (без `pip install` и скачивания весов).

---

## 2) Типовые действия — что делать мне, если…

### 2.1 Изменить / добавить **код**
1. Локально правите файлы → `git commit` → `git push` (ветка `dev`).
2. На RunPod делаете **Restart container**.  
   На старте `entrypoint.sh` выполнит `git pull` и перезапустит сервис.

> Хотите без рестарта пода: в shell пода — `git -C /workspace/src pull`, затем перезапустить процесс uvicorn. В проде надёжнее полный рестарт пода.

---

### 2.2 Добавить / обновить **обычный пакет** (не torch/xformers/vLLM)
1. В репозитории обновите `runpod/requirements-app.txt` (легкие пакеты).
2. При необходимости положите `.whl` пакета в `/workspace/wheelhouse` (на NV).
3. `git commit` → `git push` → **Restart container**.  
   `entrypoint.sh` заметит изменение (по хешу) и один раз выполнит `pip install -r ...` **в уже существующий venv** (офлайн через wheelhouse).

**Фингерпринт для авто‑установки только при изменениях:**
```bash
REQ_FILE=/workspace/src/runpod/requirements-app.txt
REQ_HASH_FILE=/workspace/venv/.req.hash

new_hash=$(sha256sum "$REQ_FILE" | cut -d' ' -f1)
old_hash=$(cat "$REQ_HASH_FILE" 2>/dev/null || echo "")

if [ "$new_hash" != "$old_hash" ]; then
  export PIP_NO_INDEX=1 PIP_FIND_LINKS=/workspace/wheelhouse
  pip install -U pip && pip install -r "$REQ_FILE"
  echo "$new_hash" > "$REQ_HASH_FILE"
fi
```

---

### 2.3 Обновить **тяжёлую базу** (torch/xformers/vLLM)
Эти пакеты входят в **base‑image**. Для стабильности и скорости их **не** ставят при старте.

1. Обновите версии в `Dockerfile.base` (или поменяйте URL вашего `vllm`‑wheel).
2. Соберите и опубликуйте образ → получите новый тег, например  
   `ghcr.io/<you>/ocr-dev:cu124-torch26-vllm083b`.
3. На RunPod смените **Image** у пода на новый тег → **Restart container**.  
   Venv с флагом `--system-site-packages` использует системные пакеты из образа.

> Для временного эксперимента можно поставить whl в venv; для постоянного решения лучше пересобрать базовый образ.

---

### 2.4 Поменять **веса модели / конфиг**
1. Обновите `ENV` (например, `TILT_MODEL=...`) в настройках пода или в `.env` репозитория.
2. **Restart container**. Веса кэшируются в `/workspace/cache/hf`, скачивание повторно не требуется.

---

### 2.5 Добавить/обновить **wheel в wheelhouse**
- С локальной машины: `scp your_pkg-*.whl runpod:/workspace/wheelhouse/`  
- Или в поде: `wget -O /workspace/wheelhouse/your_pkg-*.whl <URL>`

После этого установка пройдёт офлайн:
```bash
export PIP_NO_INDEX=1
export PIP_FIND_LINKS=/workspace/wheelhouse
pip install your_pkg-*.whl
```

---

### 2.6 Полная чистка окружения (редко)
- Пересоздать venv:  
  ```bash
  rm -rf /workspace/venv && restart
  ```
  На старте `entrypoint.sh` заново создаст venv и поставит зависимости.
- Сбросить код:  
  ```bash
  git -C /workspace/src reset --hard origin/dev
  ```

---

## 3) Когда нужен новый образ, а когда хватает `git push`
- **Только код / лёгкие пакеты** → `git push` → Restart (venv сам догонит зависимости).
- **torch / xformers / vLLM / смена CUDA** → **пересборка base‑image** и смена тега у пода.

---

## 4) Мини‑шпаргалка

**Локально:**
```bash
git add -A
git commit -m "feat: update app logic / bump small deps"
git push origin dev
```

**В UI RunPod:**
- Restart container.

**Внутри пода при необходимости:**
```bash
# обновить код без рестарта пода
git -C /workspace/src pull --ff-only

# докинуть колесо
wget -O /workspace/wheelhouse/somepkg-1.2.3-py3-none-any.whl <URL>

# пересоздать venv (крайняя мера)
rm -rf /workspace/venv
```

---

## 5) Базовая подготовка (если ещё не сделано)

### 5.1 Структура в репозитории
```
runpod/
  Dockerfile.base
  entrypoint.sh
  requirements-app.txt   # без torch/xformers/vLLM
  env.example
```

### 5.2 Пример `Dockerfile.base`
```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip git curl &&     rm -rf /var/lib/apt/lists/* && python3.10 -m pip install -U pip

# Torch/cu124 + xformers
RUN python3.10 -m pip install --extra-index-url https://download.pytorch.org/whl/cu124     torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 &&     python3.10 -m pip install xformers==0.0.29.post2

# vLLM из вашего релиза (колесо уже есть)
ARG VLLM_WHL_URL
ADD ${VLLM_WHL_URL} /tmp/vllm.whl
RUN python3.10 -m pip install /tmp/vllm.whl && rm -f /tmp/vllm.whl

# утилиты
RUN python3.10 -m pip install uvicorn fastapi httpx loguru

ENV HF_HOME=/workspace/cache/hf     VLLM_ATTENTION_BACKEND=XFORMERS     VLLM_USE_FLASH_ATTENTION=0     PYTHONUNBUFFERED=1

WORKDIR /srv
COPY runpod/entrypoint.sh /srv/entrypoint.sh
RUN chmod +x /srv/entrypoint.sh
EXPOSE 8001
CMD ["/srv/entrypoint.sh"]
```

### 5.3 Пример `entrypoint.sh`
```bash
#!/usr/bin/env bash
set -euo pipefail

: "${GIT_URL:?GIT_URL is required}"
: "${GIT_BRANCH:=dev}"
: "${HF_HOME:=/workspace/cache/hf}"
: "${PORT_VLLM:=8001}"
: "${PIP_FIND_LINKS:=/workspace/wheelhouse}"
: "${PIP_NO_INDEX:=1}"

mkdir -p /workspace/src /workspace/wheelhouse "$HF_HOME"

# 1) код (clone/pull)
if [ ! -d /workspace/src/.git ]; then
  git clone --branch "$GIT_BRANCH" --depth 1 "$GIT_URL" /workspace/src
else
  git -C /workspace/src fetch origin "$GIT_BRANCH" --depth 1
  git -C /workspace/src checkout "$GIT_BRANCH"
  git -C /workspace/src reset --hard "origin/$GIT_BRANCH"
fi

# 2) окружение (персистентный venv + системные site-packages)
if [ ! -d /workspace/venv ]; then
  python3.10 -m venv /workspace/venv --system-site-packages
  . /workspace/venv/bin/activate

  # авто-установка "лёгких" зависимостей по хешу
  REQ_FILE=/workspace/src/runpod/requirements-app.txt
  REQ_HASH_FILE=/workspace/venv/.req.hash

  if [ -f "$REQ_FILE" ]; then
    export PIP_NO_INDEX PIP_FIND_LINKS
    pip install -U pip && pip install -r "$REQ_FILE"
    sha256sum "$REQ_FILE" | cut -d" " -f1 > "$REQ_HASH_FILE"
  fi
else
  . /workspace/venv/bin/activate
  REQ_FILE=/workspace/src/runpod/requirements-app.txt
  REQ_HASH_FILE=/workspace/venv/.req.hash

  if [ -f "$REQ_FILE" ]; then
    new_hash=$(sha256sum "$REQ_FILE" | cut -d" " -f1)
    old_hash=$(cat "$REQ_HASH_FILE" 2>/dev/null || echo "")
    if [ "$new_hash" != "$old_hash" ]; then
      export PIP_NO_INDEX PIP_FIND_LINKS
      pip install -U pip && pip install -r "$REQ_FILE"
      echo "$new_hash" > "$REQ_HASH_FILE"
    fi
  fi
fi

# 3) запуск вашего TILT-сервера (пример)
cd /workspace/src
python -m uvicorn apps.tilt_api:app --host 0.0.0.0 --port "$PORT_VLLM"
```

### 5.4 ENV (пример для RunPod)
```
GIT_URL=https://github.com/<you>/<repo>.git
GIT_BRANCH=dev
HF_HOME=/workspace/cache/hf
PIP_FIND_LINKS=/workspace/wheelhouse
PIP_NO_INDEX=1
VLLM_ATTENTION_BACKEND=XFORMERS
VLLM_USE_FLASH_ATTENTION=0
PORT_VLLM=8001
```

---

## 6) Дополнительно
- **Кеши**: имеет смысл увести и другие кеши на NV
  ```bash
  PIP_CACHE_DIR=/workspace/.cache/pip
  XDG_CACHE_HOME=/workspace/.cache
  TORCH_HOME=/workspace/.cache/torch
  TRITON_CACHE_DIR=/workspace/.cache/triton
  ```
- **Прогрев** после старта (ускоряет первый запрос):
  ```bash
  curl -sS localhost:8001/v1/models || true
  ```

---

### TL;DR
- Правите код → **git push** → **Restart container**.
- Добавили пакет → правите `requirements-app.txt` → **git push** → Restart (auto‑install по хешу).
- Хотите другую версию **vLLM/torch/xformers** → **пересобираете base‑image** и меняете тег у пода.
- Колёса держите в `/workspace/wheelhouse` для офлайн‑установок.  
- Кэш моделей — в `/workspace/cache/hf` для быстрых стартов.

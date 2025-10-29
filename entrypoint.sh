#!/usr/bin/env bash
set -euo pipefail

: "${GIT_URL:=https://github.com/STrachov/OCRlty.git}"
: "${GIT_BRANCH:=main}"
: "${HF_HOME:=/workspace/cache/hf}"
: "${PORT_VLLM:=8001}"
: "${PIP_FIND_LINKS:=/workspace/wheelhouse}"
# wheelhouse оставляем как «подсказку», но НЕ режем доступ к PyPI
 
mkdir -p /workspace/src /workspace/wheelhouse "$HF_HOME"

# код (clone/pull)
if [ ! -d /workspace/src/.git ]; then
  git clone --branch "$GIT_BRANCH" --depth 1 "$GIT_URL" /workspace/src
else
  git -C /workspace/src fetch origin "$GIT_BRANCH" --depth 1
  git -C /workspace/src reset --hard "origin/$GIT_BRANCH"
fi


# venv
if [ ! -d /workspace/venv ]; then
  python3.10 -m venv /workspace/venv --system-site-packages
fi
. /workspace/venv/bin/activate

# <<< ВАЖНО: без PIP_NO_INDEX. Разрешаем PyPI + подмешиваем wheelhouse >>>
python -m pip install -U pip
# Подстрахуемся: заранее доставим билд-инструменты, которые часто нужны из PyPI
python -m pip install --extra-index-url https://pypi.org/simple \
  "cmake>=3.26" "ninja" "scikit-build-core" "build" "setuptools" || true

if [ -f /workspace/src/requirements-gpu.txt ]; then
  python -m pip install --find-links "$PIP_FIND_LINKS" \
    --extra-index-url https://pypi.org/simple \
    -r /workspace/src/requirements-gpu.txt
fi

# запуск API
cd /workspace/src
python -m uvicorn apps.tilt_api:app --host 0.0.0.0 --port "$PORT_VLLM"

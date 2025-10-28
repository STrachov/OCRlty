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
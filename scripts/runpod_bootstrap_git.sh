#!/usr/bin/env bash
# RunPod bootstrap (Git mode): clone/pull from GitHub, reuse wheelhouse+venv on Network Volume, start vLLM+API
set -euo pipefail

# -------- Config via ENV --------
: "${GIT_REPO:?e.g. git@github.com:owner/repo.git or https://github.com/owner/repo.git}"
: "${GIT_BRANCH:=main}"
: "${API_DIR:=apps/api_gpu}"
: "${REQS_FILE:=requirements-prod.txt}"
: "${MODEL_NAME:=Snowflake/snowflake-arctic-tilt-v1.3}"
: "${RELOAD:=0}"
: "${VLLM_GPU_UTIL:=0.80}"
: "${PORT_API:=8000}"
: "${PORT_VLLM:=8001}"
: "${GIT_USER_NAME:=RunPod Bot}"
: "${GIT_USER_EMAIL:=runpod@example.local}"
: "${USE_SSH:=1}"
: "${SSH_KEY_PATH:=/workspace/nv/.ssh/id_ed25519}"
: "${BASE:=/workspace/nv}"

SRC_DIR="${BASE}/src"
WHEELHOUSE="${BASE}/wheelhouse"
VENV="${BASE}/venv"
PIP_CACHE_DIR="${BASE}/.pipcache"
HF_HOME="${BASE}/cache/hf"
LOG_DIR="${BASE}/logs"

mkdir -p "${SRC_DIR}" "${WHEELHOUSE}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${LOG_DIR}"
export HF_HOME PIP_CACHE_DIR

# Ensure git
if ! command -v git >/dev/null 2>&1; then
  apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ssh-client ca-certificates     && rm -rf /var/lib/apt/lists/*
fi

git config --global user.name  "${GIT_USER_NAME}"
git config --global user.email "${GIT_USER_EMAIL}"
git config --global --add safe.directory "${SRC_DIR}"
git config --global init.defaultBranch main
git config --global push.autoSetupRemote true

# SSH or HTTPS
if [[ "${USE_SSH}" == "1" ]]; then
  mkdir -p "$(dirname "${SSH_KEY_PATH}")"
  chmod 700 "$(dirname "${SSH_KEY_PATH}")"
  if [[ ! -f "${SSH_KEY_PATH}" ]]; then
    ssh-keygen -t ed25519 -N "" -f "${SSH_KEY_PATH}" -C "${GIT_USER_EMAIL}"
    echo "ADD THIS PUBLIC KEY TO GITHUB (Deploy key or Machine user):"
    cat "${SSH_KEY_PATH}.pub"
  fi
  mkdir -p /root/.ssh; touch /root/.ssh/known_hosts; chmod 600 /root/.ssh/known_hosts
  if ! ssh-keygen -F github.com >/dev/null; then
    ssh-keyscan -t rsa,ecdsa github.com >> /root/.ssh/known_hosts 2>/dev/null || true
  fi
  export GIT_SSH_COMMAND="ssh -i ${SSH_KEY_PATH} -o StrictHostKeyChecking=yes"
else
  if [[ -n "${GH_TOKEN:-}" ]]; then
    git config --global credential.helper store
    cat > /root/.git-credentials <<CREDS
https://${GH_TOKEN}:x-oauth-basic@github.com
CREDS
    chmod 600 /root/.git-credentials
  fi
fi

# Clone or update
if [[ ! -d "${SRC_DIR}/.git" ]]; then
  git clone --branch "${GIT_BRANCH}" --depth 1 "${GIT_REPO}" "${SRC_DIR}"
else
  git -C "${SRC_DIR}" fetch --depth 1 origin "${GIT_BRANCH}" || true
  git -C "${SRC_DIR}" reset --hard "origin/${GIT_BRANCH}" || true
fi

# Resolve requirements
REQ_PATH="${SRC_DIR}/${REQS_FILE}"
if [[ ! -f "${REQ_PATH}" && -n "${API_DIR}" ]]; then
  REQ_PATH="${SRC_DIR}/${API_DIR}/${REQS_FILE}"
fi
if [[ ! -f "${REQ_PATH}" ]]; then
  echo "[!] Requirements file not found: ${REQS_FILE}"; exit 2
fi

# (Re)build venv
REQ_HASH_FILE="${BASE}/.req_hash"
NEW_HASH="$(sha256sum "${REQ_PATH}" | awk '{print $1}')"
OLD_HASH="$(cat "${REQ_HASH_FILE}" 2>/dev/null || true)"

if [[ ! -d "${VENV}" || "${NEW_HASH}" != "${OLD_HASH}" ]]; then
  python3 -m venv "${VENV}"
  source "${VENV}/bin/activate"
  python -m pip install --upgrade pip wheel setuptools
  if ! pip install --no-index --find-links="${WHEELHOUSE}" -r "${REQ_PATH}"; then
    pip install -r "${REQ_PATH}"
    pip wheel -w "${WHEELHOUSE}" -r "${REQ_PATH}" || true
  fi
  echo "${NEW_HASH}" > "${REQ_HASH_FILE}"
else
  source "${VENV}/bin/activate"
fi

# Start vLLM
python -m vllm.entrypoints.openai.api_server   --model "${MODEL_NAME}"   --host 0.0.0.0 --port "${PORT_VLLM}"   --dtype bfloat16   --max-model-len 4096   --gpu-memory-utilization "${VLLM_GPU_UTIL}"   > "${LOG_DIR}/vllm.log" 2>&1 &

# Wait for vLLM
for i in {1..90}; do
  if curl -fsS "http://127.0.0.1:${PORT_VLLM}/v1/models" >/dev/null; then
    break
  fi
  sleep 2
done

export VLLM_BASE_URL="http://127.0.0.1:${PORT_VLLM}/v1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# Start API
APP_DIR="${SRC_DIR}/${API_DIR}"
cd "${APP_DIR}"
APP_IMPORT="apps.api_gpu.main:app"
EXTRA=""; if [[ "${RELOAD}" == "1" ]]; then EXTRA="--reload"; fi
exec "${VENV}/bin/uvicorn" ${APP_IMPORT} --host 0.0.0.0 --port "${PORT_API}" --workers 1 ${EXTRA}

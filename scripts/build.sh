#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-minicpm5-1b-chat:latest}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/.models/MiniCPM5-1B}"
EMPTY_MODEL_CONTEXT="${ROOT_DIR}/.build/empty-model"

cd "${ROOT_DIR}"

if [[ -z "${CHAT_PASSWORD:-}" ]]; then
  read -r -s -p "Set CHAT_PASSWORD for container login: " CHAT_PASSWORD
  echo
fi

if [[ -z "${CHAT_PASSWORD}" ]]; then
  echo "CHAT_PASSWORD must not be empty." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to generate the password hash." >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to build the container image." >&2
  exit 1
fi

"${ROOT_DIR}/scripts/prepare_model.sh"

VENV="${ROOT_DIR}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi
"${VENV}/bin/pip" install -q bcrypt huggingface_hub

CHAT_PASSWORD_HASH="$(
  CHAT_PASSWORD="${CHAT_PASSWORD}" "${VENV}/bin/python" - <<'PY'
import bcrypt
import os

password = os.environ["CHAT_PASSWORD"].encode("utf-8")
print(bcrypt.hashpw(password, bcrypt.gensalt(rounds=12)).decode("utf-8"))
PY
)"

export CHAT_PASSWORD_HASH

USE_LOCAL_MODEL=0
MODEL_BUILD_CONTEXT="${EMPTY_MODEL_CONTEXT}"
mkdir -p "${EMPTY_MODEL_CONTEXT}"

if "${VENV}/bin/python" -m minicpm_container.model_cache check "${MODEL_DIR}"; then
  USE_LOCAL_MODEL=1
  MODEL_BUILD_CONTEXT="${MODEL_DIR}"
  echo "Building ${IMAGE_NAME} using local model at ${MODEL_DIR} (skipping download)..."
else
  echo "Building ${IMAGE_NAME} (model download may take several minutes)..."
fi

BUILD_ARGS=(
  --build-arg "CHAT_PASSWORD_HASH=${CHAT_PASSWORD_HASH}"
  --build-arg "USE_LOCAL_MODEL=${USE_LOCAL_MODEL}"
  --build-context "modeldir=${MODEL_BUILD_CONTEXT}"
  -t "${IMAGE_NAME}"
  "${ROOT_DIR}"
)

if ! docker build "${BUILD_ARGS[@]}"; then
  echo "Docker build failed." >&2
  exit 1
fi

if [[ "${USE_LOCAL_MODEL}" == "0" ]]; then
  echo "Persisting model to ${MODEL_DIR} for future builds..."
  "${VENV}/bin/pip" install -q -e ".[dev]"
  if ! "${VENV}/bin/python" -m minicpm_container.model_cache ensure \
    "${MODEL_DIR}" \
    --image "${IMAGE_NAME}"; then
    echo "Warning: could not persist model to ${MODEL_DIR}." >&2
  fi
fi

echo "Build complete: ${IMAGE_NAME}"

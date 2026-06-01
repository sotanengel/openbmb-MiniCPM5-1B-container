#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/.models/MiniCPM5-1B}"
IMAGE_NAME="${IMAGE_NAME:-minicpm5-1b-chat:latest}"
MODEL_ID="${MODEL_ID:-openbmb/MiniCPM5-1B}"

cd "${ROOT_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to prepare the local model directory." >&2
  exit 1
fi

VENV="${ROOT_DIR}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi

"${VENV}/bin/pip" install -q -e ".[dev]" huggingface_hub

export MODEL_DIR
"${VENV}/bin/python" -m minicpm_container.model_cache ensure \
  "${MODEL_DIR}" \
  --repo-id "${MODEL_ID}" \
  --image "${IMAGE_NAME}" \
  || true

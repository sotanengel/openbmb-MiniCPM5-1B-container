#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-minicpm5-1b-chat:latest}"

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

VENV="${ROOT_DIR}/.venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi
"${VENV}/bin/pip" install -q bcrypt

CHAT_PASSWORD_HASH="$(
  CHAT_PASSWORD="${CHAT_PASSWORD}" "${VENV}/bin/python" - <<'PY'
import bcrypt
import os

password = os.environ["CHAT_PASSWORD"].encode("utf-8")
print(bcrypt.hashpw(password, bcrypt.gensalt(rounds=12)).decode("utf-8"))
PY
)"

export CHAT_PASSWORD_HASH

echo "Building ${IMAGE_NAME} (model download may take several minutes)..."
if ! docker build \
  --build-arg "CHAT_PASSWORD_HASH=${CHAT_PASSWORD_HASH}" \
  -t "${IMAGE_NAME}" \
  "${ROOT_DIR}"; then
  echo "Docker build failed." >&2
  exit 1
fi

echo "Build complete: ${IMAGE_NAME}"

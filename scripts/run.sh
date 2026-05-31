#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-minicpm5-1b-chat}"

cd "${ROOT_DIR}"

if ! docker image inspect minicpm5-1b-chat:latest >/dev/null 2>&1; then
  echo "Image minicpm5-1b-chat:latest not found. Run ./scripts/build.sh first." >&2
  exit 1
fi

docker compose up -d

ready=0
while [ "${ready}" -lt 600 ]; do
  if docker exec "${CONTAINER_NAME}" test -S /run/model.sock 2>/dev/null; then
    break
  fi
  ready=$((ready + 1))
  sleep 2
done

if ! docker exec "${CONTAINER_NAME}" test -S /run/model.sock 2>/dev/null; then
  echo "Model server did not become ready. Check logs with: docker logs ${CONTAINER_NAME}" >&2
  exit 1
fi

echo "Container is ready. Starting secure chat session..."
exec docker exec -u chat -it \
  -e LANG=C.UTF-8 \
  -e LC_ALL=C.UTF-8 \
  -e PYTHONIOENCODING=utf-8 \
  "${CONTAINER_NAME}" chat-login

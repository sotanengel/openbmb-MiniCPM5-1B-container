#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-minicpm5-1b-chat:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-minicpm5-1b-chat}"

cd "${ROOT_DIR}"

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Image ${IMAGE_NAME} not found. Run ./scripts/build.sh first." >&2
  exit 1
fi

echo "Syncing source into ${IMAGE_NAME}..."
if ! docker build -f docker/Dockerfile.sync -t "${IMAGE_NAME}" "${ROOT_DIR}"; then
  echo "Sync build failed." >&2
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Recreating ${CONTAINER_NAME}..."
  docker compose up -d --force-recreate
else
  echo "Starting ${CONTAINER_NAME}..."
  docker compose up -d
fi

ready=0
while [ "${ready}" -lt 300 ]; do
  if docker exec "${CONTAINER_NAME}" test -S /run/model.sock 2>/dev/null; then
    echo "Container is ready with updated code."
    exit 0
  fi
  ready=$((ready + 1))
  sleep 2
done

echo "Model server did not become ready. Check logs with: docker logs ${CONTAINER_NAME}" >&2
exit 1

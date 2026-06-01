#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-minicpm5-1b-chat:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-minicpm5-1b-chat}"

cd "${ROOT_DIR}"

USE_NETWORK=0
CHAT_LOGIN_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --network)
      USE_NETWORK=1
      shift
      ;;
    --)
      shift
      CHAT_LOGIN_ARGS=("$@")
      break
      ;;
    *)
      CHAT_LOGIN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "${MINICPM_NETWORK:-0}" = "1" ]; then
  USE_NETWORK=1
fi

USE_GPU_COMPOSE=0
if [ "${MINICPM_GPU:-}" = "0" ]; then
  USE_GPU_COMPOSE=0
elif [ "${MINICPM_GPU:-}" = "1" ]; then
  USE_GPU_COMPOSE=1
elif docker image inspect "${IMAGE_NAME:-minicpm5-1b-chat:latest}" --format '{{index .Config.Labels "minicpm.inference"}}' 2>/dev/null | grep -qx 'gpu'; then
  USE_GPU_COMPOSE=1
fi

COMPOSE_FILES=(-f docker-compose.yml)
if [ "${USE_GPU_COMPOSE}" -eq 1 ]; then
  COMPOSE_FILES+=(-f docker-compose.gpu.yml)
  echo "Starting with NVIDIA GPU inference enabled."
fi
if [ "${USE_NETWORK}" -eq 1 ]; then
  COMPOSE_FILES+=(-f docker-compose.network.yml)
  echo "Starting with network egress enabled (http_get / web_search)."
fi

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Image ${IMAGE_NAME} not found. Run ./scripts/build.sh first." >&2
  exit 1
fi

docker compose "${COMPOSE_FILES[@]}" up -d

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
EXEC_ENV=(
  -e LANG=C.UTF-8
  -e LC_ALL=C.UTF-8
  -e PYTHONIOENCODING=utf-8
)
if [ "${USE_NETWORK}" -eq 1 ]; then
  EXEC_ENV+=(-e CHAT_NETWORK_ENABLED=1)
fi

if ((${#CHAT_LOGIN_ARGS[@]} > 0)); then
  exec docker exec -u chat -it \
    "${EXEC_ENV[@]}" \
    "${CONTAINER_NAME}" chat-login "${CHAT_LOGIN_ARGS[@]}"
else
  exec docker exec -u chat -it \
    "${EXEC_ENV[@]}" \
    "${CONTAINER_NAME}" chat-login
fi

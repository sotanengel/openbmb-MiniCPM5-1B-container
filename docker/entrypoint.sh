#!/bin/sh
set -eu

MODEL_PID=""

cleanup() {
  if [ -n "${MODEL_PID}" ] && kill -0 "${MODEL_PID}" 2>/dev/null; then
    kill "${MODEL_PID}" 2>/dev/null || true
    wait "${MODEL_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

mkdir -p /run
chown model:chat /run
chmod 775 /run

gosu model model-server &
MODEL_PID=$!

ready=0
while [ "${ready}" -lt 60 ]; do
  if [ -S /run/model.sock ]; then
    break
  fi
  if ! kill -0 "${MODEL_PID}" 2>/dev/null; then
    echo "model server exited unexpectedly" >&2
    exit 1
  fi
  ready=$((ready + 1))
  sleep 1
done

if [ ! -S /run/model.sock ]; then
  echo "model server socket not ready" >&2
  exit 1
fi

exec sleep infinity

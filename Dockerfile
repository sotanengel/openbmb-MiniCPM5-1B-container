# syntax=docker/dockerfile:1.4

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install ".[inference]" \
    && /opt/venv/bin/pip install huggingface_hub

ARG USE_LOCAL_MODEL=0
ARG MODEL_ID=openbmb/MiniCPM5-1B

RUN mkdir -p /models/MiniCPM5-1B

RUN --mount=type=bind,from=modeldir,source=.,target=/mnt/model,readonly \
    if [ "${USE_LOCAL_MODEL}" = "1" ]; then \
      cp -a /mnt/model/. /models/MiniCPM5-1B/; \
    fi

RUN <<EOF
set -eu
if [ -f /models/MiniCPM5-1B/config.json ]; then
  echo "Using model from MODEL_DIR (skipping download)"
else
  MODEL_ID="${MODEL_ID}" /opt/venv/bin/python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ['MODEL_ID'], local_dir='/models/MiniCPM5-1B')"
fi
EOF


FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder-gpu

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md ./
COPY src ./src

RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124 \
    && /opt/venv/bin/pip install "transformers>=5.6" "accelerate>=0.33" \
    && /opt/venv/bin/pip install .


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MODEL_PATH=/models/MiniCPM5-1B \
    MODEL_SOCKET_PATH=/run/model.sock

ARG CHAT_PASSWORD_HASH
ENV CHAT_PASSWORD_HASH=${CHAT_PASSWORD_HASH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1001 model \
    && useradd --uid 1001 --gid model --system --shell /usr/sbin/nologin model \
    && groupadd --gid 1002 chat \
    && useradd --uid 1002 --gid chat --system --shell /usr/sbin/nologin chat \
    && usermod -aG chat model

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models /models
COPY pyproject.toml README.md ./
COPY src ./src
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod 755 /usr/local/bin/entrypoint.sh \
    && pip install -e . \
    && chmod -R o-rwx /models \
    && chown -R model:model /models \
    && chmod 750 /models /models/MiniCPM5-1B

WORKDIR /app
USER model
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]


FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime-gpu

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    MODEL_PATH=/models/MiniCPM5-1B \
    MODEL_SOCKET_PATH=/run/model.sock \
    MODEL_USE_GPU=1

ARG CHAT_PASSWORD_HASH
ENV CHAT_PASSWORD_HASH=${CHAT_PASSWORD_HASH}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && groupadd --gid 1001 model \
    && useradd --uid 1001 --gid model --system --shell /usr/sbin/nologin model \
    && groupadd --gid 1002 chat \
    && useradd --uid 1002 --gid chat --system --shell /usr/sbin/nologin chat \
    && usermod -aG chat model

COPY --from=builder-gpu /opt/venv /opt/venv
COPY --from=builder /models /models
COPY pyproject.toml README.md ./
COPY src ./src
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod 755 /usr/local/bin/entrypoint.sh \
    && pip install -e . \
    && chmod -R o-rwx /models \
    && chown -R model:model /models \
    && chmod 750 /models /models/MiniCPM5-1B

WORKDIR /app
USER model
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

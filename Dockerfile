# syntax=docker/dockerfile:1

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

ARG MODEL_ID=openbmb/MiniCPM5-1B
RUN /opt/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openbmb/MiniCPM5-1B",
    local_dir="/models/MiniCPM5-1B",
)
PY


FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/models/MiniCPM5-1B \
    MODEL_SOCKET_PATH=/run/model.sock

ARG CHAT_PASSWORD_HASH
ENV CHAT_PASSWORD_HASH=${CHAT_PASSWORD_HASH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1001 model \
    && useradd --uid 1001 --gid model --system --shell /usr/sbin/nologin model \
    && groupadd --gid 1002 chat \
    && useradd --uid 1002 --gid chat --system --shell /usr/sbin/nologin chat

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
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

"""Unix socket model inference worker."""

from __future__ import annotations

import logging
import os
import socket
import sys
from pathlib import Path

from minicpm_container.inference_device import log_runtime_device_info
from minicpm_container.model_engine import ModelEngine
from minicpm_container.protocol import (
    DEFAULT_SOCKET_PATH,
    ChatRequest,
    ChatResponse,
    ProtocolError,
    recv_all,
    send_json,
)

logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/MiniCPM5-1B")
SOCKET_PATH = os.environ.get("MODEL_SOCKET_PATH", DEFAULT_SOCKET_PATH)
SOCKET_GROUP = os.environ.get("MODEL_SOCKET_GROUP", "chat")


def _configure_socket_permissions(socket_path: Path) -> None:
    socket_path.chmod(0o660)
    if hasattr(os, "chown"):
        import grp

        group = grp.getgrnam(SOCKET_GROUP)
        os.chown(socket_path, -1, group.gr_gid)


def _handle_client(connection: socket.socket, engine: ModelEngine) -> None:
    payload = recv_all(connection)

    if not payload:
        response = ChatResponse(content="", error="empty request")
        send_json(connection, response.to_json())
        return

    try:
        request = ChatRequest.from_json(payload.decode("utf-8"))
    except ProtocolError as exc:
        response = ChatResponse(content="", error=str(exc))
        send_json(connection, response.to_json())
        return

    response = engine.generate(request)
    send_json(connection, response.to_json())


def serve_forever(
    engine: ModelEngine,
    socket_path: str = SOCKET_PATH,
    backlog: int = 4,
) -> None:
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    engine.load()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))
        _configure_socket_permissions(path)
        server.listen(backlog)
        logger.info("Model server listening on %s", path)

        while True:
            connection, _ = server.accept()
            with connection:
                _handle_client(connection, engine)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    model_path = Path(MODEL_PATH)
    if not model_path.is_dir():
        logger.error("Model path does not exist: %s", model_path)
        raise SystemExit(1)

    log_runtime_device_info(logger)
    engine = ModelEngine(str(model_path))
    try:
        serve_forever(engine)
    except KeyboardInterrupt:
        logger.info("Shutting down model server")
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()

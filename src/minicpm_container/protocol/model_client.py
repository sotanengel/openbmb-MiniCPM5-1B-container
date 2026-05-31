"""Unix socket client for model server communication."""

from __future__ import annotations

from dataclasses import dataclass

from minicpm_container.protocol.messages import (
    DEFAULT_SOCKET_PATH,
    ChatRequest,
    ChatResponse,
    ProtocolError,
)
from minicpm_container.protocol.socket_io import recv_all, send_json


@dataclass
class ModelClient:
    socket_path: str = DEFAULT_SOCKET_PATH
    timeout_seconds: float = 300.0

    def send(self, request: ChatRequest) -> ChatResponse:
        import socket

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout_seconds)
            sock.connect(self.socket_path)
            send_json(sock, request.to_json())
            sock.shutdown(socket.SHUT_WR)
            payload = recv_all(sock)
        if not payload:
            raise ProtocolError("empty response from model server")
        return ChatResponse.from_json(payload.decode("utf-8"))

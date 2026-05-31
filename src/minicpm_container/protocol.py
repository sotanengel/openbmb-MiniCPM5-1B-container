"""JSON protocol for chat CLI <-> model server communication."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

MAX_MESSAGES = 64
MAX_MESSAGE_CHARS = 8192
MAX_NEW_TOKENS = 512
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_SOCKET_PATH = "/run/model.sock"


class ProtocolError(ValueError):
    """Raised when a request or response violates the protocol."""


@dataclass
class ChatRequest:
    messages: list[dict[str, str]]
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    enable_thinking: bool = False

    def to_json(self) -> str:
        return json.dumps(
            {
                "messages": self.messages,
                "max_new_tokens": self.max_new_tokens,
                "enable_thinking": self.enable_thinking,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, payload: str) -> ChatRequest:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProtocolError("invalid JSON payload") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatRequest:
        if not isinstance(data, dict):
            raise ProtocolError("payload must be a JSON object")

        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ProtocolError("messages must be a non-empty list")

        if len(messages) > MAX_MESSAGES:
            raise ProtocolError(f"messages exceed limit of {MAX_MESSAGES}")

        normalized: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ProtocolError(f"message at index {index} must be an object")
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant", "system"}:
                raise ProtocolError(f"invalid role at index {index}: {role!r}")
            if not isinstance(content, str) or not content.strip():
                raise ProtocolError(f"content at index {index} must be a non-empty string")
            if len(content) > MAX_MESSAGE_CHARS:
                raise ProtocolError(
                    f"content at index {index} exceeds {MAX_MESSAGE_CHARS} characters"
                )
            normalized.append({"role": role, "content": content})

        max_new_tokens = data.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ProtocolError("max_new_tokens must be a positive integer")
        if max_new_tokens > MAX_NEW_TOKENS:
            raise ProtocolError(f"max_new_tokens exceeds limit of {MAX_NEW_TOKENS}")

        enable_thinking = data.get("enable_thinking", False)
        if not isinstance(enable_thinking, bool):
            raise ProtocolError("enable_thinking must be a boolean")

        return cls(
            messages=normalized,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
        )


@dataclass
class ChatResponse:
    content: str
    error: str | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {"content": self.content}
        if self.error:
            payload["error"] = self.error
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> ChatResponse:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProtocolError("invalid JSON payload") from exc
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatResponse:
        if not isinstance(data, dict):
            raise ProtocolError("payload must be a JSON object")
        if "error" in data:
            error = data.get("error")
            if not isinstance(error, str) or not error:
                raise ProtocolError("error must be a non-empty string")
            return cls(content="", error=error)
        content = data.get("content")
        if not isinstance(content, str):
            raise ProtocolError("content must be a string")
        return cls(content=content)


@dataclass
class ModelClient:
    socket_path: str = DEFAULT_SOCKET_PATH
    timeout_seconds: float = 300.0

    def send(self, request: ChatRequest) -> ChatResponse:
        import socket

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout_seconds)
            sock.connect(self.socket_path)
            sock.sendall(request.to_json().encode("utf-8"))
            sock.shutdown(socket.SHUT_WR)
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        if not chunks:
            raise ProtocolError("empty response from model server")
        return ChatResponse.from_json(b"".join(chunks).decode("utf-8"))


@dataclass
class ConversationState:
    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def clear(self) -> None:
        self.messages.clear()

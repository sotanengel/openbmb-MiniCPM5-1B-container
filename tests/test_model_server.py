"""Tests for model server request handling."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock

from minicpm_container.model_server import ModelEngine, _handle_client
from minicpm_container.protocol import ChatRequest, ChatResponse


def test_model_engine_generate_returns_error_when_not_loaded() -> None:
    engine = ModelEngine("/tmp/model")
    response = engine.generate(
        ChatRequest(messages=[{"role": "user", "content": "hello"}]),
    )
    assert response.error == "model not loaded"


def test_handle_client_returns_protocol_error_for_invalid_payload() -> None:
    engine = MagicMock()
    client = MagicMock(spec=socket.socket)
    client.recv.side_effect = [b"{invalid", b""]

    _handle_client(client, engine)

    sent = client.sendall.call_args[0][0].decode("utf-8")
    response = ChatResponse.from_json(sent)
    assert response.error is not None
    engine.generate.assert_not_called()


def test_handle_client_generates_response() -> None:
    engine = MagicMock()
    request = ChatRequest(messages=[{"role": "user", "content": "hello"}])
    engine.generate.return_value = ChatResponse(content="hi there")

    client = MagicMock(spec=socket.socket)
    client.recv.side_effect = [request.to_json().encode("utf-8"), b""]

    _handle_client(client, engine)

    sent = client.sendall.call_args[0][0].decode("utf-8")
    response = ChatResponse.from_json(sent)
    assert response.content == "hi there"
    engine.generate.assert_called_once()

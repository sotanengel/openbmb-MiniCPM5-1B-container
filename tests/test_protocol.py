"""Tests for chat protocol validation."""

from __future__ import annotations

import pytest

from minicpm_container.protocol import (
    ChatRequest,
    ChatResponse,
    ConversationState,
    ProtocolError,
)


def test_chat_request_roundtrip() -> None:
    request = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        max_new_tokens=64,
    )
    restored = ChatRequest.from_json(request.to_json())
    assert restored.messages == request.messages
    assert restored.max_new_tokens == 64
    assert restored.enable_thinking is False


def test_chat_request_rejects_empty_messages() -> None:
    with pytest.raises(ProtocolError, match="non-empty list"):
        ChatRequest.from_dict({"messages": []})


def test_chat_request_rejects_invalid_role() -> None:
    with pytest.raises(ProtocolError, match="invalid role"):
        ChatRequest.from_dict({"messages": [{"role": "tool", "content": "x"}]})


def test_chat_request_rejects_oversized_content() -> None:
    payload = {"messages": [{"role": "user", "content": "x" * 9000}]}
    with pytest.raises(ProtocolError, match="exceeds"):
        ChatRequest.from_dict(payload)


def test_chat_request_rejects_excessive_max_new_tokens() -> None:
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_new_tokens": 9999,
    }
    with pytest.raises(ProtocolError, match="max_new_tokens exceeds"):
        ChatRequest.from_dict(payload)


def test_chat_response_error_payload() -> None:
    response = ChatResponse(content="", error="model unavailable")
    restored = ChatResponse.from_json(response.to_json())
    assert restored.error == "model unavailable"


def test_chat_response_content_payload() -> None:
    response = ChatResponse(content="Hello there")
    restored = ChatResponse.from_json(response.to_json())
    assert restored.content == "Hello there"
    assert restored.error is None


def test_conversation_state_tracks_messages() -> None:
    state = ConversationState()
    state.add_user_message("Hi")
    state.add_assistant_message("Hello")
    assert len(state.messages) == 2
    state.clear()
    assert state.messages == []


def test_invalid_json_raises_protocol_error() -> None:
    with pytest.raises(ProtocolError, match="invalid JSON"):
        ChatRequest.from_json("{not-json")

    with pytest.raises(ProtocolError, match="JSON object"):
        ChatResponse.from_json("[]")

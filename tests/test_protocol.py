"""Tests for chat protocol validation."""

from __future__ import annotations

import pytest

from minicpm_container.protocol import (
    DEFAULT_DO_SAMPLE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_NEW_TOKENS,
    ChatRequest,
    ChatResponse,
    ConversationState,
    ProtocolError,
)


def test_chat_request_roundtrip() -> None:
    request = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        max_new_tokens=64,
        tools=[{"type": "function", "function": {"name": "calculate", "parameters": {}}}],
    )
    restored = ChatRequest.from_json(request.to_json())
    assert restored.messages == request.messages
    assert restored.max_new_tokens == 64
    assert restored.enable_thinking is None
    assert restored.tools == request.tools


def test_chat_request_rejects_empty_messages() -> None:
    with pytest.raises(ProtocolError, match="non-empty list"):
        ChatRequest.from_dict({"messages": []})


def test_chat_request_accepts_tool_role() -> None:
    request = ChatRequest.from_dict({"messages": [{"role": "tool", "content": "result"}]})
    assert request.messages[0]["role"] == "tool"


def test_chat_request_rejects_invalid_role() -> None:
    with pytest.raises(ProtocolError, match="invalid role"):
        ChatRequest.from_dict({"messages": [{"role": "function", "content": "x"}]})


def test_chat_request_rejects_oversized_content() -> None:
    payload = {"messages": [{"role": "user", "content": "x" * 9000}]}
    with pytest.raises(ProtocolError, match="exceeds"):
        ChatRequest.from_dict(payload)


def test_chat_request_accepts_max_new_tokens_at_model_limit() -> None:
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    request = ChatRequest.from_dict(payload)
    assert request.max_new_tokens == MAX_NEW_TOKENS


def test_chat_request_rejects_excessive_max_new_tokens() -> None:
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_new_tokens": MAX_NEW_TOKENS + 1,
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


def test_chat_request_omits_enable_thinking_when_hybrid() -> None:
    request = ChatRequest(messages=[{"role": "user", "content": "Hello"}])
    payload = request.to_json()
    assert "enable_thinking" not in payload


def test_chat_request_roundtrip_with_sampling_fields() -> None:
    request = ChatRequest(
        messages=[{"role": "user", "content": "Hello"}],
        max_new_tokens=64,
        enable_thinking=True,
        do_sample=False,
        temperature=0.5,
        top_p=0.8,
    )
    restored = ChatRequest.from_json(request.to_json())
    assert restored.enable_thinking is True
    assert restored.do_sample is False
    assert restored.temperature == 0.5
    assert restored.top_p == 0.8


def test_chat_request_uses_sampling_defaults_when_omitted() -> None:
    request = ChatRequest.from_dict({"messages": [{"role": "user", "content": "hi"}]})
    assert request.temperature == DEFAULT_TEMPERATURE
    assert request.top_p == DEFAULT_TOP_P
    assert request.do_sample is DEFAULT_DO_SAMPLE


def test_chat_request_rejects_invalid_temperature() -> None:
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 3.0,
    }
    with pytest.raises(ProtocolError, match="temperature"):
        ChatRequest.from_dict(payload)


def test_chat_request_rejects_invalid_top_p() -> None:
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "top_p": 1.5,
    }
    with pytest.raises(ProtocolError, match="top_p"):
        ChatRequest.from_dict(payload)


def test_chat_request_rejects_invalid_do_sample() -> None:
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "do_sample": "yes",
    }
    with pytest.raises(ProtocolError, match="do_sample"):
        ChatRequest.from_dict(payload)


def test_sanitize_message_content_preserves_japanese() -> None:
    from minicpm_container.protocol import sanitize_message_content

    text = "名探偵コナンについて 解説して。"
    assert sanitize_message_content(text) == text


def test_sanitize_message_content_replaces_lone_surrogates() -> None:
    from minicpm_container.protocol import sanitize_message_content

    broken = "名探偵\ud800コナン"
    assert sanitize_message_content(broken) == "名探偵\ufffdコナン"


def test_chat_request_to_json_encodes_messages_with_surrogates() -> None:
    request = ChatRequest.from_dict(
        {"messages": [{"role": "user", "content": "名探偵\ud800コナン"}]}
    )
    request.to_json().encode("utf-8")


def test_conversation_state_sanitizes_user_message() -> None:
    state = ConversationState()
    state.add_user_message("名探偵\ud800コナン")
    assert state.messages[0]["content"] == "名探偵\ufffdコナン"


def test_conversation_state_messages_snapshot_is_copy() -> None:
    state = ConversationState()
    state.add_user_message("Hi")
    snapshot = state.messages_snapshot()
    snapshot.append({"role": "assistant", "content": "ignored"})
    assert len(state.messages) == 1


def test_conversation_state_rollback_last_message() -> None:
    state = ConversationState()
    state.add_user_message("Hi")
    state.rollback_last_message()
    assert state.messages == []


def test_conversation_state_has_tool_results_since() -> None:
    state = ConversationState()
    state.add_user_message("search")
    user_index = state.user_turn_index()
    assert state.has_tool_results_since(user_index) is False
    state.add_tool_message("result")
    assert state.has_tool_results_since(user_index) is True


def test_conversation_state_latest_tool_result_fallback() -> None:
    from minicpm_container.tools.limits import MAX_FALLBACK_TOOL_RESULT_CHARS

    state = ConversationState()
    state.add_tool_message("calculate: 42")
    assert state.latest_tool_result_fallback(MAX_FALLBACK_TOOL_RESULT_CHARS) == "calculate: 42"
    state.add_tool_message("error: failed")
    assert state.latest_tool_result_fallback(MAX_FALLBACK_TOOL_RESULT_CHARS) == "calculate: 42"


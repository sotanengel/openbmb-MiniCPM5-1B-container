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


def test_model_engine_generate_passes_sampling_parameters() -> None:
    engine = ModelEngine("/tmp/model")
    tokenizer = MagicMock()
    model = MagicMock()
    engine._tokenizer = tokenizer
    engine._model = model

    input_ids = MagicMock()
    input_ids.shape = [1, 4]
    template_output = MagicMock()
    template_output.to.return_value = {"input_ids": input_ids}
    tokenizer.apply_chat_template.return_value = template_output
    model.device = "cpu"

    generated_token = MagicMock()
    generated_ids = MagicMock()
    generated_ids.__getitem__.return_value = [generated_token]
    model.generate.return_value = [generated_ids]
    tokenizer.decode.return_value = "response"
    # decode_generated_text calls decode with skip_special_tokens=False

    request = ChatRequest(
        messages=[{"role": "user", "content": "hello"}],
        max_new_tokens=64,
        do_sample=False,
        temperature=0.5,
        top_p=0.8,
    )
    response = engine.generate(request)

    assert response.error is None
    assert response.content == "response"
    template_kwargs = tokenizer.apply_chat_template.call_args.kwargs
    assert "tools" not in template_kwargs
    assert "enable_thinking" not in template_kwargs
    generate_kwargs = model.generate.call_args.kwargs
    assert generate_kwargs["max_new_tokens"] == 64
    assert generate_kwargs["do_sample"] is False
    assert generate_kwargs["temperature"] == 0.5
    assert generate_kwargs["top_p"] == 0.8


def test_model_engine_generate_clamps_max_new_tokens_to_remaining_context() -> None:
    engine = ModelEngine("/tmp/model")
    engine._max_position_embeddings = 100
    tokenizer = MagicMock()
    model = MagicMock()
    engine._tokenizer = tokenizer
    engine._model = model

    input_ids = MagicMock()
    input_ids.shape = [1, 90]
    template_output = MagicMock()
    template_output.to.return_value = {"input_ids": input_ids}
    tokenizer.apply_chat_template.return_value = template_output
    model.device = "cpu"

    generated_token = MagicMock()
    generated_ids = MagicMock()
    generated_ids.__getitem__.return_value = [generated_token]
    model.generate.return_value = [generated_ids]
    tokenizer.decode.return_value = "short"

    request = ChatRequest(
        messages=[{"role": "user", "content": "hello"}],
        max_new_tokens=500,
    )
    engine.generate(request)

    generate_kwargs = model.generate.call_args.kwargs
    assert generate_kwargs["max_new_tokens"] == 10


def test_model_engine_passes_tools_to_chat_template() -> None:
    engine = ModelEngine("/tmp/model")
    tokenizer = MagicMock()
    model = MagicMock()
    engine._tokenizer = tokenizer
    engine._model = model

    input_ids = MagicMock()
    input_ids.shape = [1, 4]
    template_output = MagicMock()
    template_output.to.return_value = {"input_ids": input_ids}
    tokenizer.apply_chat_template.return_value = template_output
    model.device = "cpu"

    generated_token = MagicMock()
    generated_ids = MagicMock()
    generated_ids.__getitem__.return_value = [generated_token]
    model.generate.return_value = [generated_ids]
    tokenizer.decode.return_value = "ok"

    tools = [{"type": "function", "function": {"name": "calculate", "parameters": {}}}]
    request = ChatRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
    )
    engine.generate(request)

    template_kwargs = tokenizer.apply_chat_template.call_args.kwargs
    assert template_kwargs["tools"] == tools

"""Tests for agent tool loop."""

from __future__ import annotations

from unittest.mock import MagicMock

from minicpm_container.agent import run_agent_turn
from minicpm_container.generation_config import GenerationConfig
from minicpm_container.protocol import ChatResponse, ConversationState
from minicpm_container.tools.registry import get_tool_schemas


def test_agent_turn_without_tools_uses_single_shot() -> None:
    client = MagicMock()
    client.send.return_value = ChatResponse(content="Hello")
    state = ConversationState()
    config = GenerationConfig()

    text = run_agent_turn(client, config, state, "Hi")

    assert text == "Hello"
    assert len(state.messages) == 2
    client.send.assert_called_once()


def test_agent_turn_executes_tool_and_continues() -> None:
    schemas = get_tool_schemas(("calculate",))
    assert schemas is not None

    first = ChatResponse(
        content=('<function name="calculate">' '<param name="expression">1+1</param></function>'),
    )
    second = ChatResponse(content="The answer is 2.")
    client = MagicMock()
    client.send.side_effect = [first, second]

    state = ConversationState()
    config = GenerationConfig(enabled_tools=("calculate",))

    text = run_agent_turn(client, config, state, "What is 1+1?")

    assert text == "The answer is 2."
    assert client.send.call_count == 2
    roles = [message["role"] for message in state.messages]
    assert "tool" in roles

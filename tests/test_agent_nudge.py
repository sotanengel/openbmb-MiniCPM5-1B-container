"""Tests for agent tool-call nudge behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

from minicpm_container.agent import _should_nudge_for_tool_call, run_agent_turn
from minicpm_container.generation_config import GenerationConfig
from minicpm_container.protocol import ChatResponse, ConversationState
from minicpm_container.tools.registry import get_tool_schemas


def test_should_nudge_for_prose_tool_intent() -> None:
    schemas = get_tool_schemas(("calculate",))
    assert schemas is not None
    assert _should_nudge_for_tool_call("I will use the calculate tool.", schemas)
    assert not _should_nudge_for_tool_call(
        '<function name="calculate"></function>',
        schemas,
    )


def test_agent_nudge_then_executes_tool() -> None:
    prose = ChatResponse(content="I will use the calculate tool to compute 17*23.")
    xml = ChatResponse(
        content=(
            '<function name="calculate">'
            '<param name="expression">17*23</param></function>'
        ),
    )
    final = ChatResponse(content="The result is 391.")
    client = MagicMock()
    client.send.side_effect = [prose, xml, final]

    state = ConversationState()
    config = GenerationConfig(enabled_tools=("calculate",))
    text = run_agent_turn(client, config, state, "What is 17 times 23?")

    assert text == "The result is 391."
    assert client.send.call_count == 3
    assert any(message["role"] == "tool" for message in state.messages)

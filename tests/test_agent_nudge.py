"""Tests for agent tool-call nudge behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

from minicpm_container.agent import _should_nudge_for_tool_xml, run_agent_turn
from minicpm_container.generation_config import GenerationConfig
from minicpm_container.protocol import ChatResponse, ConversationState


def test_should_nudge_for_prose_tool_intent() -> None:
    assert _should_nudge_for_tool_xml("I will use the calculate tool.")
    assert not _should_nudge_for_tool_xml('<function name="calculate"></function>')


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

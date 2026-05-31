"""Tests for agent tool-call nudge behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

from minicpm_container.agent import _should_nudge_for_tool_call, run_agent_turn
from minicpm_container.generation_config import GenerationConfig
from minicpm_container.protocol import ChatResponse, ConversationState
from minicpm_container.tools.registry import get_tool_schemas


def test_should_nudge_for_prose_tool_intent() -> None:
    from minicpm_container.agent import _has_tool_results_for_turn

    schemas = get_tool_schemas(("calculate",))
    assert schemas is not None
    state = ConversationState()
    state.add_user_message("What is 17 times 23?")
    user_turn_index = 0
    assert _should_nudge_for_tool_call(
        "I will use the calculate tool.",
        schemas,
        state,
        ("calculate",),
        user_turn_index,
    )
    assert not _should_nudge_for_tool_call(
        '<function name="calculate"></function>',
        schemas,
        state,
        ("calculate",),
        user_turn_index,
    )
    state.add_tool_message("calculate: 391")
    assert not _should_nudge_for_tool_call(
        "The answer is 391.",
        schemas,
        state,
        ("calculate",),
        user_turn_index,
    )
    assert _has_tool_results_for_turn(state, user_turn_index)


def test_should_not_nudge_for_unrelated_prose_with_tools_enabled() -> None:
    schemas = get_tool_schemas(("web_search",))
    assert schemas is not None
    state = ConversationState()
    state.add_user_message("名探偵コナンって何?")
    assert not _should_nudge_for_tool_call(
        "名探偵コナンは青山剛昌による推理漫画です。",
        schemas,
        state,
        ("web_search",),
        0,
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


def test_agent_prompts_for_answer_after_tool_only_response() -> None:
    tool_xml = ChatResponse(
        content=(
            '<function name="calculate">'
            '<param name="expression">1+1</param></function>'
        ),
    )
    final = ChatResponse(content="The answer is 2.")
    client = MagicMock()
    client.send.side_effect = [tool_xml, final]

    state = ConversationState()
    config = GenerationConfig(enabled_tools=("calculate",))
    text = run_agent_turn(client, config, state, "What is 1+1?")

    assert text == "The answer is 2."
    follow_up = [m for m in state.messages if m["role"] == "user"][-1]["content"]
    assert "Do not call any more tools" in follow_up

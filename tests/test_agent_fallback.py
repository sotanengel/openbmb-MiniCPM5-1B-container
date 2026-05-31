"""Tests for agent fallback when model returns empty text."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from minicpm_container.agent import run_agent_turn
from minicpm_container.generation_config import GenerationConfig
from minicpm_container.protocol import ChatResponse, ConversationState


def test_agent_returns_tool_result_when_model_answers_empty() -> None:
    tool_xml = ChatResponse(
        content=(
            '<function name="web_search">'
            '<param name="query">名探偵コナン</param></function>'
        ),
    )
    empty = ChatResponse(content="   ")
    client = MagicMock()
    client.send.side_effect = [tool_xml, empty]

    wiki_snippet = "--- Wikipedia (ja) ---\n名探偵コナン: 青山剛昌による推理漫画。"
    state = ConversationState()
    config = GenerationConfig(enabled_tools=("web_search",))
    with patch(
        "minicpm_container.agent.execute_tool",
        return_value=wiki_snippet,
    ):
        text = run_agent_turn(client, config, state, "名探偵コナンとは？")

    assert "名探偵コナン" in text
    assert "青山" in text or "漫画" in text

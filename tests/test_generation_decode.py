"""Tests for model output decoding."""

from __future__ import annotations

from unittest.mock import MagicMock

from minicpm_container.generation_decode import (
    decode_generated_text,
    strip_generation_artifacts,
    strip_thinking_blocks,
)


def test_strip_generation_artifacts_removes_empty_thinking() -> None:
    raw = "<think>\n\n</think>\n\nHello"
    assert strip_generation_artifacts(raw) == "Hello"


def test_strip_thinking_blocks_removes_nonempty_reasoning() -> None:
    raw = (
        "<think>\nPlan: call web_search\n</think>\n\n"
        '{"name":"web_search","arguments":{"query":"test"}}'
    )
    assert "Plan" not in strip_thinking_blocks(raw)
    assert "web_search" in strip_thinking_blocks(raw)


def test_decode_generated_text_uses_skip_special_tokens_false() -> None:
    tokenizer = MagicMock()
    tokenizer.decode.return_value = (
        "<tool_call><function name=\"calculate\">"
        "<param name=\"expression\">17*23</param></function></tool_call>"
    )
    text = decode_generated_text(tokenizer, [1, 2, 3])
    tokenizer.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=False)
    assert "<function name=\"calculate\">" in text
    assert "<param name=\"expression\">" in text

"""Tests for MiniCPM5 tool call parser."""

from __future__ import annotations

from minicpm_container.tools.parser import parse_tool_calls
from minicpm_container.tools.registry import get_tool_schemas


def _schemas() -> list[dict]:
    schemas = get_tool_schemas(("calculate", "count_text"))
    assert schemas is not None
    return schemas


def test_parse_tool_calls_returns_plain_text_when_no_xml() -> None:
    result = parse_tool_calls("Hello world", _schemas())
    assert result.normal_text == "Hello world"
    assert result.calls == []


def test_parse_tool_calls_extracts_calculate() -> None:
    text = (
        'Answer: <function name="calculate">'
        '<param name="expression">2+2</param></function>'
    )
    result = parse_tool_calls(text, _schemas())
    assert result.calls[0].name == "calculate"
    assert result.calls[0].arguments["expression"] == "2+2"
    assert "Answer:" in result.normal_text


def test_parse_tool_calls_ignores_unknown_function() -> None:
    text = '<function name="run_shell"><param name="cmd">id</param></function>'
    result = parse_tool_calls(text, _schemas())
    assert result.calls == []
    assert "<function" in result.normal_text

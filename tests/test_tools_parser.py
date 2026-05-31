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
    text = 'Answer: <function name="calculate">' '<param name="expression">2+2</param></function>'
    result = parse_tool_calls(text, _schemas())
    assert result.calls[0].name == "calculate"
    assert result.calls[0].arguments["expression"] == "2+2"
    assert "Answer:" in result.normal_text


def test_parse_tool_calls_with_tool_call_wrapper() -> None:
    text = (
        "Let me calculate."
        "<|im_sep|>"
        '<tool_call><function name="calculate">'
        '<param name="expression">17*23</param></function></tool_call>'
    )
    result = parse_tool_calls(text, _schemas())
    assert result.calls[0].name == "calculate"
    assert result.calls[0].arguments["expression"] == "17*23"
    assert "function" not in result.normal_text.lower() or "calculate" in result.normal_text


def test_parse_tool_calls_ignores_unknown_function() -> None:
    text = '<function name="run_shell"><param name="cmd">id</param></function>'
    result = parse_tool_calls(text, _schemas())
    assert result.calls == []
    assert "<function" in result.normal_text


def test_parse_tool_calls_json_web_search_after_thinking() -> None:
    schemas = get_tool_schemas(("web_search",))
    assert schemas is not None
    text = (
        "<think>\nEmit web_search for 名探偵コナン\n</think>\n\n"
        '{"name":"web_search","arguments":{"query":"名探偵コナン"}}'
    )
    result = parse_tool_calls(text, schemas)
    assert len(result.calls) == 1
    assert result.calls[0].name == "web_search"
    assert result.calls[0].arguments["query"] == "名探偵コナン"
    assert "web_search" not in result.normal_text


def test_parse_tool_calls_json_with_angle_bracket_prefix() -> None:
    schemas = get_tool_schemas(("calculate",))
    assert schemas is not None
    text = '<{"name":"calculate","arguments":{"expression":"2+2"}}'
    result = parse_tool_calls(text, schemas)
    assert result.calls[0].name == "calculate"
    assert result.calls[0].arguments["expression"] == "2+2"

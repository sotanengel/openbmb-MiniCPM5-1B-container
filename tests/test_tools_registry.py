"""Tests for tool registry."""

from __future__ import annotations

import pytest

from minicpm_container.tools.registry import (
    ALL_TOOL_IDS,
    get_tool_schemas,
    parse_enabled_tools,
)


def test_parse_enabled_tools_none_and_empty() -> None:
    assert parse_enabled_tools(None) == ()
    assert parse_enabled_tools("") == ()
    assert parse_enabled_tools("none") == ()


def test_parse_enabled_tools_deduplicates() -> None:
    assert parse_enabled_tools("calculate,calculate,http_get") == ("calculate", "http_get")


def test_parse_enabled_tools_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown tool"):
        parse_enabled_tools("run_shell")


def test_get_tool_schemas_returns_none_when_disabled() -> None:
    assert get_tool_schemas(()) is None


def test_get_tool_schemas_matches_enabled_ids() -> None:
    schemas = get_tool_schemas(("calculate", "web_search"))
    assert schemas is not None
    names = {item["function"]["name"] for item in schemas}
    assert names == {"calculate", "web_search"}
    assert names <= ALL_TOOL_IDS


def test_all_tool_ids_have_handlers() -> None:
    from minicpm_container.tools.registry import get_tool_definition

    for tool_id in ALL_TOOL_IDS:
        definition = get_tool_definition(tool_id)
        assert definition is not None
        assert definition.handler is not None
        assert definition.schema["function"]["name"] == tool_id

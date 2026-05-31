"""Tests for tool registry."""

from __future__ import annotations

import pytest

from minicpm_container.tools.registry import (
    ALL_TOOL_IDS,
    DEFAULT_ENABLED_TOOLS,
    get_tool_schemas,
    parse_enabled_tools,
    resolve_tool_selection,
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


def test_default_enabled_tools_matches_all_tool_ids() -> None:
    assert DEFAULT_ENABLED_TOOLS == tuple(sorted(ALL_TOOL_IDS))


def test_resolve_tool_selection_defaults_to_all() -> None:
    assert resolve_tool_selection(None) == DEFAULT_ENABLED_TOOLS


def test_resolve_tool_selection_none_disables_all() -> None:
    assert resolve_tool_selection("none") == ()


def test_resolve_tool_selection_allow_list() -> None:
    assert resolve_tool_selection("calculate,http_get") == ("calculate", "http_get")


def test_resolve_tool_selection_disable_from_default() -> None:
    result = resolve_tool_selection("-http_get,-web_search")
    assert "http_get" not in result
    assert "web_search" not in result
    assert set(result) == ALL_TOOL_IDS - {"http_get", "web_search"}


def test_resolve_tool_selection_mixed_allow_and_disable() -> None:
    assert resolve_tool_selection("calculate,http_get,-http_get") == ("calculate",)


def test_resolve_tool_selection_rejects_unknown_disable() -> None:
    with pytest.raises(ValueError, match="unknown tool"):
        resolve_tool_selection("-run_shell")


def test_all_tool_ids_have_handlers() -> None:
    from minicpm_container.tools.registry import get_tool_definition

    for tool_id in ALL_TOOL_IDS:
        definition = get_tool_definition(tool_id)
        assert definition is not None
        assert definition.handler is not None
        assert definition.schema["function"]["name"] == tool_id

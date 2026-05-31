"""Whitelisted tool definitions and OpenAI function schemas."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from minicpm_container.tools.handlers.builtin import (
    handle_calculate,
    handle_count_text,
    handle_current_datetime,
    handle_http_get,
    handle_web_search,
)
from minicpm_container.tools.handlers.convert_units import handle_convert_units

ToolHandler = Callable[[dict[str, Any]], str]

NETWORK_TOOL_IDS = frozenset({"http_get", "web_search"})
LOCAL_TOOL_IDS = frozenset({"calculate", "current_datetime", "count_text", "convert_units"})
ALL_TOOL_IDS = frozenset(NETWORK_TOOL_IDS | LOCAL_TOOL_IDS)
DEFAULT_ENABLED_TOOLS: tuple[str, ...] = tuple(sorted(ALL_TOOL_IDS))


@dataclass(frozen=True)
class ToolDefinition:
    schema: dict[str, Any]
    handler: ToolHandler


_TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "calculate": ToolDefinition(
        schema={
            "type": "function",
            "function": {
                "name": "calculate",
                "description": (
                    "Evaluate a safe arithmetic expression "
                    "(numbers, +, -, *, /, //, %, **, parentheses)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression to evaluate.",
                        },
                    },
                    "required": ["expression"],
                },
            },
        },
        handler=handle_calculate,
    ),
    "current_datetime": ToolDefinition(
        schema={
            "type": "function",
            "function": {
                "name": "current_datetime",
                "description": "Return the current date and time in UTC (ISO 8601).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
        handler=handle_current_datetime,
    ),
    "count_text": ToolDefinition(
        schema={
            "type": "function",
            "function": {
                "name": "count_text",
                "description": "Count characters, words, or lines in the given text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze."},
                        "unit": {
                            "type": "string",
                            "enum": ["characters", "words", "lines"],
                            "description": "What to count.",
                        },
                    },
                    "required": ["text", "unit"],
                },
            },
        },
        handler=handle_count_text,
    ),
    "convert_units": ToolDefinition(
        schema={
            "type": "function",
            "function": {
                "name": "convert_units",
                "description": (
                    "Convert a numeric value between supported units "
                    "(length, mass, temperature, bytes)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "description": "Numeric value to convert."},
                        "from_unit": {"type": "string", "description": "Source unit."},
                        "to_unit": {"type": "string", "description": "Target unit."},
                    },
                    "required": ["value", "from_unit", "to_unit"],
                },
            },
        },
        handler=handle_convert_units,
    ),
    "http_get": ToolDefinition(
        schema={
            "type": "function",
            "function": {
                "name": "http_get",
                "description": "Fetch a URL using HTTP GET only (public http/https, size-limited).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "http or https URL to fetch."},
                    },
                    "required": ["url"],
                },
            },
        },
        handler=handle_http_get,
    ),
    "web_search": ToolDefinition(
        schema={
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search via Wikipedia and DuckDuckGo Instant Answer (GET JSON). "
                    "Optional SearXNG when CHAT_SEARX_BASE_URL is set."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                    },
                    "required": ["query"],
                },
            },
        },
        handler=handle_web_search,
    ),
}


def get_tool_definition(tool_id: str) -> ToolDefinition | None:
    return _TOOL_DEFINITIONS.get(tool_id)


def _validate_tool_id(tool_id: str) -> str:
    if tool_id not in ALL_TOOL_IDS:
        supported = ", ".join(sorted(ALL_TOOL_IDS))
        raise ValueError(f"unknown tool id {tool_id!r}; supported: {supported}")
    return tool_id


def parse_enabled_tools(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    normalized = raw.strip().lower()
    if not normalized or normalized == "none":
        return ()
    ids: list[str] = []
    for part in normalized.split(","):
        tool_id = part.strip().lower()
        if not tool_id:
            continue
        _validate_tool_id(tool_id)
        if tool_id not in ids:
            ids.append(tool_id)
    return tuple(ids)


def resolve_tool_selection(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return DEFAULT_ENABLED_TOOLS
    normalized = raw.strip().lower()
    if not normalized:
        return DEFAULT_ENABLED_TOOLS
    if normalized == "none":
        return ()

    positive_ids: list[str] = []
    negative_ids: list[str] = []
    for part in normalized.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token.startswith("-"):
            tool_id = _validate_tool_id(token[1:])
            if tool_id not in negative_ids:
                negative_ids.append(tool_id)
        else:
            tool_id = _validate_tool_id(token)
            if tool_id not in positive_ids:
                positive_ids.append(tool_id)

    base = tuple(positive_ids) if positive_ids else DEFAULT_ENABLED_TOOLS
    if not base:
        return ()
    disabled = set(negative_ids)
    return tuple(tool_id for tool_id in base if tool_id not in disabled)


def get_tool_schemas(enabled_tools: tuple[str, ...]) -> list[dict[str, Any]] | None:
    if not enabled_tools:
        return None
    return [
        _TOOL_DEFINITIONS[tool_id].schema
        for tool_id in enabled_tools
        if tool_id in _TOOL_DEFINITIONS
    ]


def format_tools_help() -> str:
    lines = ["Available tools (--tools id1,id2 to allow, -id to disable, none to disable all):"]
    for tool_id in sorted(ALL_TOOL_IDS):
        schema = _TOOL_DEFINITIONS[tool_id].schema
        description = schema["function"]["description"]
        network = " [network]" if tool_id in NETWORK_TOOL_IDS else ""
        lines.append(f"  {tool_id}{network}: {description}")
    return "\n".join(lines)


def parse_schema_metadata(
    tool_schemas: list[dict[str, Any]],
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]], dict[str, dict[str, str]]]:
    """Extract tool names, allowed/required params, and property types from schemas."""
    names: set[str] = set()
    allowed_props: dict[str, set[str]] = {}
    required_props: dict[str, set[str]] = {}
    prop_types: dict[str, dict[str, str]] = {}

    for tool in tool_schemas:
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str):
            continue
        names.add(name)
        params = function.get("parameters")
        if not isinstance(params, dict):
            continue
        properties = params.get("properties")
        if isinstance(properties, dict):
            allowed_props[name] = set(properties.keys())
            prop_types[name] = {
                key: value.get("type", "string")
                for key, value in properties.items()
                if isinstance(value, dict)
            }
        required = params.get("required")
        if isinstance(required, list):
            required_props[name] = {item for item in required if isinstance(item, str)}
        else:
            required_props[name] = set()

    return names, allowed_props, required_props, prop_types

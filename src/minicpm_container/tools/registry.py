"""Whitelisted tool definitions and OpenAI function schemas."""

from __future__ import annotations

from typing import Any

NETWORK_TOOL_IDS = frozenset({"http_get", "web_search"})
LOCAL_TOOL_IDS = frozenset({"calculate", "current_datetime", "count_text", "convert_units"})
ALL_TOOL_IDS = frozenset(NETWORK_TOOL_IDS | LOCAL_TOOL_IDS)

_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "calculate": {
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
    "current_datetime": {
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
    "count_text": {
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
    "convert_units": {
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
    "http_get": {
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
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web via GET (DuckDuckGo HTML) and return extracted text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                },
                "required": ["query"],
            },
        },
    },
}


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
        if tool_id not in ALL_TOOL_IDS:
            supported = ", ".join(sorted(ALL_TOOL_IDS))
            raise ValueError(f"unknown tool id {tool_id!r}; supported: {supported}")
        if tool_id not in ids:
            ids.append(tool_id)
    return tuple(ids)


def get_tool_schemas(enabled_tools: tuple[str, ...]) -> list[dict[str, Any]] | None:
    if not enabled_tools:
        return None
    return [_TOOL_SCHEMAS[tool_id] for tool_id in enabled_tools if tool_id in _TOOL_SCHEMAS]


def format_tools_help() -> str:
    lines = ["Available tools (enable with --tools id1,id2):"]
    for tool_id in sorted(ALL_TOOL_IDS):
        schema = _TOOL_SCHEMAS[tool_id]
        description = schema["function"]["description"]
        network = " [network]" if tool_id in NETWORK_TOOL_IDS else ""
        lines.append(f"  {tool_id}{network}: {description}")
    return "\n".join(lines)

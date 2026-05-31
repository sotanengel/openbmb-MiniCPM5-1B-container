"""Shared helpers for tool handlers."""

from __future__ import annotations

from typing import Any

from minicpm_container.tools.limits import MAX_TOOL_RESULT_CHARS


def require_str(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def truncate_result(result: str) -> str:
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    return result[: MAX_TOOL_RESULT_CHARS - 1] + "…"

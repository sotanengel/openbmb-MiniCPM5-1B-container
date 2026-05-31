"""Execute whitelisted tools and return string results."""

from __future__ import annotations

from typing import Any

from minicpm_container.tools.calculate import CalculateError
from minicpm_container.tools.handlers.common import truncate_result
from minicpm_container.tools.http_client import HttpClientError
from minicpm_container.tools.registry import get_tool_definition


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    definition = get_tool_definition(name)
    if definition is None:
        return f"error: unknown tool {name!r}"

    if not isinstance(arguments, dict):
        return "error: arguments must be an object"

    try:
        return truncate_result(definition.handler(arguments))
    except (CalculateError, HttpClientError, ValueError) as exc:
        return f"error: {exc}"

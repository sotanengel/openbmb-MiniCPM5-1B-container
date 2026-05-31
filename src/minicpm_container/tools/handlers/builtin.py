"""Built-in whitelisted tool handlers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from minicpm_container.tools.calculate import safe_calculate
from minicpm_container.tools.handlers.common import require_str
from minicpm_container.tools.http_client import http_get
from minicpm_container.tools.limits import MAX_TEXT_ARG_CHARS
from minicpm_container.tools.search_providers import web_search


def handle_calculate(arguments: dict[str, Any]) -> str:
    expression = require_str(arguments, "expression")
    return safe_calculate(expression)


def handle_current_datetime(arguments: dict[str, Any]) -> str:
    del arguments
    return datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017


def handle_count_text(arguments: dict[str, Any]) -> str:
    text = require_str(arguments, "text")
    if len(text) > MAX_TEXT_ARG_CHARS:
        raise ValueError(f"text exceeds {MAX_TEXT_ARG_CHARS} characters")
    unit = require_str(arguments, "unit").lower()
    if unit == "characters":
        return str(len(text))
    if unit == "words":
        return str(len(text.split()))
    if unit == "lines":
        return str(len(text.splitlines()))
    raise ValueError("unit must be characters, words, or lines")


def handle_http_get(arguments: dict[str, Any]) -> str:
    url = require_str(arguments, "url")
    return http_get(url)


def handle_web_search(arguments: dict[str, Any]) -> str:
    query = require_str(arguments, "query")
    return web_search(query)

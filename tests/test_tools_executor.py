"""Tests for tool executor."""

from __future__ import annotations

import os
from unittest.mock import patch

from minicpm_container.tools.executor import execute_tool


def test_calculate_success() -> None:
    assert execute_tool("calculate", {"expression": "(2 + 3) * 4"}) == "20"


def test_calculate_rejects_unsafe_syntax() -> None:
    result = execute_tool("calculate", {"expression": "__import__('os')"})
    assert result.startswith("error:")


def test_current_datetime_returns_utc_iso() -> None:
    result = execute_tool("current_datetime", {})
    assert "T" in result
    assert result.endswith("+00:00") or result.endswith("Z")


def test_count_text_words() -> None:
    assert execute_tool("count_text", {"text": "one two three", "unit": "words"}) == "3"


def test_convert_units_km_to_m() -> None:
    assert execute_tool("convert_units", {"value": 1, "from_unit": "km", "to_unit": "m"}) == "1000"


def test_convert_units_celsius_to_fahrenheit() -> None:
    result = execute_tool(
        "convert_units",
        {"value": 0, "from_unit": "celsius", "to_unit": "fahrenheit"},
    )
    assert result == "32.0"


def test_unknown_tool_returns_error() -> None:
    assert execute_tool("run_shell", {}) == "error: unknown tool 'run_shell'"


def test_http_get_requires_network() -> None:
    with patch.dict(os.environ, {}, clear=True):
        result = execute_tool("http_get", {"url": "https://example.com"})
    assert "network" in result.lower()


def test_http_get_success_with_mock() -> None:
    class FakeResponse:
        headers = {}

        def getcode(self) -> int:
            return 200

        def read(self, size: int = -1) -> bytes:
            return b"hello"

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeOpener:
        def open(self, request: object, timeout: float = 0) -> FakeResponse:
            return FakeResponse()

    with (
        patch.dict(os.environ, {"CHAT_NETWORK_ENABLED": "1"}, clear=False),
        patch(
            "minicpm_container.tools.executor.http_get",
            return_value="page body",
        ),
    ):
        assert execute_tool("http_get", {"url": "https://example.com"}) == "page body"


def test_web_search_delegates() -> None:
    with (
        patch.dict(os.environ, {"CHAT_NETWORK_ENABLED": "1"}, clear=False),
        patch(
            "minicpm_container.tools.executor.web_search",
            return_value="results",
        ),
    ):
        assert execute_tool("web_search", {"query": "MiniCPM"}) == "results"

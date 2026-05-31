"""Tests for GET-only HTTP client."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from minicpm_container.tools.http_client import (
    HttpClientError,
    http_get,
    validate_url,
    web_search,
)


def test_validate_url_rejects_file_scheme() -> None:
    with pytest.raises(HttpClientError, match="http and https"):
        validate_url("file:///etc/passwd")


def test_validate_url_rejects_localhost() -> None:
    with pytest.raises(HttpClientError, match="not allowed"):
        validate_url("http://localhost/")


def test_validate_url_rejects_private_ip() -> None:
    with pytest.raises(HttpClientError, match="destination IP"):
        validate_url("http://127.0.0.1/")


def test_http_get_requires_network_flag() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(HttpClientError, match="network"):
            http_get("https://example.com")


def test_http_get_reads_response_with_mock_opener() -> None:
    class FakeResponse:
        headers = {}
        _payload = b"<p>Hello</p>"
        _done = False

        def getcode(self) -> int:
            return 200

        def read(self, size: int = -1) -> bytes:
            if self._done:
                return b""
            self._done = True
            return self._payload

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
            "minicpm_container.tools.http_client._resolve_host_ips",
            return_value=[__import__("ipaddress").ip_address("93.184.216.34")],
        ),
    ):
        body = http_get("https://example.com", opener=FakeOpener())
    assert "Hello" in body


def test_http_get_rejects_oversized_body() -> None:
    class FakeResponse:
        headers = {}
        _offset = 0

        def getcode(self) -> int:
            return 200

        def read(self, size: int = -1) -> bytes:
            if self._offset >= MAX_HTTP_RESPONSE_BYTES + 1:
                return b""
            chunk = b"x" * 4096
            self._offset += len(chunk)
            return chunk

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakeOpener:
        def open(self, request: object, timeout: float = 0) -> FakeResponse:
            return FakeResponse()

    from minicpm_container.tools.limits import MAX_HTTP_RESPONSE_BYTES

    with (
        patch.dict(os.environ, {"CHAT_NETWORK_ENABLED": "1"}, clear=False),
        patch(
            "minicpm_container.tools.http_client._resolve_host_ips",
            return_value=[__import__("ipaddress").ip_address("93.184.216.34")],
        ),
    ):
        with pytest.raises(HttpClientError, match="exceeds"):
            http_get("https://example.com", opener=FakeOpener())


def test_web_search_delegates_to_search_providers() -> None:
    with (
        patch.dict(os.environ, {"CHAT_NETWORK_ENABLED": "1"}, clear=False),
        patch(
            "minicpm_container.tools.search_providers.run_web_search",
            return_value="--- Wikipedia (ja) ---\n名探偵コナン",
        ),
    ):
        result = web_search("名探偵コナン")
    assert "名探偵コナン" in result

"""Tests for web search providers (JSON APIs)."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from minicpm_container.tools.http_client import HttpClientError
from minicpm_container.tools.search_providers import (
    run_web_search,
    search_ddg_instant,
    search_searx,
    search_wikipedia,
)

WIKI_JA_SEARCH_FIXTURE = {
    "query": {
        "search": [
            {
                "title": "名探偵コナン",
                "snippet": "『名探偵コナン』は青山剛昌による推理漫画。",
            }
        ]
    }
}

DDG_INSTANT_FIXTURE = {
    "AbstractText": "Detective Conan is a Japanese manga series.",
    "RelatedTopics": [
        {"Text": "Manga - Japanese comics"},
    ],
}

SEARX_FIXTURE = {
    "results": [
        {
            "title": "名探偵コナン",
            "url": "https://example.com/conan",
            "content": "推理漫画作品",
        }
    ]
}


def test_search_wikipedia_parses_ja_results() -> None:
    class FakeOpener:
        def open(self, request: object, timeout: float = 0) -> object:
            class Resp:
                headers = {}

                def getcode(self) -> int:
                    return 200

                _done = False

                def read(self, size: int = -1) -> bytes:
                    if self._done:
                        return b""
                    self._done = True
                    return json.dumps(WIKI_JA_SEARCH_FIXTURE).encode()

                def __enter__(self) -> Resp:
                    return self

                def __exit__(self, *args: object) -> None:
                    return None

            return Resp()

    with (
        patch.dict(os.environ, {"CHAT_NETWORK_ENABLED": "1"}, clear=False),
        patch(
            "minicpm_container.tools.http_client._resolve_host_ips",
            return_value=[__import__("ipaddress").ip_address("93.184.216.34")],
        ),
    ):
        text = search_wikipedia("名探偵コナン", "ja", opener=FakeOpener())
    assert "名探偵コナン" in text
    assert "漫画" in text or "青山" in text


def test_search_ddg_instant_extracts_abstract() -> None:
    class FakeOpener:
        def open(self, request: object, timeout: float = 0) -> object:
            class Resp:
                headers = {}

                def getcode(self) -> int:
                    return 200

                _done = False

                def read(self, size: int = -1) -> bytes:
                    if self._done:
                        return b""
                    self._done = True
                    return json.dumps(DDG_INSTANT_FIXTURE).encode()

                def __enter__(self) -> Resp:
                    return self

                def __exit__(self, *args: object) -> None:
                    return None

            return Resp()

    with (
        patch.dict(os.environ, {"CHAT_NETWORK_ENABLED": "1"}, clear=False),
        patch(
            "minicpm_container.tools.http_client._resolve_host_ips",
            return_value=[__import__("ipaddress").ip_address("93.184.216.34")],
        ),
    ):
        text = search_ddg_instant("detective conan", opener=FakeOpener())
    assert "Detective Conan" in text
    assert "Manga" in text


def test_search_searx_parses_results() -> None:
    class FakeOpener:
        def open(self, request: object, timeout: float = 0) -> object:
            class Resp:
                headers = {}

                def getcode(self) -> int:
                    return 200

                _done = False

                def read(self, size: int = -1) -> bytes:
                    if self._done:
                        return b""
                    self._done = True
                    return json.dumps(SEARX_FIXTURE).encode()

                def __enter__(self) -> Resp:
                    return self

                def __exit__(self, *args: object) -> None:
                    return None

            return Resp()

    with (
        patch.dict(os.environ, {"CHAT_NETWORK_ENABLED": "1"}, clear=False),
        patch(
            "minicpm_container.tools.http_client._resolve_host_ips",
            return_value=[__import__("ipaddress").ip_address("93.184.216.34")],
        ),
    ):
        text = search_searx(
            "名探偵コナン",
            "https://searx.example.com",
            opener=FakeOpener(),
        )
    assert "名探偵コナン" in text
    assert "推理漫画" in text


def test_fetch_allowlisted_json_rejects_unknown_host() -> None:
    from minicpm_container.tools.search_providers import fetch_allowlisted_json

    with patch.dict(os.environ, {"CHAT_NETWORK_ENABLED": "1"}, clear=False):
        with pytest.raises(HttpClientError, match="not an allowed search host"):
            fetch_allowlisted_json("https://evil.example.com/data.json")


def test_run_web_search_combines_providers() -> None:
    with (
        patch.dict(os.environ, {"CHAT_NETWORK_ENABLED": "1"}, clear=False),
        patch(
            "minicpm_container.tools.search_providers.search_wikipedia",
            side_effect=["Wikipedia hit", ""],
        ),
        patch(
            "minicpm_container.tools.search_providers.search_ddg_instant",
            return_value="DDG hit",
        ),
        patch(
            "minicpm_container.tools.search_providers.search_searx",
            return_value="",
        ),
        patch(
            "minicpm_container.tools.search_providers._searx_base_url",
            return_value=None,
        ),
    ):
        result = run_web_search("名探偵コナン")
    assert "Wikipedia" in result
    assert "Wikipedia hit" in result
    assert "DuckDuckGo" in result
    assert "DDG hit" in result


def test_run_web_search_raises_when_all_fail() -> None:
    with (
        patch.dict(os.environ, {"CHAT_NETWORK_ENABLED": "1"}, clear=False),
        patch(
            "minicpm_container.tools.search_providers.search_wikipedia",
            return_value="",
        ),
        patch(
            "minicpm_container.tools.search_providers.search_ddg_instant",
            return_value="",
        ),
        patch(
            "minicpm_container.tools.search_providers._searx_base_url",
            return_value=None,
        ),
    ):
        with pytest.raises(HttpClientError, match="no search results"):
            run_web_search("unknown query xyz")

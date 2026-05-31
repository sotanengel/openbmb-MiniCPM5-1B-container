"""GET-only JSON search providers for web_search (no HTML scraping)."""

from __future__ import annotations

import json
import os
import re
import urllib.parse
from typing import Any

from minicpm_container.tools.http_client import (
    HttpClientError,
    http_get,
    require_network,
    validate_url,
)
from minicpm_container.tools.limits import (
    MAX_SEARCH_QUERY_CHARS,
    MAX_WEB_SEARCH_RESULT_CHARS,
)

BUILTIN_SEARCH_HOSTS = frozenset(
    {
        "ja.wikipedia.org",
        "en.wikipedia.org",
        "api.duckduckgo.com",
    }
)

SEARX_BASE_URL_ENV = "CHAT_SEARX_BASE_URL"
_WIKI_HOSTS = {"ja": "ja.wikipedia.org", "en": "en.wikipedia.org"}
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text).strip()


def _validate_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise HttpClientError("search query must not be empty")
    if len(normalized) > MAX_SEARCH_QUERY_CHARS:
        raise HttpClientError(f"search query exceeds {MAX_SEARCH_QUERY_CHARS} characters")
    return normalized


def _allowed_hosts() -> frozenset[str]:
    hosts = set(BUILTIN_SEARCH_HOSTS)
    base = _searx_base_url()
    if base is not None:
        parsed = validate_url(base)
        hosts.add(parsed.hostname.lower())
    return frozenset(hosts)


def fetch_allowlisted_json(
    url: str,
    *,
    extra_hosts: frozenset[str] | None = None,
    opener: Any | None = None,
) -> dict[str, Any]:
    """GET JSON from a search-provider host on the allowlist."""
    require_network()
    preliminary = urllib.parse.urlparse(url)
    if preliminary.scheme not in {"http", "https"} or not preliminary.hostname:
        raise HttpClientError("only http and https URLs are allowed")
    hostname = preliminary.hostname.lower()
    allowed = _allowed_hosts() | (extra_hosts or frozenset())
    if hostname not in allowed:
        raise HttpClientError(f"hostname is not an allowed search host: {hostname}")

    validate_url(url)

    body = http_get(url, opener=opener)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HttpClientError("search provider returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise HttpClientError("search provider JSON must be an object")
    return payload


def _searx_base_url() -> str | None:
    raw = os.environ.get(SEARX_BASE_URL_ENV, "").strip()
    if not raw:
        return None
    return raw.rstrip("/")


def search_wikipedia(
    query: str,
    lang: str,
    *,
    opener: Any | None = None,
) -> str:
    host = _WIKI_HOSTS.get(lang)
    if host is None:
        return ""

    params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": "2",
            "format": "json",
        }
    )
    url = f"https://{host}/w/api.php?{params}"
    payload = fetch_allowlisted_json(url, opener=opener)
    search_block = payload.get("query", {})
    if not isinstance(search_block, dict):
        return ""
    hits = search_block.get("search", [])
    if not isinstance(hits, list) or not hits:
        return ""

    lines: list[str] = []
    for hit in hits[:2]:
        if not isinstance(hit, dict):
            continue
        title = str(hit.get("title", "")).strip()
        snippet = _strip_html(str(hit.get("snippet", "")).strip())
        if title:
            line = f"{title}"
            if snippet:
                line = f"{line}: {snippet}"
            lines.append(line)
    return "\n".join(lines)


def _format_ddg_related(topics: list[Any], *, depth: int = 0) -> list[str]:
    if depth > 2:
        return []
    lines: list[str] = []
    for item in topics:
        if not isinstance(item, dict):
            continue
        if "Topics" in item and isinstance(item["Topics"], list):
            lines.extend(_format_ddg_related(item["Topics"], depth=depth + 1))
            continue
        text = str(item.get("Text", "")).strip()
        if text:
            lines.append(text)
    return lines


def search_ddg_instant(
    query: str,
    *,
    opener: Any | None = None,
) -> str:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
    )
    url = f"https://api.duckduckgo.com/?{params}"
    payload = fetch_allowlisted_json(url, opener=opener)

    lines: list[str] = []
    abstract = str(payload.get("AbstractText", "")).strip()
    if abstract:
        lines.append(abstract)
    heading = str(payload.get("Heading", "")).strip()
    if heading and heading not in abstract:
        lines.append(f"Heading: {heading}")

    related = payload.get("RelatedTopics", [])
    if isinstance(related, list):
        for line in _format_ddg_related(related)[:5]:
            lines.append(line)

    return "\n".join(lines)


def search_searx(
    query: str,
    base_url: str,
    *,
    opener: Any | None = None,
) -> str:
    normalized_base = base_url.rstrip("/")
    parsed_base = validate_url(normalized_base)
    searx_host = frozenset({parsed_base.hostname.lower()})
    params = urllib.parse.urlencode({"q": query, "format": "json"})
    url = f"{normalized_base}/search?{params}"
    payload = fetch_allowlisted_json(url, extra_hosts=searx_host, opener=opener)
    results = payload.get("results", [])
    if not isinstance(results, list) or not results:
        return ""

    lines: list[str] = []
    for item in results[:5]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        link = str(item.get("url", "")).strip()
        parts = [p for p in (title, content, link) if p]
        if parts:
            lines.append(" — ".join(parts))
    return "\n".join(lines)


def run_web_search(
    query: str,
    *,
    opener: Any | None = None,
) -> str:
    """Run built-in search providers and return combined text."""
    normalized = _validate_query(query)
    require_network()

    sections: list[str] = []
    errors: list[str] = []

    for lang, label in (("ja", "Wikipedia (ja)"), ("en", "Wikipedia (en)")):
        try:
            text = search_wikipedia(normalized, lang, opener=opener)
            if text.strip():
                sections.append(f"--- {label} ---\n{text.strip()}")
        except HttpClientError as exc:
            errors.append(f"{label}: {exc}")

    try:
        ddg_text = search_ddg_instant(normalized, opener=opener)
        if ddg_text.strip():
            sections.append(f"--- DuckDuckGo Instant Answer ---\n{ddg_text.strip()}")
    except HttpClientError as exc:
        errors.append(f"DuckDuckGo: {exc}")

    searx_base = _searx_base_url()
    if searx_base is not None:
        try:
            searx_text = search_searx(normalized, searx_base, opener=opener)
            if searx_text.strip():
                sections.append(f"--- SearXNG ---\n{searx_text.strip()}")
        except HttpClientError as exc:
            errors.append(f"SearXNG: {exc}")

    if not sections:
        detail = "; ".join(errors) if errors else "all providers returned empty"
        raise HttpClientError(f"no search results ({detail})")

    combined = "\n\n".join(sections)
    if len(combined) > MAX_WEB_SEARCH_RESULT_CHARS:
        return combined[:MAX_WEB_SEARCH_RESULT_CHARS] + "…"
    return combined

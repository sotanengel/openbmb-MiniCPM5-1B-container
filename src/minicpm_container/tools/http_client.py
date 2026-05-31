"""GET-only HTTP client with SSRF protections."""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request

from minicpm_container.env_utils import parse_env_bool
from minicpm_container.tools.limits import (
    HTTP_MAX_REDIRECTS,
    HTTP_TIMEOUT_SECONDS,
    MAX_HTTP_RESPONSE_BYTES,
    MAX_URL_CHARS,
    NETWORK_REQUIRED_MESSAGE,
)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
NETWORK_ENV = "CHAT_NETWORK_ENABLED"

BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
    }
)

class HttpClientError(ValueError):
    """Raised when an HTTP request is rejected or fails safely."""


def is_network_enabled() -> bool:
    return parse_env_bool(os.environ.get(NETWORK_ENV), default=False) is True


def require_network() -> None:
    if not is_network_enabled():
        raise HttpClientError(NETWORK_REQUIRED_MESSAGE)


def _resolve_host_ips(hostname: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        infos = socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise HttpClientError(f"cannot resolve host: {hostname}") from exc

    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        ip_str = sockaddr[0]
        try:
            addresses.append(ipaddress.ip_address(ip_str))
        except ValueError:
            continue
    if not addresses:
        raise HttpClientError(f"no addresses for host: {hostname}")
    return addresses


def _is_blocked_ip(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if address.is_loopback:
        return True
    if address.is_private:
        return True
    if address.is_link_local:
        return True
    if address.is_multicast:
        return True
    if address.is_reserved:
        return True
    if str(address) == "169.254.169.254":
        return True
    return False


def validate_url(url: str) -> urllib.parse.ParseResult:
    if len(url) > MAX_URL_CHARS:
        raise HttpClientError(f"URL exceeds {MAX_URL_CHARS} characters")

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HttpClientError("only http and https URLs are allowed")
    if not parsed.hostname:
        raise HttpClientError("URL must include a hostname")
    if parsed.username or parsed.password:
        raise HttpClientError("URL must not include credentials")

    hostname = parsed.hostname.lower()
    if hostname in BLOCKED_HOSTNAMES:
        raise HttpClientError(f"hostname is not allowed: {hostname}")

    for address in _resolve_host_ips(hostname):
        if _is_blocked_ip(address):
            raise HttpClientError(f"destination IP is not allowed: {address}")

    return parsed


def _read_limited_body(response: urllib.response.addinfourl) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(4096)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_HTTP_RESPONSE_BYTES:
            raise HttpClientError(f"response exceeds {MAX_HTTP_RESPONSE_BYTES} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _default_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.HTTPHandler,
        urllib.request.HTTPSHandler,
    )


def http_get(
    url: str,
    *,
    opener: urllib.request.OpenerDirector | None = None,
) -> str:
    """Perform a GET request with SSRF checks and size limits."""
    require_network()

    current_url = url
    redirects = 0
    client = opener or _default_opener()

    while True:
        parsed = validate_url(current_url)
        request = urllib.request.Request(
            parsed.geturl(),
            method="GET",
            headers={"User-Agent": USER_AGENT},
        )
        try:
            with client.open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise HttpClientError("redirect missing Location header")
                    if redirects >= HTTP_MAX_REDIRECTS:
                        raise HttpClientError("too many redirects")
                    redirects += 1
                    current_url = urllib.parse.urljoin(current_url, location)
                    continue
                if status and status >= 400:
                    raise HttpClientError(f"HTTP status {status}")
                body = _read_limited_body(response)
        except HttpClientError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code in {301, 302, 303, 307, 308}:
                location = exc.headers.get("Location")
                if not location:
                    raise HttpClientError("redirect missing Location header") from exc
                if redirects >= HTTP_MAX_REDIRECTS:
                    raise HttpClientError("too many redirects") from exc
                redirects += 1
                current_url = urllib.parse.urljoin(current_url, location)
                continue
            raise HttpClientError(f"HTTP status {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise HttpClientError(f"request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise HttpClientError("request timed out") from exc

        charset = "utf-8"
        headers = getattr(response, "headers", None)
        if headers is not None and hasattr(headers, "get_content_charset"):
            detected = headers.get_content_charset()
            if detected:
                charset = detected
        try:
            return body.decode(charset, errors="replace")
        except LookupError:
            return body.decode("utf-8", errors="replace")


def web_search(
    query: str,
    *,
    opener: urllib.request.OpenerDirector | None = None,
) -> str:
    from minicpm_container.tools.search_providers import run_web_search

    return run_web_search(query, opener=opener)

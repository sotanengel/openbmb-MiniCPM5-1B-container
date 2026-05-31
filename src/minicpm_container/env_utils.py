"""Shared helpers for reading environment variables."""

from __future__ import annotations

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"0", "false", "no", "off"})


def parse_env_bool(raw: str | None, *, default: bool | None = None) -> bool | None:
    """Parse common truthy/falsy env strings; unknown values return default."""
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if not normalized:
        return default
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSY:
        return False
    return default

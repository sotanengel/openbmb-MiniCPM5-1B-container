"""Tests for environment variable parsing helpers."""

from __future__ import annotations

import pytest

from minicpm_container.env_utils import parse_env_bool


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("off", False),
    ],
)
def test_parse_env_bool_truthy_falsy(raw: str, expected: bool) -> None:
    assert parse_env_bool(raw) is expected


def test_parse_env_bool_empty_returns_default() -> None:
    assert parse_env_bool("") is None
    assert parse_env_bool(None) is None
    assert parse_env_bool("", default=False) is False
    assert parse_env_bool("  ", default=True) is True


def test_parse_env_bool_unknown_returns_default() -> None:
    assert parse_env_bool("maybe") is None
    assert parse_env_bool("maybe", default=True) is True

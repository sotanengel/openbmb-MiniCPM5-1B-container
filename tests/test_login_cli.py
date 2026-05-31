"""Tests for login CLI argument parsing."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from minicpm_container.login_cli import (
    CHAT_TOOLS_ENV,
    build_login_parser,
    normalize_login_argv,
    resolve_enabled_tools,
    warn_if_network_tools_without_egress,
)
from minicpm_container.tools.registry import DEFAULT_ENABLED_TOOLS


def test_build_login_parser_tools() -> None:
    args = build_login_parser().parse_args(["--tools", "calculate,http_get"])
    assert args.tools == "calculate,http_get"
    assert resolve_enabled_tools(args) == ("calculate", "http_get")


def test_resolve_enabled_tools_defaults_to_all() -> None:
    args = build_login_parser().parse_args([])
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_enabled_tools(args) == DEFAULT_ENABLED_TOOLS


def test_resolve_enabled_tools_disable_from_default() -> None:
    argv = normalize_login_argv(["--tools", "-http_get,-web_search"])
    args = build_login_parser().parse_args(argv)
    assert "http_get" not in resolve_enabled_tools(args)
    assert "web_search" not in resolve_enabled_tools(args)


def test_resolve_enabled_tools_disable_with_equals_form() -> None:
    args = build_login_parser().parse_args(["--tools=-http_get,-web_search"])
    assert "http_get" not in resolve_enabled_tools(args)
    assert "web_search" not in resolve_enabled_tools(args)


def test_resolve_enabled_tools_from_env() -> None:
    args = build_login_parser().parse_args([])
    with patch.dict(os.environ, {CHAT_TOOLS_ENV: "count_text"}, clear=False):
        assert resolve_enabled_tools(args) == ("count_text",)


def test_resolve_enabled_tools_env_disable() -> None:
    args = build_login_parser().parse_args([])
    with patch.dict(os.environ, {CHAT_TOOLS_ENV: "-http_get,-web_search"}, clear=False):
        result = resolve_enabled_tools(args)
        assert "http_get" not in result
        assert "web_search" not in result


def test_resolve_enabled_tools_cli_overrides_env() -> None:
    args = build_login_parser().parse_args(["--tools", "calculate"])
    with patch.dict(os.environ, {CHAT_TOOLS_ENV: "count_text"}, clear=False):
        assert resolve_enabled_tools(args) == ("calculate",)


def test_warn_if_network_tools_without_egress(capsys: pytest.CaptureFixture[str]) -> None:
    with patch.dict(os.environ, {}, clear=True):
        warn_if_network_tools_without_egress(("http_get",))
    captured = capsys.readouterr()
    assert "Warning" in captured.err

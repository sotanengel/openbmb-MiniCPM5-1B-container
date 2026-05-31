"""Tests for password authentication."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from minicpm_container.auth import (
    AuthenticationError,
    authenticate,
    hash_password,
    load_password_hash,
    verify_password,
)


def test_hash_and_verify_password_roundtrip() -> None:
    password = "secure-test-password"
    password_hash = hash_password(password)
    assert password_hash != password
    assert verify_password(password, password_hash)
    assert not verify_password("wrong-password", password_hash)


def test_hash_password_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        hash_password("")


def test_verify_password_rejects_invalid_hash() -> None:
    assert not verify_password("password", "not-a-bcrypt-hash")


def test_load_password_hash_requires_env() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(AuthenticationError, match="CHAT_PASSWORD_HASH"):
            load_password_hash()


def test_load_password_hash_reads_env() -> None:
    expected = hash_password("secret")
    with patch.dict(os.environ, {"CHAT_PASSWORD_HASH": expected}, clear=True):
        assert load_password_hash() == expected


def test_authenticate_success() -> None:
    password = "login-secret"
    password_hash = hash_password(password)
    with (
        patch.dict(os.environ, {"CHAT_PASSWORD_HASH": password_hash}, clear=True),
        patch("minicpm_container.auth.getpass", return_value=password),
    ):
        authenticate()


def test_authenticate_fails_after_max_attempts() -> None:
    password_hash = hash_password("correct")
    with (
        patch.dict(os.environ, {"CHAT_PASSWORD_HASH": password_hash}, clear=True),
        patch("minicpm_container.auth.getpass", return_value="wrong"),
        pytest.raises(AuthenticationError, match="too many failed"),
    ):
        authenticate(max_attempts=2)


def test_main_runs_login_flow_with_generation_config() -> None:
    password = "login-secret"
    password_hash = hash_password(password)
    config = MagicMock()
    with (
        patch.dict(os.environ, {"CHAT_PASSWORD_HASH": password_hash}, clear=True),
        patch("minicpm_container.auth.getpass", return_value=password),
        patch(
            "minicpm_container.generation_config.prompt_generation_config",
            return_value=config,
        ) as prompt,
        patch("minicpm_container.chat_cli.run_chat_loop") as run_chat,
    ):
        from minicpm_container.auth import main

        main()

    prompt.assert_called_once()
    run_chat.assert_called_once_with(config=config)

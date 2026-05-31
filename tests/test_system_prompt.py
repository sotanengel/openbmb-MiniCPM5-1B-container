"""Tests for response-language system prompt generation."""

from __future__ import annotations

import pytest

from minicpm_container.system_prompt import (
    RESPONSE_LANGUAGE_AUTO,
    SUPPORTED_RESPONSE_LANGUAGES,
    build_system_message,
    validate_response_language,
)


def test_supported_languages_include_auto_and_common_codes() -> None:
    assert RESPONSE_LANGUAGE_AUTO in SUPPORTED_RESPONSE_LANGUAGES
    assert "ja" in SUPPORTED_RESPONSE_LANGUAGES
    assert "en" in SUPPORTED_RESPONSE_LANGUAGES


def test_build_system_message_auto_instructs_matching_user_language() -> None:
    message = build_system_message(RESPONSE_LANGUAGE_AUTO)
    assert message is not None
    assert message["role"] == "system"
    assert "same language" in message["content"].lower()


def test_build_system_message_ja_instructs_japanese() -> None:
    message = build_system_message("ja")
    assert message is not None
    assert message["role"] == "system"
    assert "Japanese" in message["content"]


def test_build_system_message_en_instructs_english() -> None:
    message = build_system_message("en")
    assert message is not None
    assert "English" in message["content"]


def test_build_system_message_with_tools_appends_xml_instructions() -> None:
    message = build_system_message("en", enabled_tools=("calculate",))
    assert message is not None
    assert "<function name=" in message["content"]


def test_validate_response_language_accepts_supported_codes() -> None:
    for code in SUPPORTED_RESPONSE_LANGUAGES:
        assert validate_response_language(code) == code


def test_validate_response_language_normalizes_case() -> None:
    assert validate_response_language("JA") == "ja"
    assert validate_response_language(" Auto ") == RESPONSE_LANGUAGE_AUTO


def test_validate_response_language_rejects_unknown_code() -> None:
    with pytest.raises(ValueError, match="response_language"):
        validate_response_language("xx")

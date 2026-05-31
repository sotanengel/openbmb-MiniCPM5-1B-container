"""Tests for login-time generation configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from minicpm_container.generation_config import (
    CHAT_ENABLE_THINKING_ENV,
    RESPONSE_LANGUAGE_ENV,
    GenerationConfig,
)
from minicpm_container.generation_config_prompt import (
    format_generation_settings_help,
    prompt_generation_config,
)
from minicpm_container.protocol import (
    DEFAULT_DO_SAMPLE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES,
    MAX_NEW_TOKENS,
    MAX_TEMPERATURE,
    MAX_TOP_P,
    MIN_TEMPERATURE,
    MIN_TOP_P,
)
from minicpm_container.system_prompt import RESPONSE_LANGUAGE_AUTO


def test_generation_config_defaults() -> None:
    config = GenerationConfig.defaults()
    assert config.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert config.template_enable_thinking is None
    assert config.do_sample is DEFAULT_DO_SAMPLE
    assert config.temperature == DEFAULT_TEMPERATURE
    assert config.top_p == DEFAULT_TOP_P
    assert config.response_language == RESPONSE_LANGUAGE_AUTO


def test_generation_config_validate_rejects_invalid_max_new_tokens() -> None:
    config = GenerationConfig(max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        config.validate()


def test_generation_config_validate_accepts_model_max_new_tokens() -> None:
    config = GenerationConfig(max_new_tokens=MAX_NEW_TOKENS)
    config.validate()


def test_generation_config_validate_rejects_invalid_temperature() -> None:
    config = GenerationConfig(temperature=5.0)
    with pytest.raises(ValueError, match="temperature"):
        config.validate()


def test_generation_config_validate_rejects_invalid_response_language() -> None:
    config = GenerationConfig(response_language="invalid")
    with pytest.raises(ValueError, match="response_language"):
        config.validate()


def test_generation_config_to_chat_request() -> None:
    config = GenerationConfig(
        max_new_tokens=256,
        template_enable_thinking=True,
        do_sample=False,
        temperature=0.5,
        top_p=0.8,
        enabled_tools=("calculate",),
    )
    request = config.to_chat_request([{"role": "user", "content": "hello"}])
    assert request.max_new_tokens == 256
    assert request.enable_thinking is True
    assert request.do_sample is False
    assert request.temperature == 0.5
    assert request.top_p == 0.8
    assert request.tools is not None
    assert request.tools[0]["function"]["name"] == "calculate"
    assert request.messages[0]["role"] == "system"
    assert request.messages[1] == {"role": "user", "content": "hello"}


def test_generation_config_to_chat_request_prepends_system_for_empty_history() -> None:
    config = GenerationConfig(response_language="ja")
    request = config.to_chat_request([{"role": "user", "content": "こんにちは"}])
    assert request.messages[0]["role"] == "system"
    assert "Japanese" in request.messages[0]["content"]


def test_generation_config_summary() -> None:
    config = GenerationConfig.defaults()
    summary = config.summary()
    assert "max_new_tokens=128" in summary
    assert "thinking_mode=hybrid" in summary
    assert "response_language=auto" in summary


def test_format_generation_settings_help_includes_defaults_and_ranges() -> None:
    help_text = format_generation_settings_help()
    assert "max_new_tokens" in help_text
    assert "残りコンテキストまで自動調整" in help_text
    assert "thinking_mode" in help_text
    assert "enabled_tools" in help_text
    assert "do_sample" in help_text
    assert "temperature" in help_text
    assert "top_p" in help_text
    assert "response_language" in help_text
    assert RESPONSE_LANGUAGE_AUTO in help_text
    assert str(DEFAULT_MAX_NEW_TOKENS) in help_text
    assert str(MAX_NEW_TOKENS) in help_text
    assert str(DEFAULT_TEMPERATURE) in help_text
    assert f"{MIN_TEMPERATURE}" in help_text
    assert f"{MAX_TEMPERATURE}" in help_text
    assert str(DEFAULT_TOP_P) in help_text
    assert f"{MIN_TOP_P}" in help_text
    assert f"{MAX_TOP_P}" in help_text
    assert str(MAX_MESSAGES) in help_text
    assert str(MAX_MESSAGE_CHARS) in help_text
    assert "生成設定" in help_text


def test_prompt_generation_config_prints_help_before_prompts(capsys) -> None:
    with patch("builtins.input", side_effect=["", "", "", "", ""]):
        prompt_generation_config()
    captured = capsys.readouterr().out
    help_marker = f"有効範囲: 1〜{MAX_NEW_TOKENS}"
    summary_marker = "\n設定:"
    assert help_marker in captured
    assert summary_marker in captured
    assert captured.find(help_marker) < captured.find(summary_marker)


def test_prompt_generation_config_uses_defaults_on_empty_input() -> None:
    with patch("builtins.input", side_effect=["", "", "", "", ""]):
        config = prompt_generation_config()
    assert config == GenerationConfig.defaults()


def test_prompt_generation_config_uses_env_default_for_response_language() -> None:
    with patch.dict("os.environ", {RESPONSE_LANGUAGE_ENV: "ja"}):
        with patch("builtins.input", side_effect=["", "", "", "", ""]):
            config = prompt_generation_config()
    assert config.response_language == "ja"


def test_prompt_generation_config_accepts_custom_values() -> None:
    with patch(
        "builtins.input",
        side_effect=["256", "no", "0.5", "0.8", "en"],
    ):
        config = prompt_generation_config()
    assert config.max_new_tokens == 256
    assert config.template_enable_thinking is None
    assert config.do_sample is False
    assert config.temperature == 0.5
    assert config.top_p == 0.8
    assert config.response_language == "en"


def test_prompt_generation_config_hybrid_thinking_with_tools() -> None:
    with patch("builtins.input", side_effect=["", "", "", "", ""]):
        config = prompt_generation_config(enabled_tools=("web_search",))
    assert config.template_enable_thinking is None
    assert config.enabled_tools == ("web_search",)
    request = config.to_chat_request([{"role": "user", "content": "hi"}])
    assert request.enable_thinking is None
    assert "enable_thinking" not in request.to_json()


def test_resolve_template_enable_thinking_from_env() -> None:
    with patch.dict("os.environ", {CHAT_ENABLE_THINKING_ENV: "0"}):
        config = GenerationConfig.defaults()
    assert config.template_enable_thinking is False


def test_prompt_generation_config_retries_invalid_input() -> None:
    with patch(
        "builtins.input",
        side_effect=["abc", "256", "", "", "", "xx", "ja"],
    ):
        config = prompt_generation_config()
    assert config.max_new_tokens == 256
    assert config.response_language == "ja"

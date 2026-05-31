"""Tests for login-time generation configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from minicpm_container.generation_config import GenerationConfig, prompt_generation_config
from minicpm_container.protocol import (
    DEFAULT_DO_SAMPLE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)


def test_generation_config_defaults() -> None:
    config = GenerationConfig.defaults()
    assert config.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert config.enable_thinking is False
    assert config.do_sample is DEFAULT_DO_SAMPLE
    assert config.temperature == DEFAULT_TEMPERATURE
    assert config.top_p == DEFAULT_TOP_P


def test_generation_config_validate_rejects_invalid_max_new_tokens() -> None:
    config = GenerationConfig(max_new_tokens=0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        config.validate()


def test_generation_config_validate_rejects_invalid_temperature() -> None:
    config = GenerationConfig(temperature=5.0)
    with pytest.raises(ValueError, match="temperature"):
        config.validate()


def test_generation_config_to_chat_request() -> None:
    config = GenerationConfig(
        max_new_tokens=256,
        enable_thinking=True,
        do_sample=False,
        temperature=0.5,
        top_p=0.8,
    )
    request = config.to_chat_request([{"role": "user", "content": "hello"}])
    assert request.max_new_tokens == 256
    assert request.enable_thinking is True
    assert request.do_sample is False
    assert request.temperature == 0.5
    assert request.top_p == 0.8


def test_generation_config_summary() -> None:
    config = GenerationConfig.defaults()
    summary = config.summary()
    assert "max_new_tokens=128" in summary
    assert "enable_thinking=false" in summary


def test_prompt_generation_config_uses_defaults_on_empty_input() -> None:
    with patch("builtins.input", side_effect=["", "", "", "", ""]):
        config = prompt_generation_config()
    assert config == GenerationConfig.defaults()


def test_prompt_generation_config_accepts_custom_values() -> None:
    with patch(
        "builtins.input",
        side_effect=["256", "yes", "no", "0.5", "0.8"],
    ):
        config = prompt_generation_config()
    assert config.max_new_tokens == 256
    assert config.enable_thinking is True
    assert config.do_sample is False
    assert config.temperature == 0.5
    assert config.top_p == 0.8


def test_prompt_generation_config_retries_invalid_input() -> None:
    with patch(
        "builtins.input",
        side_effect=["abc", "256", "", "", "", ""],
    ):
        config = prompt_generation_config()
    assert config.max_new_tokens == 256

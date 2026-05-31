"""Session generation settings configured at login time."""

from __future__ import annotations

import os
from dataclasses import dataclass

from minicpm_container.env_utils import parse_env_bool
from minicpm_container.protocol import (
    DEFAULT_DO_SAMPLE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_NEW_TOKENS,
    MAX_TEMPERATURE,
    MAX_TOP_P,
    MIN_TEMPERATURE,
    MIN_TOP_P,
    ChatRequest,
)
from minicpm_container.system_prompt import (
    RESPONSE_LANGUAGE_AUTO,
    build_system_message,
    validate_response_language,
)
from minicpm_container.tools.registry import get_tool_schemas

RESPONSE_LANGUAGE_ENV = "CHAT_RESPONSE_LANGUAGE"
CHAT_ENABLE_THINKING_ENV = "CHAT_ENABLE_THINKING"
THINKING_MODE_HYBRID = "hybrid"


def resolve_template_enable_thinking() -> bool | None:
    """Map CHAT_ENABLE_THINKING to template flag; omitted (None) = official Hybrid mode."""
    return parse_env_bool(os.environ.get(CHAT_ENABLE_THINKING_ENV))


@dataclass
class GenerationConfig:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    template_enable_thinking: bool | None = None
    do_sample: bool = DEFAULT_DO_SAMPLE
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    response_language: str = RESPONSE_LANGUAGE_AUTO
    enabled_tools: tuple[str, ...] = ()

    @classmethod
    def defaults(cls) -> GenerationConfig:
        return cls(template_enable_thinking=resolve_template_enable_thinking())

    def validate(self) -> None:
        if self.max_new_tokens < 1 or self.max_new_tokens > MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
        if self.temperature < MIN_TEMPERATURE or self.temperature > MAX_TEMPERATURE:
            raise ValueError(f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
        if self.top_p < MIN_TOP_P or self.top_p > MAX_TOP_P:
            raise ValueError(f"top_p must be between {MIN_TOP_P} and {MAX_TOP_P}")
        validate_response_language(self.response_language)

    def to_chat_request(self, messages: list[dict[str, str]]) -> ChatRequest:
        self.validate()
        system_message = build_system_message(
            self.response_language,
            enabled_tools=self.enabled_tools,
        )
        request_messages = [system_message, *messages] if system_message else messages
        return ChatRequest(
            messages=request_messages,
            max_new_tokens=self.max_new_tokens,
            enable_thinking=self.template_enable_thinking,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            tools=get_tool_schemas(self.enabled_tools),
        )

    def summary(self) -> str:
        if self.template_enable_thinking is None:
            thinking_label = THINKING_MODE_HYBRID
        else:
            thinking_label = str(self.template_enable_thinking).lower()
        return (
            f"max_new_tokens={self.max_new_tokens}, "
            f"thinking_mode={thinking_label}, "
            f"do_sample={str(self.do_sample).lower()}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"response_language={self.response_language}, "
            f"enabled_tools={','.join(self.enabled_tools) if self.enabled_tools else 'none'}"
        )

"""Session generation settings configured at login time."""

from __future__ import annotations

import os
from dataclasses import dataclass

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
    ChatRequest,
)
from minicpm_container.system_prompt import (
    RESPONSE_LANGUAGE_AUTO,
    SUPPORTED_RESPONSE_LANGUAGES,
    build_system_message,
    validate_response_language,
)
from minicpm_container.tools.registry import (
    ALL_TOOL_IDS,
    format_tools_help,
    get_tool_schemas,
    parse_enabled_tools,
)

RESPONSE_LANGUAGE_ENV = "CHAT_RESPONSE_LANGUAGE"
CHAT_ENABLE_THINKING_ENV = "CHAT_ENABLE_THINKING"
THINKING_MODE_HYBRID = "hybrid"


def resolve_template_enable_thinking() -> bool | None:
    """Map CHAT_ENABLE_THINKING to template flag; omitted (None) = official Hybrid mode."""
    raw = os.environ.get(CHAT_ENABLE_THINKING_ENV, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def _bool_default_label(value: bool) -> str:
    return "yes" if value else "no"


def format_generation_settings_help() -> str:
    do_sample_default = _bool_default_label(DEFAULT_DO_SAMPLE)
    lines = [
        "",
        "生成設定（Enter でデフォルト）:",
        "",
        "  max_new_tokens — 1回の応答で生成する最大トークン数",
        f"    デフォルト: {DEFAULT_MAX_NEW_TOKENS}  有効範囲: 1〜{MAX_NEW_TOKENS}",
        "    プロンプトが長い場合、残りコンテキストまで自動調整されます",
        "",
        "  thinking_mode — Hybrid 思考（固定）",
        "    モデルが思考ブロックの要否を判断（公式 MiniCPM5 と同様）",
        "",
        "  do_sample — サンプリングの有無（no で greedy / 決定的生成）",
        f"    デフォルト: {do_sample_default}  入力: yes / no",
        "",
        "  temperature — サンプリング時の多様性（高いほどランダム）",
        f"    デフォルト: {DEFAULT_TEMPERATURE}  有効範囲: {MIN_TEMPERATURE}〜{MAX_TEMPERATURE}",
        "",
        "  top_p — nucleus sampling の累積確率上限",
        f"    デフォルト: {DEFAULT_TOP_P}  有効範囲: {MIN_TOP_P}〜{MAX_TOP_P}",
        "",
        "  response_language — 応答言語（auto でユーザー入力言語に追従）",
        (
            f"    デフォルト: {RESPONSE_LANGUAGE_AUTO}  入力: "
            f"{', '.join(sorted(SUPPORTED_RESPONSE_LANGUAGES))}"
        ),
        "",
        f"  （参考）会話上限: メッセージ数 {MAX_MESSAGES}、1メッセージ {MAX_MESSAGE_CHARS} 文字",
        "",
        "  enabled_tools — 有効化するツール ID（カンマ区切り、none で無効）",
        f"    デフォルト: none  利用可能: {', '.join(sorted(ALL_TOOL_IDS))}",
        "",
    ]
    return "\n".join(lines)


def print_generation_settings_help() -> None:
    print(format_generation_settings_help())


def _parse_bool(value: str, default: bool) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return default
    if normalized in {"y", "yes", "true", "1", "on"}:
        return True
    if normalized in {"n", "no", "false", "0", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _prompt_bool(label: str, default: bool) -> bool:
    default_label = "yes" if default else "no"
    while True:
        raw = input(f"  {label} [{default_label}]: ")
        try:
            return _parse_bool(raw, default)
        except ValueError:
            print("  yes/no で入力してください。")


def _prompt_int(label: str, default: int, minimum: int, maximum: int) -> int:
    while True:
        raw = input(f"  {label} [{default}]: ")
        if not raw.strip():
            return default
        try:
            value = int(raw.strip())
        except ValueError:
            print("  整数で入力してください。")
            continue
        if value < minimum or value > maximum:
            print(f"  {minimum}〜{maximum} の範囲で入力してください。")
            continue
        return value


def _prompt_float(label: str, default: float, minimum: float, maximum: float) -> float:
    while True:
        raw = input(f"  {label} [{default}]: ")
        if not raw.strip():
            return default
        try:
            value = float(raw.strip())
        except ValueError:
            print("  数値で入力してください。")
            continue
        if value < minimum or value > maximum:
            print(f"  {minimum}〜{maximum} の範囲で入力してください。")
            continue
        return value


def _load_default_response_language() -> str:
    env_value = os.environ.get(RESPONSE_LANGUAGE_ENV, "").strip()
    if not env_value:
        return RESPONSE_LANGUAGE_AUTO
    return validate_response_language(env_value)


def _prompt_response_language(default: str) -> str:
    supported = ", ".join(sorted(SUPPORTED_RESPONSE_LANGUAGES))
    while True:
        raw = input(f"  response_language [{default}]: ")
        if not raw.strip():
            return default
        try:
            return validate_response_language(raw)
        except ValueError:
            print(f"  次のいずれかで入力してください: {supported}")


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


def _prompt_enabled_tools(default: tuple[str, ...]) -> tuple[str, ...]:
    default_label = ",".join(default) if default else "none"
    supported = ", ".join(sorted(ALL_TOOL_IDS))
    while True:
        raw = input(f"  enabled_tools [{default_label}]: ")
        if not raw.strip():
            return default
        try:
            return parse_enabled_tools(raw)
        except ValueError as exc:
            print(f"  {exc}")
            print(f"  利用可能: {supported}")


def prompt_generation_config(
    *,
    enabled_tools: tuple[str, ...] | None = None,
    prompt_tools: bool = False,
) -> GenerationConfig:
    print("\nパスワード認証に成功しました。")
    print_generation_settings_help()
    default_response_language = _load_default_response_language()
    if enabled_tools is None:
        resolved_tools: tuple[str, ...] = ()
    else:
        resolved_tools = enabled_tools
    if prompt_tools:
        resolved_tools = _prompt_enabled_tools(resolved_tools)

    max_new_tokens = _prompt_int("max_new_tokens", DEFAULT_MAX_NEW_TOKENS, 1, MAX_NEW_TOKENS)
    do_sample = _prompt_bool("do_sample", DEFAULT_DO_SAMPLE)
    temperature = _prompt_float(
        "temperature", DEFAULT_TEMPERATURE, MIN_TEMPERATURE, MAX_TEMPERATURE
    )
    top_p = _prompt_float("top_p", DEFAULT_TOP_P, MIN_TOP_P, MAX_TOP_P)
    response_language = _prompt_response_language(default_response_language)

    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        template_enable_thinking=resolve_template_enable_thinking(),
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        response_language=response_language,
        enabled_tools=resolved_tools,
    )
    config.validate()
    print(f"\n設定: {config.summary()}\n")
    if config.enabled_tools:
        print(format_tools_help())
        print()
    return config

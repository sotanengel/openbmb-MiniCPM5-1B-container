"""Session generation settings configured at login time."""

from __future__ import annotations

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
        "",
        "  enable_thinking — 思考モード（apply_chat_template の reasoning）",
        "    デフォルト: no  入力: yes / no",
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
        f"  （参考）会話上限: メッセージ数 {MAX_MESSAGES}、1メッセージ {MAX_MESSAGE_CHARS} 文字",
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


@dataclass
class GenerationConfig:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    enable_thinking: bool = False
    do_sample: bool = DEFAULT_DO_SAMPLE
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P

    @classmethod
    def defaults(cls) -> GenerationConfig:
        return cls()

    def validate(self) -> None:
        if self.max_new_tokens < 1 or self.max_new_tokens > MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}")
        if self.temperature < MIN_TEMPERATURE or self.temperature > MAX_TEMPERATURE:
            raise ValueError(f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
        if self.top_p < MIN_TOP_P or self.top_p > MAX_TOP_P:
            raise ValueError(f"top_p must be between {MIN_TOP_P} and {MAX_TOP_P}")

    def to_chat_request(self, messages: list[dict[str, str]]) -> ChatRequest:
        self.validate()
        return ChatRequest(
            messages=messages,
            max_new_tokens=self.max_new_tokens,
            enable_thinking=self.enable_thinking,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def summary(self) -> str:
        return (
            f"max_new_tokens={self.max_new_tokens}, "
            f"enable_thinking={str(self.enable_thinking).lower()}, "
            f"do_sample={str(self.do_sample).lower()}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}"
        )


def prompt_generation_config() -> GenerationConfig:
    print("\nパスワード認証に成功しました。")
    print_generation_settings_help()
    config = GenerationConfig(
        max_new_tokens=_prompt_int("max_new_tokens", DEFAULT_MAX_NEW_TOKENS, 1, MAX_NEW_TOKENS),
        enable_thinking=_prompt_bool("enable_thinking", False),
        do_sample=_prompt_bool("do_sample", DEFAULT_DO_SAMPLE),
        temperature=_prompt_float(
            "temperature", DEFAULT_TEMPERATURE, MIN_TEMPERATURE, MAX_TEMPERATURE
        ),
        top_p=_prompt_float("top_p", DEFAULT_TOP_P, MIN_TOP_P, MAX_TOP_P),
    )
    config.validate()
    print(f"\n設定: {config.summary()}\n")
    return config

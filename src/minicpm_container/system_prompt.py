"""System prompt generation for response language control."""

from __future__ import annotations

RESPONSE_LANGUAGE_AUTO = "auto"

LANGUAGE_NAMES: dict[str, str] = {
    "ja": "Japanese",
    "en": "English",
    "zh": "Chinese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}

SUPPORTED_RESPONSE_LANGUAGES = frozenset({RESPONSE_LANGUAGE_AUTO, *LANGUAGE_NAMES.keys()})

AUTO_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "You must always respond in the same language as the user's most recent message. "
    "If the user writes in Japanese, respond in Japanese; "
    "if in English, respond in English; and match any other language the user uses."
)

TOOLS_USAGE_APPEND = (
    "When you must call a tool, emit the call as XML only, using this shape: "
    '<function name="TOOL_NAME"><param name="PARAM">value</param></function>. '
    "Do not describe the call in prose instead of XML. "
    "After tool results appear in the conversation, answer the user concisely."
)


def validate_response_language(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_RESPONSE_LANGUAGES:
        supported = ", ".join(sorted(SUPPORTED_RESPONSE_LANGUAGES))
        raise ValueError(f"response_language must be one of: {supported}")
    return normalized


def build_system_message(
    response_language: str,
    *,
    enabled_tools: tuple[str, ...] = (),
) -> dict[str, str] | None:
    language = validate_response_language(response_language)
    if language == RESPONSE_LANGUAGE_AUTO:
        content = AUTO_SYSTEM_PROMPT
    else:
        language_name = LANGUAGE_NAMES[language]
        content = f"You are a helpful assistant. You must always respond in {language_name}."
    if enabled_tools:
        content = f"{content} {TOOLS_USAGE_APPEND}"
    return {"role": "system", "content": content}

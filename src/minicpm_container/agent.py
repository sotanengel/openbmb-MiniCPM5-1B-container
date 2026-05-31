"""Agent loop: model generation with whitelisted tool execution."""

from __future__ import annotations

from minicpm_container.generation_config import GenerationConfig
from minicpm_container.protocol import ConversationState, ModelClient, ProtocolError
from minicpm_container.tools.executor import execute_tool
from minicpm_container.tools.limits import (
    MAX_FALLBACK_TOOL_RESULT_CHARS,
    MAX_TOOL_CALLS_PER_RESPONSE,
    MAX_TOOL_ROUNDS,
)
from minicpm_container.tools.parser import parse_tool_calls
from minicpm_container.tools.registry import get_tool_schemas


def run_agent_turn(
    client: ModelClient,
    config: GenerationConfig,
    state: ConversationState,
    user_text: str,
) -> str:
    """Process one user message, running tool rounds until completion."""
    state.add_user_message(user_text)
    user_turn_index = state.user_turn_index()
    tool_schemas = get_tool_schemas(config.enabled_tools)
    if not tool_schemas:
        return _single_shot(client, config, state)

    final_text = ""
    nudge_used = False
    for _round in range(MAX_TOOL_ROUNDS):
        request = config.to_chat_request(state.messages_snapshot())
        response = client.send(request)
        if response.error:
            state.rollback_last_message()
            raise ProtocolError(response.error)

        raw_content = response.content
        parsed = parse_tool_calls(raw_content, tool_schemas)
        if not parsed.calls:
            if not nudge_used and _should_nudge_for_tool_call(
                raw_content,
                tool_schemas,
                state,
                config.enabled_tools,
                user_turn_index,
            ):
                nudge_used = True
                state.add_assistant_message(raw_content)
                state.add_user_message(_tool_call_nudge_message(config.enabled_tools))
                continue
            final_text = parsed.normal_text or raw_content.strip()
            if not final_text:
                final_text = state.latest_tool_result_fallback(MAX_FALLBACK_TOOL_RESULT_CHARS)
            state.add_assistant_message(raw_content)
            return final_text

        state.add_assistant_message(raw_content)
        calls = parsed.calls[:MAX_TOOL_CALLS_PER_RESPONSE]
        for call in calls:
            result = execute_tool(call.name, call.arguments)
            state.add_tool_message(_format_tool_result(call.name, result))

        final_text = parsed.normal_text
        if not final_text.strip():
            state.add_user_message(
                "Using the tool results above, answer the user concisely in their language. "
                "Do not call any more tools."
            )

    fallback = state.latest_tool_result_fallback(MAX_FALLBACK_TOOL_RESULT_CHARS)
    return final_text or fallback or "(tool round limit reached)"


def _should_nudge_for_tool_call(
    content: str,
    tool_schemas: list[dict],
    state: ConversationState,
    enabled_tools: tuple[str, ...],
    user_turn_index: int,
) -> bool:
    """Retry once when tools are enabled but the model answered without calling them."""
    if parse_tool_calls(content, tool_schemas).calls:
        return False
    if state.has_tool_results_since(user_turn_index):
        return False
    if "<function" in content or "<tool_call>" in content or '"name"' in content:
        return False
    return _prose_signals_tool_intent(content, enabled_tools)


_TOOL_INTENT_MARKERS = (
    "will use",
    "i'll use",
    "going to use",
    "let me use",
    "need to use",
    "using the ",
)


def _prose_signals_tool_intent(content: str, enabled_tools: tuple[str, ...]) -> bool:
    """True when the model clearly promised a tool call but did not emit one."""
    lowered = content.lower()
    if not any(marker in lowered for marker in _TOOL_INTENT_MARKERS):
        return False
    if "tool" in lowered:
        return True
    for tool_id in enabled_tools:
        if tool_id in lowered or tool_id.replace("_", " ") in lowered:
            return True
    return False


def _tool_call_nudge_message(enabled_tools: tuple[str, ...]) -> str:
    if "web_search" in enabled_tools:
        return (
            "You indicated you would search. If needed, call web_search, e.g. "
            '{"name":"web_search","arguments":{"query":"..."}}.'
        )
    if len(enabled_tools) == 1:
        tool_id = enabled_tools[0]
        return (
            f"You indicated you would use {tool_id}. If needed, emit XML "
            f'(<function name="{tool_id}"><param name="...">...</param></function>) '
            "or JSON."
        )
    return (
        "You indicated you would use a tool. If needed, emit XML "
        '(<function name="..."><param name="...">...</param></function>) '
        'or JSON ({"name":"...","arguments":{...}}).'
    )


def _single_shot(
    client: ModelClient,
    config: GenerationConfig,
    state: ConversationState,
) -> str:
    response = client.send(config.to_chat_request(state.messages_snapshot()))
    if response.error:
        state.rollback_last_message()
        raise ProtocolError(response.error)
    content = response.content.strip()
    state.add_assistant_message(content)
    return content


def _format_tool_result(tool_name: str, result: str) -> str:
    # Chat template wraps tool role content in <tool_response> blocks.
    return f"{tool_name}: {result}"

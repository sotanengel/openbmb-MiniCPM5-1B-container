"""Agent loop: model generation with whitelisted tool execution."""

from __future__ import annotations

from minicpm_container.generation_config import GenerationConfig
from minicpm_container.protocol import ConversationState, ModelClient, ProtocolError
from minicpm_container.tools.executor import execute_tool
from minicpm_container.tools.limits import (
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
    tool_schemas = get_tool_schemas(config.enabled_tools)
    if not tool_schemas:
        return _single_shot(client, config, state)

    final_text = ""
    nudge_used = False
    for _round in range(MAX_TOOL_ROUNDS):
        request = config.to_chat_request(list(state.messages))
        response = client.send(request)
        if response.error:
            state.messages.pop()
            raise ProtocolError(response.error)

        raw_content = response.content
        parsed = parse_tool_calls(raw_content, tool_schemas)
        if not parsed.calls:
            if not nudge_used and _should_nudge_for_tool_xml(raw_content):
                nudge_used = True
                state.add_assistant_message(raw_content)
                state.add_user_message(
                    "Emit the required tool call as XML only "
                    '(<function name="..."><param name="...">...</param></function>). '
                    "Do not explain; output the XML."
                )
                continue
            final_text = parsed.normal_text or raw_content.strip()
            state.add_assistant_message(raw_content)
            return final_text

        state.add_assistant_message(raw_content)
        calls = parsed.calls[:MAX_TOOL_CALLS_PER_RESPONSE]
        for call in calls:
            result = execute_tool(call.name, call.arguments)
            state.add_tool_message(_format_tool_result(call.name, result))

        final_text = parsed.normal_text

    return final_text or "(tool round limit reached)"


def _should_nudge_for_tool_xml(content: str) -> bool:
    """Detect prose-only tool intent so we can retry once with an explicit XML request."""
    lowered = content.lower()
    if "<function" in content or "<tool_call>" in content:
        return False
    hints = ("tool", "calculate", "function call", "xml")
    return any(hint in lowered for hint in hints)


def _single_shot(
    client: ModelClient,
    config: GenerationConfig,
    state: ConversationState,
) -> str:
    response = client.send(config.to_chat_request(list(state.messages)))
    if response.error:
        state.messages.pop()
        raise ProtocolError(response.error)
    content = response.content.strip()
    state.add_assistant_message(content)
    return content


def _format_tool_result(tool_name: str, result: str) -> str:
    # Chat template wraps tool role content in <tool_response> blocks.
    return f"{tool_name}: {result}"

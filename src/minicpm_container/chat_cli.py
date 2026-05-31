"""Interactive CLI chat client."""

from __future__ import annotations

import sys

from minicpm_container.agent import run_agent_turn
from minicpm_container.generation_config import GenerationConfig
from minicpm_container.protocol import (
    ConversationState,
    ModelClient,
    ProtocolError,
    sanitize_message_content,
)
from minicpm_container.tools.registry import format_tools_help

HELP_TEXT = """Commands:
  /exit   Exit the chat session
  /clear  Clear conversation history
  /help   Show this help message

Response language is configured at login (response_language).
Use auto to follow the language of each user message.
Tools are configured at login (--tools or enabled_tools prompt).
"""


def run_chat_loop(
    client: ModelClient | None = None,
    config: GenerationConfig | None = None,
) -> None:
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_client = client or ModelClient()
    session_config = config or GenerationConfig.defaults()
    state = ConversationState()

    print("MiniCPM5-1B secure chat. Type /help for commands.")
    print(f"Generation settings: {session_config.summary()}")
    if session_config.enabled_tools:
        print(f"Enabled tools: {', '.join(session_config.enabled_tools)}")
    else:
        print("Enabled tools: none")
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not user_input:
            continue

        user_input = sanitize_message_content(user_input)

        if user_input.startswith("/"):
            command = user_input.lower()
            if command == "/exit":
                print("Bye.")
                return
            if command == "/clear":
                state.clear()
                print("Conversation cleared.")
                continue
            if command == "/help":
                print(HELP_TEXT.rstrip())
                if session_config.enabled_tools:
                    print()
                    print(format_tools_help())
                continue
            print(f"Unknown command: {user_input}. Type /help for available commands.")
            continue

        try:
            assistant_text = run_agent_turn(
                model_client,
                session_config,
                state,
                user_input,
            )
        except (ProtocolError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue

        print(f"Assistant> {assistant_text}")


def main() -> None:
    try:
        run_chat_loop()
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

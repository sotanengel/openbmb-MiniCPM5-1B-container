"""Interactive CLI chat client."""

from __future__ import annotations

import sys

from minicpm_container.generation_config import GenerationConfig
from minicpm_container.protocol import (
    ConversationState,
    ModelClient,
    ProtocolError,
    sanitize_message_content,
)

HELP_TEXT = """Commands:
  /exit   Exit the chat session
  /clear  Clear conversation history
  /help   Show this help message

Response language is configured at login (response_language).
Use auto to follow the language of each user message.
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
                continue
            print(f"Unknown command: {user_input}. Type /help for available commands.")
            continue

        state.add_user_message(user_input)
        request = session_config.to_chat_request(list(state.messages))

        try:
            response = model_client.send(request)
        except (ProtocolError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            state.messages.pop()
            continue

        if response.error:
            print(f"Model error: {response.error}", file=sys.stderr)
            state.messages.pop()
            continue

        print(f"Assistant> {response.content}")
        state.add_assistant_message(response.content)


def main() -> None:
    try:
        run_chat_loop()
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

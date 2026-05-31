"""Interactive CLI chat client."""

from __future__ import annotations

import sys

from minicpm_container.protocol import (
    ChatRequest,
    ConversationState,
    ModelClient,
    ProtocolError,
)

HELP_TEXT = """Commands:
  /exit   Exit the chat session
  /clear  Clear conversation history
  /help   Show this help message
"""


def run_chat_loop(client: ModelClient | None = None) -> None:
    model_client = client or ModelClient()
    state = ConversationState()

    print("MiniCPM5-1B secure chat. Type /help for commands.")
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not user_input:
            continue

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
        request = ChatRequest(messages=list(state.messages))

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

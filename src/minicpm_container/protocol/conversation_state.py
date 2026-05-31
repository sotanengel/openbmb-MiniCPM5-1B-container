"""In-memory conversation history for the chat CLI."""

from __future__ import annotations

from dataclasses import dataclass, field

from minicpm_container.protocol.messages import sanitize_message_content


@dataclass
class ConversationState:
    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": sanitize_message_content(content)})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": sanitize_message_content(content)})

    def add_tool_message(self, content: str) -> None:
        self.messages.append({"role": "tool", "content": sanitize_message_content(content)})

    def clear(self) -> None:
        self.messages.clear()

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

    def messages_snapshot(self) -> list[dict[str, str]]:
        """Return a shallow copy of messages for read-only request building."""
        return list(self.messages)

    def rollback_last_message(self) -> None:
        """Remove the most recently appended message."""
        if self.messages:
            self.messages.pop()

    def user_turn_index(self) -> int:
        """Index of the user message that started the current turn."""
        return len(self.messages) - 1

    def has_tool_results_since(self, index: int) -> bool:
        """True if a tool message exists after the message at index."""
        for message in self.messages[index + 1 :]:
            if message.get("role") == "tool":
                return True
        return False

    def latest_tool_result_fallback(self, max_chars: int) -> str:
        """Use recent tool output when the model returns an empty final message."""
        for message in reversed(self.messages):
            if message.get("role") != "tool":
                continue
            content = message.get("content", "").strip()
            if content and not content.lower().startswith("error:"):
                if len(content) > max_chars:
                    return content[:max_chars] + "…"
                return content
        return ""

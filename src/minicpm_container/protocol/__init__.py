"""JSON protocol for chat CLI <-> model server communication."""

from minicpm_container.protocol.conversation_state import ConversationState
from minicpm_container.protocol.messages import (
    ALLOWED_MESSAGE_ROLES,
    DEFAULT_DO_SAMPLE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SOCKET_PATH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MAX_MESSAGES,
    MAX_NEW_TOKENS,
    MAX_TEMPERATURE,
    MAX_TOP_P,
    MIN_TEMPERATURE,
    MIN_TOP_P,
    ChatRequest,
    ChatResponse,
    ProtocolError,
    sanitize_message_content,
)
from minicpm_container.protocol.model_client import ModelClient
from minicpm_container.protocol.socket_io import recv_all, send_json
from minicpm_container.tools.limits import MAX_MESSAGE_CHARS

__all__ = [
    "ALLOWED_MESSAGE_ROLES",
    "ChatRequest",
    "ChatResponse",
    "ConversationState",
    "DEFAULT_DO_SAMPLE",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_SOCKET_PATH",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "MAX_MESSAGES",
    "MAX_MESSAGE_CHARS",
    "MAX_NEW_TOKENS",
    "MAX_TEMPERATURE",
    "MAX_TOP_P",
    "MIN_TEMPERATURE",
    "MIN_TOP_P",
    "ModelClient",
    "ProtocolError",
    "recv_all",
    "sanitize_message_content",
    "send_json",
]

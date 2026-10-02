"""Process-local conversation history for the portfolio demo."""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from app.models.schemas import ChatMessage

# ponytail: process-local sessions; add persistence/expiry for a deployed service.
_SESSION_STORE: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def get_formatted_history(session_id: str) -> list[ChatMessage]:
    return [
        ChatMessage(role="human" if isinstance(msg, HumanMessage) else "ai", content=msg.content)
        for msg in get_session_history(session_id).messages
    ]


def clear_session_history(session_id: str) -> bool:
    return _SESSION_STORE.pop(session_id, None) is not None

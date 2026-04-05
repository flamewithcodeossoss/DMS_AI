from collections import defaultdict
from app.core.config import get_settings

settings = get_settings()

# In-memory conversation history per session
_history: dict[str, list[dict[str, str]]] = defaultdict(list)


def add_message(session_id: str, role: str, content: str):
    """Append a message and trim to MAX_HISTORY."""
    _history[session_id].append({"role": role, "content": content})
    if len(_history[session_id]) > settings.MAX_HISTORY:
        _history[session_id] = _history[session_id][-settings.MAX_HISTORY:]


def get_history(session_id: str) -> str:
    """Return formatted conversation history."""
    messages = _history.get(session_id, [])
    if not messages:
        return ""
    lines = []
    for msg in messages:
        lines.append(f"{msg['role'].upper()}: {msg['content']}")
    return "\n".join(lines)


def clear_history(session_id: str):
    """Clear history for a session."""
    _history.pop(session_id, None)

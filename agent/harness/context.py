"""Conversation context manager (stub).

Real implementation will handle:
  - rolling-window message pruning
  - per-message char/token caps
  - prompt-cache-aware prefix locking (Anthropic-style cache_control)

The MVP keeps a flat list and truncates oversized messages. Agents call
``ContextManager.window()`` to get the active message list right before an LLM
call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ContextManager:
    max_messages_keep: int = 20
    per_message_max_chars: int = 6000
    messages: List[Dict[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        if len(content) > self.per_message_max_chars:
            content = content[: self.per_message_max_chars] + "\n…[truncated]"
        self.messages.append({"role": role, "content": content})

    def window(self) -> List[Dict[str, str]]:
        if len(self.messages) <= self.max_messages_keep:
            return list(self.messages)
        # Keep system messages plus the tail.
        system = [m for m in self.messages if m["role"] == "system"]
        tail = [m for m in self.messages if m["role"] != "system"][
            -(self.max_messages_keep - len(system)) :
        ]
        return system + tail

    def reset(self) -> None:
        self.messages.clear()

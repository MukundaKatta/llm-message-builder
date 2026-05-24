"""Fluent builder for LLM message arrays.

Build the ``messages`` list expected by Anthropic, OpenAI, and compatible
APIs without manually constructing dicts.

:class:`MessageBuilder` provides a chainable API: call :meth:`~MessageBuilder.system`,
:meth:`~MessageBuilder.user`, :meth:`~MessageBuilder.assistant`,
or :meth:`~MessageBuilder.tool` to append turns, then :meth:`~MessageBuilder.build`
to get the final list of dicts.  Each dict has ``"role"`` and ``"content"`` keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class Role(str, Enum):
    """LLM conversation roles.

    Values match the wire-format strings used by Anthropic and OpenAI.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A single conversation turn.

    Attributes:
        role: Speaker role.
        content: Text content of the turn.
    """

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        """Return ``{"role": ..., "content": ...}``."""
        return {"role": self.role.value, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Reconstruct a :class:`Message` from a plain dict."""
        return cls(role=Role(data["role"]), content=str(data["content"]))

    def __repr__(self) -> str:
        preview = self.content[:40] + "..." if len(self.content) > 40 else self.content
        return f"Message(role={self.role.value!r}, content={preview!r})"


class MessageBuilder:
    """Fluent builder for an ordered list of :class:`Message` objects.

    Example::

        messages = (
            MessageBuilder()
            .system("You are a helpful assistant.")
            .user("What is 2+2?")
            .assistant("4.")
            .user("And 3+3?")
            .build()
        )
        # [{"role": "system", ...}, {"role": "user", ...}, ...]
    """

    def __init__(self) -> None:
        self._messages: list[Message] = []

    # ------------------------------------------------------------------
    # Fluent append methods
    # ------------------------------------------------------------------

    def add(self, role: Role, content: str) -> MessageBuilder:
        """Append a message with any :class:`Role`."""
        self._messages.append(Message(role=role, content=content))
        return self

    def system(self, content: str) -> MessageBuilder:
        """Append a system message."""
        return self.add(Role.SYSTEM, content)

    def user(self, content: str) -> MessageBuilder:
        """Append a user message."""
        return self.add(Role.USER, content)

    def assistant(self, content: str) -> MessageBuilder:
        """Append an assistant message."""
        return self.add(Role.ASSISTANT, content)

    def tool(self, content: str) -> MessageBuilder:
        """Append a tool-result message."""
        return self.add(Role.TOOL, content)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def count(self, role: Role | None = None) -> int:
        """Number of messages, optionally filtered by *role*."""
        if role is None:
            return len(self._messages)
        return sum(1 for m in self._messages if m.role is role)

    def last(self) -> Message | None:
        """Return the most recently added message, or ``None``."""
        return self._messages[-1] if self._messages else None

    def first(self) -> Message | None:
        """Return the first message, or ``None``."""
        return self._messages[0] if self._messages else None

    def filter_by_role(self, role: Role) -> list[Message]:
        """Return all messages with *role*, in insertion order."""
        return [m for m in self._messages if m.role is role]

    def messages(self) -> list[Message]:
        """Return a copy of the message list."""
        return list(self._messages)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def pop(self) -> Message:
        """Remove and return the last message.

        Raises:
            IndexError: If the builder has no messages.
        """
        if not self._messages:
            raise IndexError("No messages to pop.")
        return self._messages.pop()

    def clear(self) -> MessageBuilder:
        """Remove all messages and return ``self``."""
        self._messages.clear()
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> list[dict[str, str]]:
        """Return the messages as a list of ``{"role": ..., "content": ...}`` dicts."""
        return [m.to_dict() for m in self._messages]

    def build_objects(self) -> list[Message]:
        """Return a copy of the messages as :class:`Message` objects."""
        return list(self._messages)

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_list(cls, messages: list[dict[str, Any]]) -> MessageBuilder:
        """Construct a :class:`MessageBuilder` from an existing list of dicts.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.

        Returns:
            A new :class:`MessageBuilder` with those messages loaded.
        """
        builder = cls()
        for m in messages:
            builder._messages.append(Message.from_dict(m))
        return builder

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        roles = [m.role.value for m in self._messages]
        return f"MessageBuilder(messages={roles!r})"

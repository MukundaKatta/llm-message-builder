"""Fluent builder for LLM conversation message lists.

This module exposes :class:`MessageBuilder`, a small dependency-free helper for
assembling the ``messages`` list that LLM chat APIs (Anthropic, OpenAI, and
compatible providers) expect. It supports plain string content, mixed content
blocks, and ``tool_use`` / ``tool_result`` pairs, and it deep-copies data on the
way in and out so a builder never shares mutable state with its callers.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

# A single content block, e.g. ``{"type": "text", "text": "hi"}``.
Block = Dict[str, Any]
# Message content is either a plain string or a list of content blocks.
Content = Any
# A fully-formed message, e.g. ``{"role": "user", "content": "hi"}``.
Message = Dict[str, Any]


class MessageBuilderError(Exception):
    """Raised on invalid message construction."""


class MessageBuilder:
    """Build a list of LLM conversation messages with a fluent interface.

    Every mutating method returns ``self`` so calls can be chained::

        messages = (
            MessageBuilder()
            .system("You are helpful.")
            .user("Hello")
            .assistant("Hi!")
            .build()
        )

    The builder owns its internal message list. Data is deep-copied on input
    (``__init__``, :meth:`extend`, :meth:`user_blocks`, ...) and on output
    (:meth:`build`, :meth:`last`, :meth:`copy`), so mutating either the inputs
    or the results never corrupts the builder's state.
    """

    def __init__(self, messages: Optional[List[Message]] = None) -> None:
        """Create a builder, optionally seeded with existing messages.

        Args:
            messages: An optional list of pre-built message dicts to start
                from. The list is deep-copied, so later mutations of the
                passed-in list do not affect the builder.
        """
        self._messages: List[Message] = copy.deepcopy(messages) if messages else []

    def system(self, content: Content) -> "MessageBuilder":
        """Append a ``system`` message and return ``self``."""
        return self._add("system", content)

    def user(self, content: Content) -> "MessageBuilder":
        """Append a ``user`` message and return ``self``."""
        return self._add("user", content)

    def assistant(self, content: Content) -> "MessageBuilder":
        """Append an ``assistant`` message and return ``self``."""
        return self._add("assistant", content)

    def user_blocks(self, blocks: List[Block]) -> "MessageBuilder":
        """Append a ``user`` message whose content is a list of blocks.

        Args:
            blocks: A list of content-block dicts (e.g. ``text`` and ``image``
                blocks). The list is validated and deep-copied.

        Raises:
            MessageBuilderError: If ``blocks`` is not a list of dicts.
        """
        self._validate_blocks(blocks)
        return self._add("user", copy.deepcopy(blocks))

    def assistant_blocks(self, blocks: List[Block]) -> "MessageBuilder":
        """Append an ``assistant`` message whose content is a list of blocks.

        Args:
            blocks: A list of content-block dicts. The list is validated and
                deep-copied.

        Raises:
            MessageBuilderError: If ``blocks`` is not a list of dicts.
        """
        self._validate_blocks(blocks)
        return self._add("assistant", copy.deepcopy(blocks))

    def assistant_tool_use(
        self,
        tool_use_id: str,
        name: str,
        input_data: Optional[Dict[str, Any]] = None,
        *,
        text: Optional[str] = None,
    ) -> "MessageBuilder":
        """Append an ``assistant`` message that invokes a tool.

        Args:
            tool_use_id: Unique id for this tool call; the matching
                :meth:`user_tool_result` must reference the same id.
            name: The tool name to invoke.
            input_data: The tool's input arguments. Defaults to an empty dict.
            text: Optional assistant text emitted before the ``tool_use`` block
                (for example, "Let me search for that.").

        Raises:
            MessageBuilderError: If ``tool_use_id`` or ``name`` is empty.
        """
        if not tool_use_id:
            raise MessageBuilderError("tool_use_id must not be empty")
        if not name:
            raise MessageBuilderError("tool name must not be empty")
        blocks: List[Block] = []
        if text is not None:
            blocks.append({"type": "text", "text": text})
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": name,
                "input": copy.deepcopy(input_data) if input_data else {},
            }
        )
        return self._add("assistant", blocks)

    def user_tool_result(
        self,
        tool_use_id: str,
        content: Content,
        *,
        is_error: bool = False,
    ) -> "MessageBuilder":
        """Append a ``user`` message carrying a tool's result.

        Args:
            tool_use_id: The id of the ``tool_use`` block this result answers.
            content: The result payload — either a string or a list of content
                blocks.
            is_error: When ``True``, marks the result as an error so the model
                knows the tool call failed.

        Raises:
            MessageBuilderError: If ``tool_use_id`` is empty.
        """
        if not tool_use_id:
            raise MessageBuilderError("tool_use_id must not be empty")
        block: Block = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": copy.deepcopy(content),
        }
        if is_error:
            block["is_error"] = True
        return self._add("user", [block])

    def add(self, role: str, content: Content) -> "MessageBuilder":
        """Append a message with an arbitrary ``role``.

        Use this for providers or roles not covered by the dedicated helpers
        (for example, ``"function"`` or ``"tool"``).

        Args:
            role: The message role; must be non-empty.
            content: The message content.

        Raises:
            MessageBuilderError: If ``role`` is empty.
        """
        if not role:
            raise MessageBuilderError("role must not be empty")
        return self._add(role, content)

    def last(self) -> Optional[Message]:
        """Return a deep copy of the last message, or ``None`` if empty."""
        return copy.deepcopy(self._messages[-1]) if self._messages else None

    def count(self) -> int:
        """Return the number of messages accumulated so far."""
        return len(self._messages)

    def roles(self) -> List[str]:
        """Return the ordered list of roles, e.g. ``["system", "user"]``."""
        return [m["role"] for m in self._messages]

    def build(self) -> List[Message]:
        """Return a deep copy of the accumulated messages.

        The returned list is independent of the builder, so mutating it does
        not change the builder's state and vice versa.
        """
        return copy.deepcopy(self._messages)

    def copy(self) -> "MessageBuilder":
        """Return an independent deep copy of this builder.

        Useful for forking a shared prefix (e.g. a system prompt) into several
        separate conversations.
        """
        return MessageBuilder(self._messages)

    def reset(self) -> "MessageBuilder":
        """Remove all messages and return ``self``."""
        self._messages.clear()
        return self

    def extend(self, messages: List[Message]) -> "MessageBuilder":
        """Append several pre-built messages and return ``self``.

        Args:
            messages: An iterable of message dicts, each of which must contain
                both a ``role`` and a ``content`` key. The messages are
                deep-copied before being stored.

        Raises:
            MessageBuilderError: If any message lacks ``role`` or ``content``.
        """
        for m in messages:
            if "role" not in m or "content" not in m:
                raise MessageBuilderError("each message must have 'role' and 'content'")
        self._messages.extend(copy.deepcopy(messages))
        return self

    def _add(self, role: str, content: Content) -> "MessageBuilder":
        self._messages.append({"role": role, "content": content})
        return self

    @staticmethod
    def _validate_blocks(blocks: List[Block]) -> None:
        if not isinstance(blocks, list):
            raise MessageBuilderError("blocks must be a list of dicts")
        for block in blocks:
            if not isinstance(block, dict):
                raise MessageBuilderError("each block must be a dict")

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"MessageBuilder(count={len(self._messages)})"

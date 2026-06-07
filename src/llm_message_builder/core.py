"""Fluent builder for LLM conversation message lists."""

from __future__ import annotations

import copy


class MessageBuilderError(Exception):
    """Raised on invalid message construction."""


class MessageBuilder:
    """Build a list of LLM conversation messages with a fluent interface."""

    def __init__(self, messages=None):
        self._messages = copy.deepcopy(messages) if messages else []

    def system(self, content):
        return self._add("system", content)

    def user(self, content):
        return self._add("user", content)

    def assistant(self, content):
        return self._add("assistant", content)

    def user_blocks(self, blocks):
        self._validate_blocks(blocks)
        return self._add("user", copy.deepcopy(blocks))

    def assistant_blocks(self, blocks):
        self._validate_blocks(blocks)
        return self._add("assistant", copy.deepcopy(blocks))

    def assistant_tool_use(self, tool_use_id, name, input_data=None, *, text=None):
        if not tool_use_id:
            raise MessageBuilderError("tool_use_id must not be empty")
        if not name:
            raise MessageBuilderError("tool name must not be empty")
        blocks = []
        if text is not None:
            blocks.append({"type": "text", "text": text})
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": name,
                "input": input_data or {},
            }
        )
        return self._add("assistant", blocks)

    def user_tool_result(self, tool_use_id, content, *, is_error=False):
        if not tool_use_id:
            raise MessageBuilderError("tool_use_id must not be empty")
        block = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            block["is_error"] = True
        return self._add("user", [block])

    def add(self, role, content):
        if not role:
            raise MessageBuilderError("role must not be empty")
        return self._add(role, content)

    def last(self):
        return copy.deepcopy(self._messages[-1]) if self._messages else None

    def count(self):
        return len(self._messages)

    def roles(self):
        return [m["role"] for m in self._messages]

    def build(self):
        return copy.deepcopy(self._messages)

    def copy(self):
        return MessageBuilder(self._messages)

    def reset(self):
        self._messages.clear()
        return self

    def extend(self, messages):
        for m in messages:
            if "role" not in m or "content" not in m:
                raise MessageBuilderError("each message must have 'role' and 'content'")
        self._messages.extend(copy.deepcopy(messages))
        return self

    def _add(self, role, content):
        self._messages.append({"role": role, "content": content})
        return self

    @staticmethod
    def _validate_blocks(blocks):
        if not isinstance(blocks, list):
            raise MessageBuilderError("blocks must be a list of dicts")
        for block in blocks:
            if not isinstance(block, dict):
                raise MessageBuilderError("each block must be a dict")

    def __len__(self):
        return len(self._messages)

    def __repr__(self):
        return f"MessageBuilder(count={len(self._messages)})"

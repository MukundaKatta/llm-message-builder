"""Tests for llm-message-builder."""

from __future__ import annotations

import pytest

from llm_message_builder import Message, MessageBuilder, Role

# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------


def test_role_values():
    assert Role.SYSTEM.value == "system"
    assert Role.USER.value == "user"
    assert Role.ASSISTANT.value == "assistant"
    assert Role.TOOL.value == "tool"


def test_role_from_string():
    assert Role("user") is Role.USER


# ---------------------------------------------------------------------------
# Message — construction
# ---------------------------------------------------------------------------


def test_message_construction():
    m = Message(role=Role.USER, content="hello")
    assert m.role is Role.USER
    assert m.content == "hello"


def test_message_to_dict():
    m = Message(role=Role.ASSISTANT, content="hi")
    d = m.to_dict()
    assert d == {"role": "assistant", "content": "hi"}


def test_message_from_dict():
    m = Message.from_dict({"role": "system", "content": "be helpful"})
    assert m.role is Role.SYSTEM
    assert m.content == "be helpful"


def test_message_from_dict_round_trip():
    original = Message(role=Role.USER, content="a question")
    restored = Message.from_dict(original.to_dict())
    assert restored.role is original.role
    assert restored.content == original.content


def test_message_repr_short():
    m = Message(role=Role.USER, content="short")
    r = repr(m)
    assert "user" in r
    assert "short" in r


def test_message_repr_long_truncated():
    m = Message(role=Role.USER, content="x" * 50)
    assert "..." in repr(m)


# ---------------------------------------------------------------------------
# MessageBuilder — fluent appending
# ---------------------------------------------------------------------------


def test_builder_system():
    mb = MessageBuilder().system("sys prompt")
    assert mb.count() == 1
    assert mb.last().role is Role.SYSTEM


def test_builder_user():
    mb = MessageBuilder().user("hello")
    assert mb.last().role is Role.USER


def test_builder_assistant():
    mb = MessageBuilder().assistant("reply")
    assert mb.last().role is Role.ASSISTANT


def test_builder_tool():
    mb = MessageBuilder().tool("tool result")
    assert mb.last().role is Role.TOOL


def test_builder_add_generic():
    mb = MessageBuilder().add(Role.USER, "hi")
    assert mb.last().content == "hi"


def test_builder_chaining():
    mb = MessageBuilder().system("sys").user("q1").assistant("a1").user("q2")
    assert mb.count() == 4


def test_builder_methods_return_self():
    mb = MessageBuilder()
    assert mb.system("s") is mb
    assert mb.user("u") is mb
    assert mb.assistant("a") is mb
    assert mb.tool("t") is mb
    assert mb.add(Role.USER, "x") is mb


# ---------------------------------------------------------------------------
# MessageBuilder — count / first / last
# ---------------------------------------------------------------------------


def test_builder_count_empty():
    assert MessageBuilder().count() == 0


def test_builder_count_by_role():
    mb = MessageBuilder().user("u1").user("u2").assistant("a1")
    assert mb.count(Role.USER) == 2
    assert mb.count(Role.ASSISTANT) == 1
    assert mb.count(Role.SYSTEM) == 0


def test_builder_first_empty():
    assert MessageBuilder().first() is None


def test_builder_last_empty():
    assert MessageBuilder().last() is None


def test_builder_first():
    mb = MessageBuilder().system("sys").user("q")
    assert mb.first().role is Role.SYSTEM


def test_builder_last():
    mb = MessageBuilder().system("sys").user("q")
    assert mb.last().role is Role.USER


# ---------------------------------------------------------------------------
# MessageBuilder — filter_by_role
# ---------------------------------------------------------------------------


def test_filter_by_role():
    mb = MessageBuilder().user("q1").assistant("a1").user("q2")
    user_msgs = mb.filter_by_role(Role.USER)
    assert len(user_msgs) == 2
    assert all(m.role is Role.USER for m in user_msgs)


def test_filter_by_role_empty():
    mb = MessageBuilder().user("q")
    assert mb.filter_by_role(Role.SYSTEM) == []


# ---------------------------------------------------------------------------
# MessageBuilder — messages / len
# ---------------------------------------------------------------------------


def test_messages_returns_copy():
    mb = MessageBuilder().user("q")
    copy = mb.messages()
    copy.clear()
    assert mb.count() == 1


def test_len():
    mb = MessageBuilder().user("a").user("b")
    assert len(mb) == 2


# ---------------------------------------------------------------------------
# MessageBuilder — pop
# ---------------------------------------------------------------------------


def test_pop_removes_last():
    mb = MessageBuilder().user("q1").user("q2")
    removed = mb.pop()
    assert removed.content == "q2"
    assert mb.count() == 1


def test_pop_empty_raises():
    with pytest.raises(IndexError):
        MessageBuilder().pop()


# ---------------------------------------------------------------------------
# MessageBuilder — clear
# ---------------------------------------------------------------------------


def test_clear():
    mb = MessageBuilder().user("q").assistant("a")
    mb.clear()
    assert mb.count() == 0


def test_clear_returns_self():
    mb = MessageBuilder()
    assert mb.clear() is mb


# ---------------------------------------------------------------------------
# MessageBuilder — build
# ---------------------------------------------------------------------------


def test_build_returns_dicts():
    result = (
        MessageBuilder()
        .system("You are helpful.")
        .user("Hi")
        .assistant("Hello!")
        .build()
    )
    assert result == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_build_empty():
    assert MessageBuilder().build() == []


def test_build_does_not_mutate_builder():
    mb = MessageBuilder().user("q")
    mb.build()
    assert mb.count() == 1


# ---------------------------------------------------------------------------
# MessageBuilder — build_objects
# ---------------------------------------------------------------------------


def test_build_objects_returns_messages():
    mb = MessageBuilder().user("q").assistant("a")
    objs = mb.build_objects()
    assert all(isinstance(m, Message) for m in objs)
    assert len(objs) == 2


def test_build_objects_returns_copy():
    mb = MessageBuilder().user("q")
    objs = mb.build_objects()
    objs.clear()
    assert mb.count() == 1


# ---------------------------------------------------------------------------
# MessageBuilder — from_list
# ---------------------------------------------------------------------------


def test_from_list():
    raw = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
    ]
    mb = MessageBuilder.from_list(raw)
    assert mb.count() == 2
    assert mb.first().role is Role.SYSTEM
    assert mb.last().content == "q"


def test_from_list_empty():
    mb = MessageBuilder.from_list([])
    assert mb.count() == 0


def test_from_list_round_trip():
    original = MessageBuilder().system("s").user("u").assistant("a").build()
    restored = MessageBuilder.from_list(original).build()
    assert restored == original


# ---------------------------------------------------------------------------
# MessageBuilder — repr
# ---------------------------------------------------------------------------


def test_repr():
    mb = MessageBuilder().system("s").user("u")
    r = repr(mb)
    assert "MessageBuilder" in r
    assert "system" in r
    assert "user" in r

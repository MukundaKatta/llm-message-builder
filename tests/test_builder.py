"""Tests for llm-message-builder."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from llm_message_builder import MessageBuilder, MessageBuilderError


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_empty_builder():
    b = MessageBuilder()
    assert b.count() == 0
    assert b.build() == []

def test_system():
    b = MessageBuilder()
    b.system("You are helpful.")
    msgs = b.build()
    assert len(msgs) == 1
    assert msgs[0] == {"role": "system", "content": "You are helpful."}

def test_user():
    b = MessageBuilder()
    b.user("Hello!")
    assert b.build()[0]["role"] == "user"

def test_assistant():
    b = MessageBuilder()
    b.assistant("Hi there!")
    assert b.build()[0]["role"] == "assistant"

def test_chain():
    msgs = (
        MessageBuilder()
        .system("You are helpful.")
        .user("What is 2+2?")
        .assistant("4.")
        .build()
    )
    assert len(msgs) == 3
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]

def test_build_returns_list():
    b = MessageBuilder()
    result = b.build()
    assert isinstance(result, list)

def test_build_deep_copy():
    b = MessageBuilder()
    b.user("hello")
    msgs = b.build()
    msgs[0]["content"] = "modified"
    # Original should be unchanged
    assert b.build()[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# roles() / last() / count() / len()
# ---------------------------------------------------------------------------

def test_roles():
    b = MessageBuilder().system("s").user("u").assistant("a")
    assert b.roles() == ["system", "user", "assistant"]

def test_last_returns_last_message():
    b = MessageBuilder().user("first").assistant("second")
    assert b.last() == {"role": "assistant", "content": "second"}

def test_last_empty_returns_none():
    assert MessageBuilder().last() is None

def test_count():
    b = MessageBuilder().user("a").user("b").user("c")
    assert b.count() == 3

def test_len():
    b = MessageBuilder().user("x")
    assert len(b) == 1

def test_repr():
    b = MessageBuilder().user("x").user("y")
    assert "2" in repr(b)


# ---------------------------------------------------------------------------
# add() generic
# ---------------------------------------------------------------------------

def test_add_generic():
    b = MessageBuilder()
    b.add("function", "result")
    assert b.build()[0]["role"] == "function"

def test_add_empty_role_raises():
    with pytest.raises(MessageBuilderError):
        MessageBuilder().add("", "content")


# ---------------------------------------------------------------------------
# user_blocks / assistant_blocks
# ---------------------------------------------------------------------------

def test_user_blocks():
    blocks = [{"type": "text", "text": "hello"}]
    b = MessageBuilder().user_blocks(blocks)
    msg = b.build()[0]
    assert msg["role"] == "user"
    assert msg["content"] == blocks

def test_assistant_blocks():
    blocks = [{"type": "text", "text": "response"}]
    b = MessageBuilder().assistant_blocks(blocks)
    assert b.build()[0]["role"] == "assistant"

def test_blocks_deep_copy():
    blocks = [{"type": "text", "text": "original"}]
    b = MessageBuilder().user_blocks(blocks)
    blocks[0]["text"] = "modified"
    assert b.build()[0]["content"][0]["text"] == "original"

def test_blocks_not_list_raises():
    with pytest.raises(MessageBuilderError):
        MessageBuilder().user_blocks("not a list")

def test_blocks_non_dict_item_raises():
    with pytest.raises(MessageBuilderError):
        MessageBuilder().user_blocks(["not", "dicts"])


# ---------------------------------------------------------------------------
# assistant_tool_use
# ---------------------------------------------------------------------------

def test_assistant_tool_use_basic():
    b = MessageBuilder().assistant_tool_use("call_1", "web_search", {"q": "Paris"})
    msg = b.build()[0]
    assert msg["role"] == "assistant"
    content = msg["content"]
    assert isinstance(content, list)
    block = content[0]
    assert block["type"] == "tool_use"
    assert block["id"] == "call_1"
    assert block["name"] == "web_search"
    assert block["input"] == {"q": "Paris"}

def test_assistant_tool_use_empty_id_raises():
    with pytest.raises(MessageBuilderError):
        MessageBuilder().assistant_tool_use("", "search", {})

def test_assistant_tool_use_empty_name_raises():
    with pytest.raises(MessageBuilderError):
        MessageBuilder().assistant_tool_use("call_1", "", {})

def test_assistant_tool_use_with_text():
    b = MessageBuilder().assistant_tool_use(
        "call_1", "search", {"q": "test"}, text="Searching now."
    )
    content = b.build()[0]["content"]
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Searching now."
    assert content[1]["type"] == "tool_use"

def test_assistant_tool_use_no_input():
    b = MessageBuilder().assistant_tool_use("call_1", "ping")
    block = b.build()[0]["content"][0]
    assert block["input"] == {}


# ---------------------------------------------------------------------------
# user_tool_result
# ---------------------------------------------------------------------------

def test_user_tool_result_string():
    b = MessageBuilder().user_tool_result("call_1", "Paris is the capital.")
    msg = b.build()[0]
    assert msg["role"] == "user"
    block = msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    assert block["content"] == "Paris is the capital."

def test_user_tool_result_is_error():
    b = MessageBuilder().user_tool_result("call_1", "error msg", is_error=True)
    block = b.build()[0]["content"][0]
    assert block["is_error"] is True

def test_user_tool_result_not_error_by_default():
    b = MessageBuilder().user_tool_result("call_1", "ok")
    block = b.build()[0]["content"][0]
    assert "is_error" not in block

def test_user_tool_result_empty_id_raises():
    with pytest.raises(MessageBuilderError):
        MessageBuilder().user_tool_result("", "result")

def test_user_tool_result_blocks_content():
    b = MessageBuilder().user_tool_result(
        "call_1", [{"type": "text", "text": "done"}]
    )
    block = b.build()[0]["content"][0]
    assert isinstance(block["content"], list)


# ---------------------------------------------------------------------------
# Full tool-use round-trip
# ---------------------------------------------------------------------------

def test_full_tool_use_conversation():
    msgs = (
        MessageBuilder()
        .system("You are helpful.")
        .user("Search for Python.")
        .assistant_tool_use("c1", "web_search", {"q": "Python"})
        .user_tool_result("c1", "Python is a programming language.")
        .assistant("Python is a general-purpose language.")
        .build()
    )
    assert len(msgs) == 5
    assert msgs[0]["role"] == "system"
    assert msgs[2]["role"] == "assistant"
    assert msgs[3]["role"] == "user"
    # Verify tool_result references tool_use id
    result_block = msgs[3]["content"][0]
    use_block = msgs[2]["content"][0]
    assert result_block["tool_use_id"] == use_block["id"]


# ---------------------------------------------------------------------------
# copy / reset / extend
# ---------------------------------------------------------------------------

def test_copy():
    b = MessageBuilder().user("hello")
    b2 = b.copy()
    b2.user("world")
    assert b.count() == 1
    assert b2.count() == 2

def test_reset():
    b = MessageBuilder().user("a").user("b")
    b.reset()
    assert b.count() == 0

def test_reset_returns_self():
    b = MessageBuilder()
    assert b.reset() is b

def test_extend():
    pre = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    b = MessageBuilder().extend(pre).user("more")
    assert b.count() == 3

def test_extend_validates_keys():
    with pytest.raises(MessageBuilderError):
        MessageBuilder().extend([{"role": "user"}])  # missing content

def test_init_with_messages():
    seed = [{"role": "user", "content": "seed"}]
    b = MessageBuilder(seed)
    assert b.count() == 1
    assert b.build()[0]["content"] == "seed"

def test_init_does_not_share_state():
    seed = [{"role": "user", "content": "original"}]
    b = MessageBuilder(seed)
    seed[0]["content"] = "modified"
    assert b.build()[0]["content"] == "original"

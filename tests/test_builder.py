"""Standard-library ``unittest`` suite for llm-message-builder.

These tests depend only on the Python standard library, so they run without
installing any third-party packages::

    python3 -m unittest discover -s tests

They import and exercise the real :class:`MessageBuilder` implementation,
including its deep-copy isolation guarantees.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from llm_message_builder import MessageBuilder, MessageBuilderError  # noqa: E402


class TestBasicConstruction(unittest.TestCase):
    def test_empty_builder(self):
        b = MessageBuilder()
        self.assertEqual(b.count(), 0)
        self.assertEqual(b.build(), [])

    def test_system(self):
        msgs = MessageBuilder().system("You are helpful.").build()
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0], {"role": "system", "content": "You are helpful."})

    def test_user_and_assistant_roles(self):
        self.assertEqual(MessageBuilder().user("Hi").build()[0]["role"], "user")
        self.assertEqual(
            MessageBuilder().assistant("Hi").build()[0]["role"], "assistant"
        )

    def test_chain_order(self):
        msgs = (
            MessageBuilder()
            .system("You are helpful.")
            .user("What is 2+2?")
            .assistant("4.")
            .build()
        )
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])

    def test_build_returns_list(self):
        self.assertIsInstance(MessageBuilder().build(), list)

    def test_build_is_deep_copy(self):
        b = MessageBuilder().user("hello")
        msgs = b.build()
        msgs[0]["content"] = "modified"
        self.assertEqual(b.build()[0]["content"], "hello")


class TestInspection(unittest.TestCase):
    def test_roles(self):
        b = MessageBuilder().system("s").user("u").assistant("a")
        self.assertEqual(b.roles(), ["system", "user", "assistant"])

    def test_last_returns_last_message(self):
        b = MessageBuilder().user("first").assistant("second")
        self.assertEqual(b.last(), {"role": "assistant", "content": "second"})

    def test_last_empty_returns_none(self):
        self.assertIsNone(MessageBuilder().last())

    def test_last_is_deep_copy(self):
        b = MessageBuilder().user_blocks([{"type": "text", "text": "x"}])
        snapshot = b.last()
        snapshot["content"][0]["text"] = "mutated"
        self.assertEqual(b.last()["content"][0]["text"], "x")

    def test_count_and_len(self):
        b = MessageBuilder().user("a").user("b").user("c")
        self.assertEqual(b.count(), 3)
        self.assertEqual(len(b), 3)

    def test_repr_contains_count(self):
        self.assertIn("2", repr(MessageBuilder().user("x").user("y")))


class TestGenericAdd(unittest.TestCase):
    def test_add_generic_role(self):
        b = MessageBuilder().add("function", "result")
        self.assertEqual(b.build()[0]["role"], "function")

    def test_add_empty_role_raises(self):
        with self.assertRaises(MessageBuilderError):
            MessageBuilder().add("", "content")


class TestBlocks(unittest.TestCase):
    def test_user_blocks(self):
        blocks = [{"type": "text", "text": "hello"}]
        msg = MessageBuilder().user_blocks(blocks).build()[0]
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"], blocks)

    def test_assistant_blocks_role(self):
        blocks = [{"type": "text", "text": "response"}]
        self.assertEqual(
            MessageBuilder().assistant_blocks(blocks).build()[0]["role"], "assistant"
        )

    def test_blocks_deep_copy_on_input(self):
        blocks = [{"type": "text", "text": "original"}]
        b = MessageBuilder().user_blocks(blocks)
        blocks[0]["text"] = "modified"
        self.assertEqual(b.build()[0]["content"][0]["text"], "original")

    def test_blocks_not_list_raises(self):
        with self.assertRaises(MessageBuilderError):
            MessageBuilder().user_blocks("not a list")

    def test_blocks_non_dict_item_raises(self):
        with self.assertRaises(MessageBuilderError):
            MessageBuilder().user_blocks(["not", "dicts"])


class TestAssistantToolUse(unittest.TestCase):
    def test_basic(self):
        block = (
            MessageBuilder()
            .assistant_tool_use("call_1", "web_search", {"q": "Paris"})
            .build()[0]["content"][0]
        )
        self.assertEqual(block["type"], "tool_use")
        self.assertEqual(block["id"], "call_1")
        self.assertEqual(block["name"], "web_search")
        self.assertEqual(block["input"], {"q": "Paris"})

    def test_empty_id_raises(self):
        with self.assertRaises(MessageBuilderError):
            MessageBuilder().assistant_tool_use("", "search", {})

    def test_empty_name_raises(self):
        with self.assertRaises(MessageBuilderError):
            MessageBuilder().assistant_tool_use("call_1", "", {})

    def test_with_text(self):
        content = (
            MessageBuilder()
            .assistant_tool_use("call_1", "search", {"q": "t"}, text="Searching now.")
            .build()[0]["content"]
        )
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], "Searching now.")
        self.assertEqual(content[1]["type"], "tool_use")

    def test_no_input_defaults_to_empty_dict(self):
        block = MessageBuilder().assistant_tool_use("call_1", "ping").build()[0][
            "content"
        ][0]
        self.assertEqual(block["input"], {})

    def test_input_is_deep_copied(self):
        # Mutating the passed-in input must not leak into the builder.
        payload = {"q": "Paris"}
        b = MessageBuilder().assistant_tool_use("c1", "search", payload)
        payload["q"] = "London"
        self.assertEqual(b.build()[0]["content"][0]["input"], {"q": "Paris"})


class TestUserToolResult(unittest.TestCase):
    def test_string_content(self):
        block = (
            MessageBuilder()
            .user_tool_result("call_1", "Paris is the capital.")
            .build()[0]["content"][0]
        )
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "call_1")
        self.assertEqual(block["content"], "Paris is the capital.")

    def test_is_error_flag(self):
        block = (
            MessageBuilder()
            .user_tool_result("call_1", "boom", is_error=True)
            .build()[0]["content"][0]
        )
        self.assertIs(block["is_error"], True)

    def test_not_error_by_default(self):
        block = (
            MessageBuilder().user_tool_result("call_1", "ok").build()[0]["content"][0]
        )
        self.assertNotIn("is_error", block)

    def test_empty_id_raises(self):
        with self.assertRaises(MessageBuilderError):
            MessageBuilder().user_tool_result("", "result")

    def test_block_content(self):
        block = (
            MessageBuilder()
            .user_tool_result("call_1", [{"type": "text", "text": "done"}])
            .build()[0]["content"][0]
        )
        self.assertIsInstance(block["content"], list)

    def test_list_content_is_deep_copied(self):
        content = [{"type": "text", "text": "done"}]
        b = MessageBuilder().user_tool_result("c1", content)
        content[0]["text"] = "changed"
        self.assertEqual(b.build()[0]["content"][0]["content"][0]["text"], "done")


class TestFullConversation(unittest.TestCase):
    def test_tool_use_round_trip(self):
        msgs = (
            MessageBuilder()
            .system("You are helpful.")
            .user("Search for Python.")
            .assistant_tool_use("c1", "web_search", {"q": "Python"})
            .user_tool_result("c1", "Python is a programming language.")
            .assistant("Python is a general-purpose language.")
            .build()
        )
        self.assertEqual(len(msgs), 5)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertEqual(msgs[3]["role"], "user")
        result_block = msgs[3]["content"][0]
        use_block = msgs[2]["content"][0]
        self.assertEqual(result_block["tool_use_id"], use_block["id"])


class TestCopyResetExtend(unittest.TestCase):
    def test_copy_is_independent(self):
        b = MessageBuilder().user("hello")
        b2 = b.copy()
        b2.user("world")
        self.assertEqual(b.count(), 1)
        self.assertEqual(b2.count(), 2)

    def test_copy_is_deeply_independent(self):
        base = MessageBuilder().user_blocks([{"type": "text", "text": "shared"}])
        fork = base.copy()
        fork.build()[0]["content"][0]["text"] = "noop"  # build() is a copy
        fork._messages[0]["content"][0]["text"] = "changed"
        self.assertEqual(base.build()[0]["content"][0]["text"], "shared")

    def test_reset(self):
        b = MessageBuilder().user("a").user("b")
        b.reset()
        self.assertEqual(b.count(), 0)

    def test_reset_returns_self(self):
        b = MessageBuilder()
        self.assertIs(b.reset(), b)

    def test_extend(self):
        pre = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        b = MessageBuilder().extend(pre).user("more")
        self.assertEqual(b.count(), 3)

    def test_extend_validates_keys(self):
        with self.assertRaises(MessageBuilderError):
            MessageBuilder().extend([{"role": "user"}])

    def test_extend_does_not_share_state(self):
        pre = [{"role": "user", "content": "hello"}]
        b = MessageBuilder().extend(pre)
        pre[0]["content"] = "mutated"
        self.assertEqual(b.build()[0]["content"], "hello")

    def test_init_with_messages(self):
        b = MessageBuilder([{"role": "user", "content": "seed"}])
        self.assertEqual(b.count(), 1)
        self.assertEqual(b.build()[0]["content"], "seed")

    def test_init_does_not_share_state(self):
        seed = [{"role": "user", "content": "original"}]
        b = MessageBuilder(seed)
        seed[0]["content"] = "modified"
        self.assertEqual(b.build()[0]["content"], "original")


if __name__ == "__main__":
    unittest.main()

# llm-message-builder

Fluent builder for LLM message arrays (system, user, assistant, tool).

Build the `messages` list expected by Anthropic, OpenAI, and compatible APIs without manually constructing dicts.

## Install

```bash
pip install llm-message-builder
```

## Quick start

```python
from llm_message_builder import MessageBuilder

messages = (
    MessageBuilder()
    .system("You are a helpful assistant.")
    .user("What is 2+2?")
    .assistant("4.")
    .user("And 3×3?")
    .build()
)
# [
#   {"role": "system",    "content": "You are a helpful assistant."},
#   {"role": "user",      "content": "What is 2+2?"},
#   {"role": "assistant", "content": "4."},
#   {"role": "user",      "content": "And 3×3?"},
# ]
```

## API

### `MessageBuilder`

| Method | Description |
|--------|-------------|
| `system(content)` | Append a system message |
| `user(content)` | Append a user message |
| `assistant(content)` | Append an assistant message |
| `tool(content)` | Append a tool-result message |
| `add(role, content)` | Append with any `Role` |
| `build()` | Return `list[dict[str, str]]` |
| `build_objects()` | Return `list[Message]` |
| `count(role=None)` | Message count, optionally by role |
| `first()` / `last()` | First / last message, or `None` |
| `filter_by_role(role)` | Messages with that role |
| `messages()` | Copy of message list |
| `pop()` | Remove and return last message |
| `clear()` | Remove all messages; returns `self` |
| `MessageBuilder.from_list(raw)` | Load from existing list of dicts |

### `Role`

`SYSTEM` · `USER` · `ASSISTANT` · `TOOL`

### `Message`

```python
m = Message(role=Role.USER, content="hello")
m.to_dict()          # {"role": "user", "content": "hello"}
Message.from_dict(d) # reconstruct
```

## License

MIT

# llm-message-builder

[![CI](https://github.com/MukundaKatta/llm-message-builder/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/llm-message-builder/actions/workflows/ci.yml)

Fluent builder for LLM conversation message lists — Anthropic content blocks, tool_use/tool_result pairs.

Zero runtime dependencies. Python 3.10+. Fully type-hinted (ships `py.typed`). MIT.

## Why

Hand-assembling the `messages` list that chat APIs expect is fiddly: you end up
juggling role/content dicts, nested content blocks, and matching
`tool_use`/`tool_result` ids by hand. `MessageBuilder` gives you a small,
chainable API for that, and it deep-copies data in and out so a builder never
shares mutable state with your code (see [Isolation guarantees](#isolation-guarantees)).

## Install

```bash
pip install llm-message-builder
```

## Usage

```python
from llm_message_builder import MessageBuilder

messages = (
    MessageBuilder()
    .system("You are a helpful assistant.")
    .user("What is the capital of France?")
    .assistant("Paris.")
    .build()
)
# [
#   {"role": "system", "content": "You are a helpful assistant."},
#   {"role": "user", "content": "What is the capital of France?"},
#   {"role": "assistant", "content": "Paris."},
# ]
```

## Tool use (Anthropic)

```python
messages = (
    MessageBuilder()
    .system("You are helpful.")
    .user("Search for Python.")
    .assistant_tool_use("call_1", "web_search", {"q": "Python"})
    .user_tool_result("call_1", "Python is a programming language.")
    .assistant("Python is a general-purpose language.")
    .build()
)
```

## Content blocks

```python
# User message with mixed blocks
builder.user_blocks([
    {"type": "text", "text": "Look at this:"},
    {"type": "image", "source": {"type": "base64", ...}},
])

# Assistant with text before tool_use
builder.assistant_tool_use(
    "call_1", "search", {"q": "query"},
    text="Let me search for that."
)

# Tool result as error
builder.user_tool_result("call_1", "timeout", is_error=True)
```

## Copy / extend / reset

```python
base = MessageBuilder().system("You are helpful.")

# Fork for different conversations
conv_a = base.copy().user("Question A").assistant("Answer A")
conv_b = base.copy().user("Question B").assistant("Answer B")

# Extend with pre-built messages
builder.extend([{"role": "user", "content": "appended"}])

# Reset
builder.reset()
```

## Inspect

```python
builder.count()   # number of messages
builder.roles()   # ["system", "user", "assistant", ...]
builder.last()    # last message dict (deep copy)
len(builder)      # same as count()
```

## API reference

All mutating methods return the builder, so calls can be chained.

| Method | Description |
| --- | --- |
| `MessageBuilder(messages=None)` | Create a builder, optionally seeded with a list of pre-built messages (deep-copied). |
| `.system(content)` | Append a `system` message. |
| `.user(content)` | Append a `user` message. |
| `.assistant(content)` | Append an `assistant` message. |
| `.user_blocks(blocks)` | Append a `user` message whose content is a list of content-block dicts. |
| `.assistant_blocks(blocks)` | Append an `assistant` message whose content is a list of content-block dicts. |
| `.assistant_tool_use(tool_use_id, name, input_data=None, *, text=None)` | Append an `assistant` message that invokes a tool, with an optional leading text block. |
| `.user_tool_result(tool_use_id, content, *, is_error=False)` | Append a `user` message carrying a tool's result, optionally marked as an error. |
| `.add(role, content)` | Append a message with an arbitrary (non-empty) role. |
| `.extend(messages)` | Append several pre-built messages; each must have `role` and `content`. |
| `.copy()` | Return an independent deep copy of the builder. |
| `.reset()` | Remove all messages. |
| `.build()` | Return a deep copy of the accumulated message list. |
| `.last()` | Return a deep copy of the last message, or `None`. |
| `.count()` / `len(builder)` | Number of messages. |
| `.roles()` | Ordered list of message roles. |

Invalid construction raises `MessageBuilderError` (a subclass of `Exception`),
for example an empty `role`, an empty `tool_use_id`, content blocks that are not
a list of dicts, or messages missing `role`/`content` in `.extend()`.

## Isolation guarantees

The builder owns its internal list and never shares mutable references with
caller code:

- **On input** — `MessageBuilder(messages)`, `.extend()`, `.user_blocks()`,
  `.assistant_blocks()`, `.assistant_tool_use(input_data=...)` and
  `.user_tool_result(content=...)` all deep-copy their arguments, so mutating
  the original objects afterward does not change the builder.
- **On output** — `.build()`, `.last()` and `.copy()` all return deep copies,
  so mutating the result does not change the builder.

## Development

The test suite uses only the Python standard library (`unittest`), so no
third-party packages are required to run it:

```bash
python -m unittest discover -s tests -v
```

Linting/formatting is done with [ruff](https://docs.astral.sh/ruff/):

```bash
pip install ruff
ruff check src/ tests/
ruff format --check src/ tests/
```

## License

MIT

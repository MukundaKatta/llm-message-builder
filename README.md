# llm-message-builder

Fluent builder for LLM conversation message lists — Anthropic content blocks, tool_use/tool_result pairs.

Zero dependencies. Python 3.10+. MIT.

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

## License

MIT

"""llm-message-builder: fluent builder for LLM conversation message lists."""

from .core import (
    MessageBuilder,
    MessageBuilderError,
)

__all__ = [
    "MessageBuilder",
    "MessageBuilderError",
]

"""Unified types for multi-provider support.

These types decouple the sampling loop from any specific LLM provider SDK,
allowing Anthropic, OpenAI, and Gemini to be used interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProviderName = Literal["anthropic", "openai", "gemini"]


# ---------------------------------------------------------------------------
# Tool definition (provider-agnostic)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDefinition:
    """Provider-agnostic description of a tool that can be offered to the LLM."""

    name: str
    type: str  # e.g. "computer_20250124", "bash_20250124", "custom"
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    # Extra fields consumed only by specific providers (e.g. Anthropic options)
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Unified content blocks
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnifiedTextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True)
class UnifiedThinkingBlock:
    thinking: str
    signature: str | None = None
    type: Literal["thinking"] = "thinking"


@dataclass(frozen=True)
class UnifiedToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True)
class UnifiedToolResultBlock:
    tool_use_id: str
    content: list[dict[str, Any]]
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True)
class UnifiedImageBlock:
    base64_data: str
    media_type: str = "image/png"
    type: Literal["image"] = "image"


# Union of all content block types
UnifiedContentBlock = (
    UnifiedTextBlock
    | UnifiedThinkingBlock
    | UnifiedToolUseBlock
    | UnifiedToolResultBlock
    | UnifiedImageBlock
)


# ---------------------------------------------------------------------------
# Unified response
# ---------------------------------------------------------------------------

@dataclass
class UnifiedResponse:
    """Provider-agnostic representation of an LLM response."""

    content: list[UnifiedContentBlock]
    stop_reason: str | None = None
    # Raw provider-specific objects for logging / debugging
    raw_request: Any = None
    raw_response: Any = None

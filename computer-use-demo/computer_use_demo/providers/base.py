"""Abstract base class that every provider adapter must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import ToolDefinition, UnifiedResponse


class BaseProviderAdapter(ABC):
    """Contract that Anthropic, OpenAI and Gemini adapters must satisfy."""

    @abstractmethod
    def __init__(self, api_key: str, **kwargs: Any) -> None: ...

    @abstractmethod
    async def send_message(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        max_tokens: int,
        thinking_budget: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> UnifiedResponse:
        """Send a conversation turn and return a unified response."""
        ...

    # ------------------------------------------------------------------
    # Helpers that sub-classes may override
    # ------------------------------------------------------------------

    @staticmethod
    def supports_thinking() -> bool:
        """Whether this provider supports an explicit thinking/reasoning budget."""
        return False

    @staticmethod
    def supports_prompt_caching() -> bool:
        """Whether this provider supports prompt caching."""
        return False

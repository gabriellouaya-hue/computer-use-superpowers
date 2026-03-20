from .base import BaseProviderAdapter
from .types import (
    ProviderName,
    UnifiedContentBlock,
    UnifiedImageBlock,
    UnifiedResponse,
    UnifiedTextBlock,
    UnifiedThinkingBlock,
    UnifiedToolResultBlock,
    UnifiedToolUseBlock,
    ToolDefinition,
)
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider

PROVIDER_MAP: dict[ProviderName, type[BaseProviderAdapter]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
}

__all__ = [
    "BaseProviderAdapter",
    "AnthropicProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "PROVIDER_MAP",
    "ProviderName",
    "UnifiedContentBlock",
    "UnifiedImageBlock",
    "UnifiedResponse",
    "UnifiedTextBlock",
    "UnifiedThinkingBlock",
    "UnifiedToolResultBlock",
    "UnifiedToolUseBlock",
    "ToolDefinition",
]

"""Anthropic provider adapter – wraps the Anthropic SDK to the unified interface."""

from __future__ import annotations

from typing import Any, cast

from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaMessage,
    BetaMessageParam,
)

from .base import BaseProviderAdapter
from .types import (
    ToolDefinition,
    UnifiedContentBlock,
    UnifiedResponse,
    UnifiedTextBlock,
    UnifiedThinkingBlock,
    UnifiedToolUseBlock,
)

PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class AnthropicProvider(BaseProviderAdapter):
    """Adapter for the Anthropic Messages API (direct, Bedrock, Vertex)."""

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        self.api_key = api_key
        # sub_provider can be "anthropic", "bedrock", or "vertex"
        self.sub_provider: str = kwargs.get("sub_provider", "anthropic")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        extra = extra or {}
        betas: list[str] = list(extra.get("betas", []))
        token_efficient: bool = extra.get("token_efficient_tools_beta", False)

        if token_efficient:
            betas.append("token-efficient-tools-2025-02-19")

        # Build client
        enable_prompt_caching = False
        if self.sub_provider == "anthropic":
            client = Anthropic(api_key=self.api_key, max_retries=4)
            enable_prompt_caching = True
        elif self.sub_provider == "vertex":
            client = AnthropicVertex()
        elif self.sub_provider == "bedrock":
            client = AnthropicBedrock()
        else:
            raise ValueError(f"Unknown Anthropic sub-provider: {self.sub_provider}")

        # System block
        system_block: dict[str, Any] = {"type": "text", "text": system}

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            self._inject_prompt_caching(messages)
            system_block["cache_control"] = {"type": "ephemeral"}

        # Convert tool definitions to Anthropic format
        anthropic_tools = self._tools_to_anthropic(tools)

        # Extra body (thinking)
        extra_body: dict[str, Any] = {}
        if thinking_budget:
            extra_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        # API call
        try:
            raw_response = client.beta.messages.with_raw_response.create(
                max_tokens=max_tokens,
                messages=cast(list[BetaMessageParam], messages),
                model=model,
                system=[system_block],
                tools=anthropic_tools,
                betas=betas,
                extra_body=extra_body,
            )
        except (APIStatusError, APIResponseValidationError) as e:
            return UnifiedResponse(
                content=[UnifiedTextBlock(text=f"API error: {e}")],
                stop_reason="error",
                raw_request=e.request,
                raw_response=e.response,
            )
        except APIError as e:
            return UnifiedResponse(
                content=[UnifiedTextBlock(text=f"API error: {e}")],
                stop_reason="error",
                raw_request=e.request,
                raw_response=e.body,
            )

        response = raw_response.parse()

        return UnifiedResponse(
            content=self._parse_response(response),
            stop_reason=response.stop_reason,
            raw_request=raw_response.http_response.request,
            raw_response=raw_response.http_response,
        )

    @staticmethod
    def supports_thinking() -> bool:
        return True

    @staticmethod
    def supports_prompt_caching() -> bool:
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tools_to_anthropic(tools: list[ToolDefinition]) -> list[Any]:
        """Convert unified ToolDefinitions to Anthropic beta tool params."""
        result: list[Any] = []
        for t in tools:
            entry: dict[str, Any] = {"name": t.name, "type": t.type}
            if t.description:
                entry["description"] = t.description
            if t.input_schema:
                entry["input_schema"] = t.input_schema
            entry.update(t.extra)
            result.append(entry)
        return result

    @staticmethod
    def _parse_response(response: BetaMessage) -> list[UnifiedContentBlock]:
        """Convert an Anthropic BetaMessage into unified content blocks."""
        blocks: list[UnifiedContentBlock] = []
        for block in response.content:
            dumped = block.model_dump()
            block_type = dumped.get("type", "")

            if block_type == "text":
                if dumped.get("text"):
                    blocks.append(UnifiedTextBlock(text=dumped["text"]))
            elif block_type == "thinking":
                blocks.append(
                    UnifiedThinkingBlock(
                        thinking=dumped.get("thinking", "") or "",
                        signature=dumped.get("signature", None),
                    )
                )
            elif block_type == "tool_use":
                blocks.append(
                    UnifiedToolUseBlock(
                        id=dumped["id"],
                        name=dumped["name"],
                        input=dumped.get("input", {}),
                    )
                )
        return blocks

    @staticmethod
    def _inject_prompt_caching(messages: list[dict[str, Any]]) -> None:
        """Set cache breakpoints for the 3 most recent user turns."""
        breakpoints_remaining = 3
        for message in reversed(messages):
            if message["role"] == "user" and isinstance(
                content := message["content"], list
            ):
                if breakpoints_remaining:
                    breakpoints_remaining -= 1
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                else:
                    if isinstance(content[-1], dict) and "cache_control" in content[-1]:
                        del content[-1]["cache_control"]
                    break

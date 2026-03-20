"""OpenAI provider adapter – wraps the OpenAI SDK to the unified interface.

Supports any model available through the OpenAI API (GPT-4o, GPT-4.1, o3, etc.).
The adapter translates unified tool definitions into OpenAI function-calling format
and normalises responses back into unified content blocks.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from .base import BaseProviderAdapter
from .types import (
    ToolDefinition,
    UnifiedContentBlock,
    UnifiedResponse,
    UnifiedTextBlock,
    UnifiedToolUseBlock,
)


class OpenAIProvider(BaseProviderAdapter):
    """Adapter for the OpenAI Chat Completions API with tool/function calling."""

    def __init__(self, api_key: str, **kwargs: Any) -> None:
        self.api_key = api_key

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
        # Lazy import so that 'openai' is only required when actually used
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The 'openai' package is required for the OpenAI provider. "
                "Install it with: pip install openai"
            ) from exc

        client = OpenAI(api_key=self.api_key)

        # Build messages in OpenAI format
        oai_messages = self._build_messages(system, messages)

        # Build tools in OpenAI format
        oai_tools = self._tools_to_openai(tools) or None

        try:
            response = client.chat.completions.create(
                model=model,
                messages=oai_messages,
                tools=oai_tools,
                max_completion_tokens=max_tokens,
            )
        except Exception as e:
            return UnifiedResponse(
                content=[UnifiedTextBlock(text=f"OpenAI API error: {e}")],
                stop_reason="error",
            )

        return UnifiedResponse(
            content=self._parse_response(response),
            stop_reason=self._map_stop_reason(response),
            raw_request=None,
            raw_response=response,
        )

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        system: str, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert the unified message history into OpenAI chat format."""
        oai_msgs: list[dict[str, Any]] = []

        # System message
        oai_msgs.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "assistant":
                oai_msgs.append(
                    OpenAIProvider._convert_assistant_message(content)
                )
            elif role == "user":
                converted = OpenAIProvider._convert_user_message(content)
                if isinstance(converted, list):
                    oai_msgs.extend(converted)
                else:
                    oai_msgs.append(converted)
            else:
                oai_msgs.append({"role": role, "content": str(content)})

        return oai_msgs

    @staticmethod
    def _convert_assistant_message(content: Any) -> dict[str, Any]:
        """Convert an assistant message (may contain tool_use blocks) to OpenAI format."""
        if isinstance(content, str):
            return {"role": "assistant", "content": content}

        if isinstance(content, list):
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "thinking":
                    # Include thinking as text for models that don't support it natively
                    pass
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })

            msg: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if tool_calls:
                msg["tool_calls"] = tool_calls
            return msg

        return {"role": "assistant", "content": str(content)}

    @staticmethod
    def _convert_user_message(content: Any) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert a user message (may contain tool_result blocks and images) to OpenAI format.

        Returns either a single message dict or a list of message dicts when
        there are tool results that need to become separate 'tool' role messages.
        """
        if isinstance(content, str):
            return {"role": "user", "content": content}

        if isinstance(content, list):
            tool_results: list[dict[str, Any]] = []
            user_parts: list[dict[str, Any]] = []
            text_parts: list[str] = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "tool_result":
                    # Must become a separate message with role=tool
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        pieces: list[str] = []
                        image_parts: list[dict[str, Any]] = []
                        for item in result_content:
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    pieces.append(item.get("text", ""))
                                elif item.get("type") == "image":
                                    source = item.get("source", {})
                                    if source.get("type") == "base64":
                                        image_parts.append({
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}",
                                            },
                                        })
                        result_text = "\n".join(pieces) if pieces else ""
                        # For tool results with images, we add images as a follow-up user message
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": result_text,
                        })
                        if image_parts:
                            user_parts.extend(image_parts)
                    elif isinstance(result_content, str):
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": result_content,
                        })
                elif btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "image":
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        user_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}",
                            },
                        })

            # Build the return value
            result_msgs: list[dict[str, Any]] = []

            # Tool result messages come first
            result_msgs.extend(tool_results)

            # Then user content (text + images)
            if text_parts or user_parts:
                combined_content: list[dict[str, Any]] = []
                if text_parts:
                    combined_content.append({"type": "text", "text": "\n".join(text_parts)})
                combined_content.extend(user_parts)
                if combined_content:
                    result_msgs.append({"role": "user", "content": combined_content})

            if not result_msgs:
                return {"role": "user", "content": ""}
            if len(result_msgs) == 1:
                return result_msgs[0]
            return result_msgs

        return {"role": "user", "content": str(content)}

    # ------------------------------------------------------------------
    # Tool conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _tools_to_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Convert unified tool definitions to OpenAI function-calling format."""
        result: list[dict[str, Any]] = []
        for t in tools:
            # Build a JSON schema for the function parameters
            if t.input_schema:
                parameters = t.input_schema
            else:
                # For Anthropic-native tools (computer, bash, edit) that don't
                # carry an explicit input_schema, we build one from their type.
                parameters = _build_schema_for_native_tool(t)

            func_def: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": t.name,
                    "parameters": parameters,
                },
            }
            if t.description:
                func_def["function"]["description"] = t.description
            result.append(func_def)
        return result

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: Any) -> list[UnifiedContentBlock]:
        """Parse an OpenAI ChatCompletion response into unified blocks."""
        blocks: list[UnifiedContentBlock] = []
        choice = response.choices[0] if response.choices else None
        if not choice:
            return blocks

        msg = choice.message

        # Text content
        if msg.content:
            blocks.append(UnifiedTextBlock(text=msg.content))

        # Tool calls
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc.function.arguments}
                blocks.append(
                    UnifiedToolUseBlock(
                        id=tc.id,
                        name=tc.function.name,
                        input=args,
                    )
                )

        return blocks

    @staticmethod
    def _map_stop_reason(response: Any) -> str | None:
        choice = response.choices[0] if response.choices else None
        if not choice:
            return None
        reason = choice.finish_reason
        if reason == "tool_calls":
            return "tool_use"
        return reason


# ---------------------------------------------------------------------------
# Schema builders for Anthropic-native tools
# ---------------------------------------------------------------------------

def _build_schema_for_native_tool(t: ToolDefinition) -> dict[str, Any]:
    """Build a JSON Schema for tools that Anthropic defines natively
    (computer, bash, edit) so OpenAI can use them as function calls."""

    if t.name == "computer":
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "key", "type", "mouse_move", "left_click", "left_click_drag",
                        "right_click", "middle_click", "double_click", "triple_click",
                        "screenshot", "cursor_position", "left_mouse_down", "left_mouse_up",
                        "scroll", "hold_key", "wait",
                    ],
                    "description": "The action to perform on the computer.",
                },
                "text": {"type": "string", "description": "Text to type or key to press."},
                "coordinate": {
                    "type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                    "description": "[x, y] coordinate on screen.",
                },
                "start_coordinate": {
                    "type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                    "description": "Start [x, y] for drag operations.",
                },
                "scroll_direction": {
                    "type": "string", "enum": ["up", "down", "left", "right"],
                    "description": "Direction to scroll.",
                },
                "scroll_amount": {"type": "integer", "description": "Number of scroll clicks."},
                "duration": {"type": "number", "description": "Duration in seconds for hold_key or wait."},
                "key": {"type": "string", "description": "Modifier key for click actions."},
            },
            "required": ["action"],
        }

    if t.name == "bash":
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute."},
                "restart": {"type": "boolean", "description": "Whether to restart the bash session."},
            },
        }

    if t.name == "str_replace_based_edit_tool":
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert"],
                    "description": "The editing command to perform.",
                },
                "path": {"type": "string", "description": "Absolute path to the file."},
                "file_text": {"type": "string", "description": "Content for file creation."},
                "view_range": {
                    "type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2,
                    "description": "[start_line, end_line] for view command.",
                },
                "old_str": {"type": "string", "description": "String to replace."},
                "new_str": {"type": "string", "description": "Replacement string."},
                "insert_line": {"type": "integer", "description": "Line number for insert command."},
                "insert_text": {"type": "string", "description": "Text to insert."},
            },
            "required": ["command", "path"],
        }

    # Fallback for custom tools (like skill)
    return {
        "type": "object",
        "properties": {},
    }

"""Google Gemini provider adapter – wraps the Google GenAI SDK to the unified interface.

Supports any Gemini model (gemini-2.5-pro, gemini-2.5-flash, etc.).
The adapter translates unified tool definitions into Gemini function declarations
and normalises responses back into unified content blocks.
"""

from __future__ import annotations

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


class GeminiProvider(BaseProviderAdapter):
    """Adapter for the Google Gemini API with function calling."""

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
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-genai' package is required for the Gemini provider. "
                "Install it with: pip install google-genai"
            ) from exc

        client = genai.Client(api_key=self.api_key)

        # Build contents in Gemini format
        gemini_contents = self._build_contents(messages)

        # Build tools in Gemini format
        gemini_tools = self._tools_to_gemini(tools, genai_types)

        # Build config
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            tools=gemini_tools if gemini_tools else None,
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=gemini_contents,
                config=config,
            )
        except Exception as e:
            return UnifiedResponse(
                content=[UnifiedTextBlock(text=f"Gemini API error: {e}")],
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
    def _build_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert unified message history into Gemini contents format."""
        contents: list[dict[str, Any]] = []

        # Build a map from tool_use_id -> function name across all messages
        tool_id_to_name: dict[str, str] = {}
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_id_to_name[block.get("id", "")] = block.get("name", "unknown")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Map roles: Gemini uses "user" and "model"
            gemini_role = "model" if role == "assistant" else "user"

            if isinstance(content, str):
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}],
                })
            elif isinstance(content, list):
                parts = GeminiProvider._convert_content_blocks(content, role, tool_id_to_name)
                if parts:
                    # Tool results in Gemini must be in a "user" role message
                    # but function calls are in "model" role
                    has_function_response = any(
                        "function_response" in p for p in parts
                    )
                    has_function_call = any(
                        "function_call" in p for p in parts
                    )

                    if has_function_call:
                        contents.append({"role": "model", "parts": parts})
                    elif has_function_response:
                        contents.append({"role": "user", "parts": parts})
                    else:
                        contents.append({"role": gemini_role, "parts": parts})
            else:
                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": str(content)}],
                })

        return contents

    @staticmethod
    def _convert_content_blocks(
        blocks: list[Any], original_role: str, tool_id_to_name: dict[str, str] | None = None
    ) -> list[dict[str, Any]]:
        """Convert a list of content blocks to Gemini parts."""
        tool_id_to_name = tool_id_to_name or {}
        parts: list[dict[str, Any]] = []

        for block in blocks:
            if not isinstance(block, dict):
                continue

            btype = block.get("type", "")

            if btype == "text":
                text = block.get("text", "")
                if text:
                    parts.append({"text": text})

            elif btype == "thinking":
                # Skip thinking blocks — Gemini doesn't have an equivalent
                pass

            elif btype == "tool_use":
                # Assistant requesting a tool call → function_call part
                parts.append({
                    "function_call": {
                        "name": block.get("name", ""),
                        "args": block.get("input", {}),
                    }
                })

            elif btype == "tool_result":
                # Tool result → function_response part
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    text_pieces: list[str] = []
                    for item in result_content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_pieces.append(item.get("text", ""))
                            elif item.get("type") == "image":
                                source = item.get("source", {})
                                if source.get("type") == "base64":
                                    # Add image as inline_data part
                                    parts.append({
                                        "inline_data": {
                                            "mime_type": source.get("media_type", "image/png"),
                                            "data": source.get("data", ""),
                                        }
                                    })
                    result_text = "\n".join(text_pieces) if text_pieces else ""
                elif isinstance(result_content, str):
                    result_text = result_content
                else:
                    result_text = str(result_content)

                tool_use_id = block.get("tool_use_id", "")
                func_name = tool_id_to_name.get(tool_use_id, tool_use_id or "unknown")
                parts.append({
                    "function_response": {
                        "name": func_name,
                        "response": {"result": result_text},
                    }
                })

            elif btype == "image":
                source = block.get("source", {})
                if source.get("type") == "base64":
                    parts.append({
                        "inline_data": {
                            "mime_type": source.get("media_type", "image/png"),
                            "data": source.get("data", ""),
                        }
                    })

        return parts

    # ------------------------------------------------------------------
    # Tool conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _tools_to_gemini(
        tools: list[ToolDefinition], genai_types: Any
    ) -> list[Any] | None:
        """Convert unified tool definitions to Gemini function declarations."""
        if not tools:
            return None

        declarations: list[Any] = []
        for t in tools:
            if t.input_schema:
                parameters = _clean_schema_for_gemini(t.input_schema)
            else:
                parameters = _build_gemini_schema_for_native_tool(t)

            decl = genai_types.FunctionDeclaration(
                name=t.name,
                description=t.description or f"Tool: {t.name}",
                parameters=parameters,
            )
            declarations.append(decl)

        return [genai_types.Tool(function_declarations=declarations)]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: Any) -> list[UnifiedContentBlock]:
        """Parse a Gemini GenerateContentResponse into unified blocks."""
        blocks: list[UnifiedContentBlock] = []

        if not response.candidates:
            return blocks

        candidate = response.candidates[0]

        if not candidate.content or not candidate.content.parts:
            return blocks

        for part in candidate.content.parts:
            # Text part
            if hasattr(part, "text") and part.text:
                blocks.append(UnifiedTextBlock(text=part.text))

            # Function call part
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                args = dict(fc.args) if fc.args else {}
                blocks.append(
                    UnifiedToolUseBlock(
                        id=f"call_{uuid.uuid4().hex[:12]}",
                        name=fc.name,
                        input=args,
                    )
                )

        return blocks

    @staticmethod
    def _map_stop_reason(response: Any) -> str | None:
        if not response.candidates:
            return None
        candidate = response.candidates[0]
        reason = getattr(candidate, "finish_reason", None)
        if reason is not None:
            reason_str = str(reason)
            if "STOP" in reason_str:
                return "end_turn"
            if "TOOL" in reason_str or "FUNCTION" in reason_str:
                return "tool_use"
        return str(reason) if reason else None


# ---------------------------------------------------------------------------
# Schema helpers for Gemini
# ---------------------------------------------------------------------------

def _clean_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Clean a JSON Schema to be compatible with Gemini's function calling.
    Gemini doesn't support all JSON Schema features."""
    cleaned = dict(schema)
    # Remove unsupported keys
    for key in ["$schema", "additionalProperties", "default"]:
        cleaned.pop(key, None)
    # Recursively clean nested properties
    if "properties" in cleaned:
        cleaned["properties"] = {
            k: _clean_schema_for_gemini(v)
            for k, v in cleaned["properties"].items()
        }
    if "items" in cleaned and isinstance(cleaned["items"], dict):
        cleaned["items"] = _clean_schema_for_gemini(cleaned["items"])
    return cleaned


def _build_gemini_schema_for_native_tool(t: ToolDefinition) -> dict[str, Any]:
    """Build a Gemini-compatible schema for Anthropic-native tools."""

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
                    "type": "array", "items": {"type": "integer"},
                    "description": "[x, y] coordinate on screen.",
                },
                "start_coordinate": {
                    "type": "array", "items": {"type": "integer"},
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
                    "type": "array", "items": {"type": "integer"},
                    "description": "[start_line, end_line] for view command.",
                },
                "old_str": {"type": "string", "description": "String to replace."},
                "new_str": {"type": "string", "description": "Replacement string."},
                "insert_line": {"type": "integer", "description": "Line number for insert command."},
                "insert_text": {"type": "string", "description": "Text to insert."},
            },
            "required": ["command", "path"],
        }

    return {"type": "object", "properties": {}}

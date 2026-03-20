"""
Agentic sampling loop that calls LLM providers and local implementation of computer use tools.

Supports Anthropic (direct / Bedrock / Vertex), OpenAI, and Google Gemini.
"""

import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from .providers import PROVIDER_MAP
from .providers.types import (
    ToolDefinition,
    UnifiedContentBlock,
    UnifiedResponse,
    UnifiedTextBlock,
    UnifiedThinkingBlock,
    UnifiedToolUseBlock,
)
from .tools import (
    TOOL_GROUPS_BY_VERSION,
    ToolCollection,
    ToolResult,
    ToolVersion,
)

MANDATORY_SKILL_PATH = Path(__file__).resolve().parent / "skills" / "using-superpowers.md"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OPENAI = "openai"
    GEMINI = "gemini"


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_based_edit_tool or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* You have access to a skill tool containing reusable workflows. For multi-step tasks or recurring patterns, consult the skill tool before acting so you can follow an established procedure when one is available.
* The current date is {datetime.today().strftime("%A, %B %-d, %Y")}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your str_replace_based_edit_tool.
* Before starting a task that involves browsing, research, editing files, or planning several steps, consider calling the skill tool to list, search, or read the most relevant workflow.
</IMPORTANT>"""


def _load_mandatory_skill_prompt() -> str:
    if not MANDATORY_SKILL_PATH.exists():
        return ""
    skill_text = MANDATORY_SKILL_PATH.read_text().strip()
    if not skill_text:
        return ""
    return f"\n\n<MANDATORY_SKILL>\nYou must follow the mandatory skill below for every user task.\n\n{skill_text}\n</MANDATORY_SKILL>"


async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[dict[str, Any]],
    output_callback: Callable[[dict[str, Any]], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [Any, Any, Exception | None], None
    ],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    tool_version: ToolVersion,
    thinking_budget: int | None = None,
    token_efficient_tools_beta: bool = False,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    Works with Anthropic, OpenAI, and Gemini via the provider adapter layer.
    """
    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))
    mandatory_skill_prompt = _load_mandatory_skill_prompt()
    system_text = (
        f"{SYSTEM_PROMPT}"
        f"{mandatory_skill_prompt}"
        f"{' ' + system_prompt_suffix if system_prompt_suffix else ''}"
    )

    # Resolve provider adapter
    adapter_cls = _resolve_adapter(provider)
    adapter_kwargs: dict[str, Any] = {}
    if provider in (APIProvider.BEDROCK, APIProvider.VERTEX):
        adapter_kwargs["sub_provider"] = provider.value
    adapter = adapter_cls(api_key=api_key, **adapter_kwargs)

    # Convert tool collection to unified ToolDefinitions
    tool_definitions = _tool_collection_to_definitions(tool_collection)

    while True:
        image_truncation_threshold = only_n_most_recent_images or 0

        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                messages,
                only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )

        # Build extra params for provider-specific features
        extra: dict[str, Any] = {}
        betas = [tool_group.beta_flag] if tool_group.beta_flag else []
        if token_efficient_tools_beta:
            extra["token_efficient_tools_beta"] = True
        extra["betas"] = betas

        # Call the provider adapter
        response: UnifiedResponse = await adapter.send_message(
            model=model,
            system=system_text,
            messages=messages,
            tools=tool_definitions,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget if adapter.supports_thinking() else None,
            extra=extra,
        )

        # Notify the UI about the raw request/response for logging
        api_response_callback(
            response.raw_request,
            response.raw_response,
            None if response.stop_reason != "error" else Exception(str(response.content)),
        )

        if response.stop_reason == "error":
            return messages

        # Convert unified response to message-history dict format
        response_params = _unified_to_params(response.content)
        messages.append(
            {
                "role": "assistant",
                "content": response_params,
            }
        )

        tool_result_content: list[dict[str, Any]] = []
        for content_block in response_params:
            output_callback(content_block)
            if (
                isinstance(content_block, dict)
                and content_block.get("type") == "tool_use"
            ):
                result = await tool_collection.run(
                    name=content_block["name"],
                    tool_input=cast(dict[str, Any], content_block.get("input", {})),
                )
                tool_result_content.append(
                    _make_api_tool_result(result, content_block["id"])
                )
                tool_output_callback(result, content_block["id"])

        if not tool_result_content:
            return messages

        messages.append({"content": tool_result_content, "role": "user"})


def _resolve_adapter(provider: APIProvider):
    """Map an APIProvider enum value to the correct adapter class."""
    from .providers import PROVIDER_MAP

    if provider in (APIProvider.ANTHROPIC, APIProvider.BEDROCK, APIProvider.VERTEX):
        return PROVIDER_MAP["anthropic"]
    elif provider == APIProvider.OPENAI:
        return PROVIDER_MAP["openai"]
    elif provider == APIProvider.GEMINI:
        return PROVIDER_MAP["gemini"]
    raise ValueError(f"Unsupported provider: {provider}")


def _tool_collection_to_definitions(tool_collection: ToolCollection) -> list[ToolDefinition]:
    """Convert a ToolCollection into a list of provider-agnostic ToolDefinitions."""
    definitions: list[ToolDefinition] = []
    for tool in tool_collection.tools:
        params = cast(dict[str, Any], tool.to_params())
        name = params.get("name", "")
        tool_type = params.get("type", "custom")
        description = params.get("description", None)
        input_schema = params.get("input_schema", None)
        # Collect extra keys (e.g. display_width_px, display_height_px for computer tool)
        known_keys = {"name", "type", "description", "input_schema"}
        extra = {k: v for k, v in params.items() if k not in known_keys}
        definitions.append(
            ToolDefinition(
                name=name,
                type=tool_type,
                description=description,
                input_schema=input_schema,
                extra=extra,
            )
        )
    return definitions


def _unified_to_params(content: list[UnifiedContentBlock]) -> list[dict[str, Any]]:
    """Convert unified content blocks into the dict format used by the message history."""
    params: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, UnifiedTextBlock):
            params.append({"type": "text", "text": block.text})
        elif isinstance(block, UnifiedThinkingBlock):
            entry: dict[str, Any] = {
                "type": "thinking",
                "thinking": block.thinking,
            }
            if block.signature:
                entry["signature"] = block.signature
            params.append(entry)
        elif isinstance(block, UnifiedToolUseBlock):
            params.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
    return params


def _maybe_filter_to_n_most_recent_images(
    messages: list[dict[str, Any]],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = [
        item
        for message in messages
        for item in (
            message["content"] if isinstance(message["content"], list) else []
        )
        if isinstance(item, dict) and item.get("type") == "tool_result"
    ]

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> dict[str, Any]:
    """Convert an agent ToolResult to a tool_result dict for the message history."""
    tool_result_content: list[dict[str, Any]] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text

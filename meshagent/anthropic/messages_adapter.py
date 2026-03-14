from __future__ import annotations

from meshagent.agents.agent import AgentSessionContext
from meshagent.api import RoomClient, RoomException, RemoteParticipant
from meshagent.agents.event_publisher import (
    _AnthropicAgentEventPublisher,
    make_anthropic_agent_event_publisher,
)
from meshagent.agents.messages import AgentMessage
from meshagent.tools import Toolkit, ToolContext, FunctionTool, BaseTool
from meshagent.api.messaging import (
    Content,
    LinkContent,
    FileContent,
    JsonContent,
    TextContent,
    EmptyContent,
    RawOutputsContent,
    ensure_content,
)
from meshagent.agents.adapter import ToolResponseAdapter, LLMAdapter, SteeringCallback

import json
from typing import Any, Optional, Callable, Literal
from collections.abc import AsyncIterable
import os
import logging
import re
import asyncio
import base64
import mimetypes
import copy
from html_to_markdown import convert
from urllib.parse import urlparse

from meshagent.anthropic.proxy import get_client, get_logging_httpx_client
from meshagent.anthropic.mcp import MCPTool as MCPConnectorTool
from meshagent.anthropic.usage import (
    add_usage_metrics,
    normalize_anthropic_usage,
    preprocess_anthropic_usage,
)
from meshagent.tools.strict_schema import ensure_strict_json_schema

try:
    from anthropic import APIStatusError, transform_schema
except Exception:  # pragma: no cover
    APIStatusError = Exception  # type: ignore
    transform_schema = None  # type: ignore

logger = logging.getLogger("anthropic_agent")

_CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"
_COMPACTION_BETA = "compact-2026-01-12"
ToolCallingMode = Literal["loose", "strict", "explicit", "adaptive"]
_LEGACY_ANTHROPIC_MAX_TOKENS = 8192
_MAX_PAUSE_TURN_CONTINUATIONS = 5
_ANTHROPIC_STEERING_MARKER = "TURN INTERRUPTED"
_ANTHROPIC_STEERING_INSTRUCTIONS = (
    "If the transcript contains an assistant message exactly equal to "
    f"'{_ANTHROPIC_STEERING_MARKER}', then treat the immediately following user "
    "message as steering that takes precedence over any unfinished prior plan."
)


class _AnthropicToolCallingState:
    def __init__(self, *, mode: ToolCallingMode):
        self.mode = mode
        self._adaptive_strict_tools: set[str] = set()
        self._adaptive_strict_disabled = False

    def strict_for_tool(self, *, tool_name: str, tool: FunctionTool) -> bool:
        if self.mode == "loose":
            return False
        if self.mode == "strict":
            return True
        if self.mode == "explicit":
            return tool.strict
        if self._adaptive_strict_disabled:
            return False
        return tool.strict and tool_name in self._adaptive_strict_tools

    def record_tool_failure(self, *, tool_name: str, tool: FunctionTool) -> None:
        if self.mode != "adaptive":
            return
        if self._adaptive_strict_disabled:
            return
        if not tool.strict:
            return
        self._adaptive_strict_tools.add(tool_name)

    def disable_all_strict(self) -> None:
        if self.mode != "adaptive":
            return
        self._adaptive_strict_tools.clear()
        self._adaptive_strict_disabled = True


def _transform_strict_anthropic_schema(schema: dict[str, Any]) -> dict[str, Any]:
    normalized_schema = _normalize_anthropic_union_types(schema)
    if transform_schema is None:
        return normalized_schema
    return transform_schema(normalized_schema)


def _default_max_tokens_for_model(model: str) -> int:
    normalized_model = model.strip().lower()
    if normalized_model.startswith("claude-opus-4-6"):
        return 128_000
    if normalized_model.startswith("claude-sonnet-4"):
        return 64_000
    if normalized_model.startswith("claude-3-7-sonnet"):
        return 64_000
    if normalized_model.startswith("claude-opus-4"):
        return 32_000
    return _LEGACY_ANTHROPIC_MAX_TOKENS


def _normalize_anthropic_union_types(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_anthropic_union_types(item) for item in value]

    if not isinstance(value, dict):
        return value

    normalized = {
        key: _normalize_anthropic_union_types(item) for key, item in value.items()
    }
    normalized = _lower_nullable_object_properties(normalized)
    type_value = normalized.get("type")
    if not isinstance(type_value, list):
        return _strip_empty_object_compound_wrapper(normalized)

    normalized_types = [item for item in type_value if isinstance(item, str)]
    if len(normalized_types) == 0:
        return normalized

    if len(normalized_types) == 1:
        normalized["type"] = normalized_types[0]
        return normalized

    base_variant = {key: item for key, item in normalized.items() if key != "type"}
    any_of: list[dict[str, Any]] = []
    for item_type in normalized_types:
        if item_type == "null":
            any_of.append({"type": "null"})
            continue

        variant = dict(base_variant)
        variant["type"] = item_type
        any_of.append(variant)

    normalized = {"anyOf": any_of}
    return _strip_empty_object_compound_wrapper(normalized)


def _content_blocks_to_text(*, content: list[dict[str, Any]]) -> str:
    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip() != "":
                text_parts.append(text.strip())
            continue

        if block_type == "image":
            source = block.get("source")
            if isinstance(source, dict):
                media_type = source.get("media_type")
                if isinstance(media_type, str) and media_type.strip() != "":
                    text_parts.append(
                        f"the user attached an image ({media_type.strip()})"
                    )
                    continue
            text_parts.append("the user attached an image")
            continue

        if block_type == "file":
            file_name = block.get("filename")
            if isinstance(file_name, str) and file_name.strip() != "":
                text_parts.append(f"the user attached a file: {file_name.strip()}")
                continue
            text_parts.append("the user attached a file")

    return "\n\n".join(text_parts)


def _lower_nullable_object_properties(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if value.get("type") != "object":
        return value

    properties = value.get("properties")
    if not isinstance(properties, dict):
        return value

    required_value = value.get("required")
    required = (
        [item for item in required_value if isinstance(item, str)]
        if isinstance(required_value, list)
        else None
    )

    updated_properties = dict[str, Any]()
    updated_required = None if required is None else [*required]
    changed = False

    for key, property_schema in properties.items():
        lowered_schema = _extract_optional_property_schema(property_schema)
        if lowered_schema is None:
            updated_properties[key] = property_schema
            continue

        changed = True
        updated_properties[key] = lowered_schema
        if updated_required is not None and key in updated_required:
            updated_required.remove(key)

    if not changed:
        return value

    normalized = dict(value)
    normalized["properties"] = updated_properties
    if updated_required is not None:
        if len(updated_required) == 0:
            normalized.pop("required", None)
        else:
            normalized["required"] = updated_required
    return normalized


def _extract_optional_property_schema(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    any_of = value.get("anyOf")
    if not isinstance(any_of, list) or len(any_of) != 2:
        return None

    non_null_variants = list[dict[str, Any]]()
    has_null_variant = False
    for variant in any_of:
        if not isinstance(variant, dict):
            return None
        if variant.get("type") == "null":
            has_null_variant = True
            continue
        non_null_variants.append(variant)

    if not has_null_variant or len(non_null_variants) != 1:
        return None

    return non_null_variants[0]


def _strip_empty_object_compound_wrapper(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    has_compound = any(
        isinstance(value.get(key), list) for key in ("anyOf", "oneOf", "allOf")
    )
    if not has_compound:
        return value

    properties = value.get("properties")
    required = value.get("required")
    additional_properties = value.get("additionalProperties")
    if value.get("type") != "object":
        return value
    if properties not in (None, {}):
        return value
    if required not in (None, []):
        return value
    if additional_properties not in (None, False):
        return value

    stripped = dict(value)
    stripped.pop("type", None)
    stripped.pop("properties", None)
    stripped.pop("required", None)
    stripped.pop("additionalProperties", None)
    return stripped


def _is_html_mime_type(mime_type: str | None) -> bool:
    if not mime_type:
        return False
    normalized = mime_type.split(";")[0].strip().lower()
    return normalized in {"text/html", "application/xhtml+xml"}


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _replace_non_matching(text: str, allowed_chars: str, replacement: str) -> str:
    pattern = rf"[^{allowed_chars}]"
    return re.sub(pattern, replacement, text)


def safe_tool_name(name: str) -> str:
    return _replace_non_matching(name, "a-zA-Z0-9_-", "_")


def _as_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return obj
    return obj.model_dump(
        mode="json",
        exclude_none=True,
        exclude_unset=True,
    )


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _tool_result_message(
    *,
    tool_use_id: str | None,
    content: list[dict],
) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
        ],
    }


def _cancelled_tool_result_message(*, tool_use: dict) -> dict:
    return _tool_result_message(
        tool_use_id=tool_use.get("id"),
        content=[
            _text_block(
                "Cancelled because queued steering took priority over the remaining tool calls."
            )
        ],
    )


def _split_tool_execution_messages(
    results: list[list[dict]],
) -> tuple[list[dict], list[dict]]:
    tool_result_blocks: list[dict] = []
    trailing_messages: list[dict] = []

    for msgs in results:
        for msg in msgs:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), list)
                and all(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in msg["content"]
                )
            ):
                tool_result_blocks.extend(msg["content"])
            else:
                trailing_messages.append(msg)

    return tool_result_blocks, trailing_messages


async def _consume_streaming_tool_result(
    *, stream: AsyncIterable[Any], event_handler: Optional[Callable[[dict], None]]
) -> Content:
    has_last = False
    last_item: Any = None
    async for item in stream:
        if (
            has_last
            and isinstance(last_item, JsonContent)
            and event_handler is not None
        ):
            event_handler(last_item.json)
        last_item = item
        has_last = True

    if not has_last:
        return ensure_content(None)

    if isinstance(last_item, dict):
        last_type = last_item.get("type")
        if last_type in ("agent.event", "codex.event"):
            return ensure_content(None)

    return ensure_content(last_item)


class AnthropicMessagesChatContext(AgentSessionContext):
    @property
    def supports_images(self) -> bool:
        return True

    @property
    def supports_files(self) -> bool:
        return True

    def append_image_message(self, *, mime_type: str, data: bytes) -> dict:
        normalized_mime_type = mime_type.lower().strip()
        if normalized_mime_type == "image/jpg":
            normalized_mime_type = "image/jpeg"

        allowed_mime_types = {"image/jpeg", "image/png", "image/gif", "image/webp"}
        if normalized_mime_type not in allowed_mime_types:
            message = {
                "role": "user",
                "content": [
                    _text_block(
                        f"the user attached an image in unsupported format {normalized_mime_type}"
                    )
                ],
            }
            self.messages.append(message)
            return message

        message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": normalized_mime_type,
                        "data": base64.b64encode(data).decode("utf-8"),
                    },
                }
            ],
        }
        self.messages.append(message)
        return message

    def append_image_url(self, *, url: str) -> dict:
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": url,
                    },
                }
            ],
        }
        self.messages.append(message)
        return message

    def append_file_message(
        self, *, filename: str, mime_type: str, data: bytes
    ) -> dict:
        normalized_mime_type = (mime_type or "application/octet-stream").lower().strip()

        if normalized_mime_type.startswith("image/"):
            return self.append_image_message(
                mime_type=normalized_mime_type,
                data=data,
            )

        if normalized_mime_type == "application/pdf":
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "title": filename,
                        "source": {
                            "type": "base64",
                            "media_type": normalized_mime_type,
                            "data": base64.b64encode(data).decode("utf-8"),
                        },
                    }
                ],
            }
            self.messages.append(message)
            return message

        if (
            normalized_mime_type.startswith("text/")
            or normalized_mime_type == "application/json"
            or normalized_mime_type == "application/xhtml+xml"
        ):
            if _is_html_mime_type(normalized_mime_type):
                text = convert(_decode_text(data))
            else:
                text = _decode_text(data)

            message = {
                "role": "user",
                "content": [
                    _text_block(
                        f"attached file {filename} ({normalized_mime_type}):\n{text}"
                    )
                ],
            }
            self.messages.append(message)
            return message

        message = {
            "role": "user",
            "content": [
                _text_block(
                    f"the user attached a file named {filename} with mime type {normalized_mime_type}"
                )
            ],
        }
        self.messages.append(message)
        return message

    def append_file_url(self, *, url: str) -> dict:
        parsed_url = urlparse(url)
        guessed_mime_type, _ = mimetypes.guess_type(parsed_url.path)
        normalized_mime_type = (guessed_mime_type or "application/octet-stream").lower()

        if normalized_mime_type.startswith("image/"):
            return self.append_image_url(url=url)

        if normalized_mime_type == "application/pdf":
            title = os.path.basename(parsed_url.path) or "attachment.pdf"
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "title": title,
                        "source": {
                            "type": "url",
                            "url": url,
                        },
                    }
                ],
            }
            self.messages.append(message)
            return message

        message = {
            "role": "user",
            "content": [_text_block(f"the user attached a file available at {url}")],
        }
        self.messages.append(message)
        return message


class MessagesToolBundle:
    def __init__(
        self,
        toolkits: list[Toolkit],
        *,
        tool_calling_state: _AnthropicToolCallingState | None = None,
    ):
        self._executors: dict[str, Toolkit] = {}
        self._safe_names: dict[str, str] = {}
        self._tools_by_safe_name: dict[str, FunctionTool] = {}
        self._effective_strict_by_safe_name: dict[str, bool] = {}
        self._tool_calling_state = (
            tool_calling_state
            if tool_calling_state is not None
            else _AnthropicToolCallingState(mode="explicit")
        )

        tools: list[dict] = []

        for toolkit in toolkits:
            for v in toolkit.tools:
                if not isinstance(v, FunctionTool):
                    raise RoomException(f"unsupported tool type {type(v)}")

                original_name = v.name
                safe_name = safe_tool_name(original_name)

                if original_name in self._executors:
                    raise Exception(
                        f"duplicate in bundle '{original_name}', tool names must be unique."
                    )

                self._executors[original_name] = toolkit
                self._safe_names[safe_name] = original_name
                self._tools_by_safe_name[safe_name] = v

                schema = {**v.input_schema}
                if v.defs is not None:
                    schema["$defs"] = v.defs
                effective_strict = self._tool_calling_state.strict_for_tool(
                    tool_name=original_name,
                    tool=v,
                )
                self._effective_strict_by_safe_name[safe_name] = effective_strict
                if effective_strict:
                    schema = _transform_strict_anthropic_schema(schema)

                tools.append(
                    {
                        "name": safe_name,
                        "description": v.description,
                        "input_schema": schema,
                        "strict": effective_strict,
                    }
                )

        self._tools = tools or None

    def to_json(self) -> list[dict] | None:
        return None if self._tools is None else self._tools.copy()

    def uses_strict_tools(self) -> bool:
        return any(self._effective_strict_by_safe_name.values())

    def get_tool(self, safe_name: str) -> FunctionTool | None:
        return self._tools_by_safe_name.get(safe_name)

    def resolve_function_tool_name(self, safe_name: str) -> tuple[str, str] | None:
        original_name = self._safe_names.get(safe_name)
        if original_name is None:
            return None

        toolkit = self._executors.get(original_name)
        if toolkit is None:
            return None

        return toolkit.name, original_name

    def record_tool_failure(self, *, tool_use: dict) -> None:
        safe_name = tool_use.get("name")
        if not isinstance(safe_name, str):
            return
        original_name = self._safe_names.get(safe_name)
        if original_name is None:
            return
        tool = self._tools_by_safe_name.get(safe_name)
        if tool is None:
            return
        self._tool_calling_state.record_tool_failure(
            tool_name=original_name,
            tool=tool,
        )

    def disable_all_strict(self) -> None:
        self._tool_calling_state.disable_all_strict()

    def validation_mode_for_tool_use(self, *, tool_use: dict) -> str | None:
        safe_name = tool_use.get("name")
        if not isinstance(safe_name, str):
            return None
        if self._effective_strict_by_safe_name.get(safe_name) is True:
            return "content_types"
        return None

    async def execute(
        self, *, context: ToolContext, tool_use: dict
    ) -> Content | AsyncIterable[Any]:
        safe_name = tool_use.get("name")
        if safe_name not in self._safe_names:
            raise RoomException(
                f"Invalid tool name {safe_name}, check the name of the tool"
            )

        name = self._safe_names[safe_name]
        if name not in self._executors:
            raise Exception(f"Unregistered tool name {name}")

        arguments = tool_use.get("input") or {}
        proxy = self._executors[name]
        result = await proxy.execute(
            context=context,
            name=name,
            input=JsonContent(json=arguments),
        )
        return result


class AnthropicMessagesToolResponseAdapter(ToolResponseAdapter):
    async def to_plain_text(self, *, room: RoomClient, response: Content) -> str:
        if isinstance(response, LinkContent):
            return json.dumps({"name": response.name, "url": response.url})
        if isinstance(response, JsonContent):
            return json.dumps(response.json)
        if isinstance(response, TextContent):
            return response.text
        if isinstance(response, FileContent):
            return response.name
        if isinstance(response, EmptyContent):
            return "ok"
        if isinstance(response, dict):
            return json.dumps(response)
        if isinstance(response, str):
            return response
        if response is None:
            return "ok"
        raise Exception("unexpected return type: {type}".format(type=type(response)))

    async def create_messages(
        self,
        *,
        context: AgentSessionContext,
        tool_call: Any,
        room: RoomClient,
        response: Content,
    ) -> list:
        tool_use = tool_call if isinstance(tool_call, dict) else _as_jsonable(tool_call)
        tool_use_id = tool_use.get("id")
        if tool_use_id is None:
            raise RoomException("anthropic tool_use block was missing an id")

        if isinstance(response, RawOutputsContent):
            # Allow advanced tools to return pre-built Anthropic blocks.
            return [{"role": "user", "content": response.outputs}]

        tool_result_content: list[dict]
        try:
            if isinstance(response, FileContent):
                mime_type = (response.mime_type or "").lower()

                if mime_type == "image/jpg":
                    mime_type = "image/jpeg"

                if mime_type.startswith("image/"):
                    allowed = {"image/jpeg", "image/png", "image/gif", "image/webp"}
                    if mime_type not in allowed:
                        output = f"{response.name} was returned as {response.mime_type}, which Anthropic does not accept as an image block"
                        tool_result_content = [_text_block(output)]
                    else:
                        tool_result_content = [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime_type,
                                    "data": base64.b64encode(response.data).decode(
                                        "utf-8"
                                    ),
                                },
                            }
                        ]

                elif mime_type == "application/pdf":
                    tool_result_content = [
                        {
                            "type": "document",
                            "title": response.name,
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": base64.b64encode(response.data).decode("utf-8"),
                            },
                        }
                    ]

                elif (
                    mime_type.startswith("text/")
                    or mime_type == "application/json"
                    or mime_type == "application/xhtml+xml"
                ):
                    if _is_html_mime_type(mime_type):
                        text = convert(_decode_text(response.data))
                    else:
                        text = _decode_text(response.data)
                    tool_result_content = [_text_block(text)]

                else:
                    output = await self.to_plain_text(room=room, response=response)
                    tool_result_content = [_text_block(output)]

            else:
                output = await self.to_plain_text(room=room, response=response)
                tool_result_content = [_text_block(output)]

        except Exception as ex:
            logger.error("unable to process tool call results", exc_info=ex)
            tool_result_content = [_text_block(f"Error: {ex}")]

        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": tool_result_content,
                }
            ],
        }

        room.developer.log_nowait(
            type="llm.message",
            data={
                "context": context.id,
                "participant_id": room.local_participant.id,
                "participant_name": room.local_participant.get_attribute("name"),
                "message": message,
            },
        )

        return [message]


class AnthropicMessagesAdapter(LLMAdapter[dict]):
    def __init__(
        self,
        model: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        max_tokens: int | None = None,
        client: Optional[Any] = None,
        message_options: Optional[dict] = None,
        provider: str = "anthropic",
        log_requests: bool = False,
        context_management: Literal["auto", "none"] = "none",
        compaction_threshold: int = 150000,
        compaction_pause_after: bool = False,
        compaction_instructions: Optional[str] = None,
        tool_calling_mode: ToolCallingMode = "adaptive",
    ):
        if context_management not in ("auto", "none"):
            raise ValueError("context_management must be one of 'auto' or 'none'")
        if tool_calling_mode not in ("loose", "strict", "explicit", "adaptive"):
            raise ValueError(
                "tool_calling_mode must be one of 'loose', 'strict', 'explicit', or 'adaptive'"
            )
        if compaction_threshold < 50000:
            raise ValueError(
                "compaction_threshold must be greater than or equal to 50000"
            )
        self._model = model
        env_max_tokens = os.getenv("ANTHROPIC_MAX_TOKENS")
        self._max_tokens = (
            int(env_max_tokens)
            if env_max_tokens is not None and env_max_tokens != ""
            else max_tokens
        )
        self._client = client
        self._message_options = message_options or {}
        self._provider = provider
        self._log_requests = log_requests
        self._context_management_mode = context_management
        self._compaction_threshold = compaction_threshold
        self._compaction_pause_after = compaction_pause_after
        self._compaction_instructions = compaction_instructions
        self._tool_calling_state = _AnthropicToolCallingState(mode=tool_calling_mode)

    def default_model(self) -> str:
        return self._model

    def get_additional_instructions(self) -> str | None:
        return _ANTHROPIC_STEERING_INSTRUCTIONS

    def on_turn_steer(self, *, context: AgentSessionContext, interrupted: bool) -> None:
        del interrupted
        context.append_assistant_message(_ANTHROPIC_STEERING_MARKER)

    def _resolved_max_tokens(self, *, model: str) -> int:
        if self._max_tokens is not None:
            return self._max_tokens
        return _default_max_tokens_for_model(model)

    def _stop_reason_error_message(self, *, stop_reason: str) -> str | None:
        if stop_reason == "max_tokens":
            return (
                "Anthropic response hit max_tokens before completing the turn. "
                "Increase max_tokens and retry."
            )
        if stop_reason == "model_context_window_exceeded":
            return "Anthropic response hit the model context window before completing the turn."
        if stop_reason == "refusal":
            return "Anthropic refused to respond."
        return None

    def create_session(self) -> AgentSessionContext:
        return AnthropicMessagesChatContext(system_role=None)

    def make_agent_event_publisher(
        self,
        turn_id: str,
        thread_id: str,
        callback: Callable[[AgentMessage], None],
    ) -> Callable[[dict[str, Any]], None]:
        return make_anthropic_agent_event_publisher(
            turn_id=turn_id,
            thread_id=thread_id,
            callback=callback,
        )

    def _set_function_tool_name_resolver(
        self,
        *,
        event_handler: Callable[[dict[str, Any]], None] | None,
        resolver: Callable[[str], tuple[str, str] | None] | None,
    ) -> None:
        if isinstance(event_handler, _AnthropicAgentEventPublisher):
            event_handler.set_function_tool_name_resolver(resolver)

    def get_anthropic_client(self, *, room: RoomClient) -> Any:
        if self._client is not None:
            return self._client
        http_client = get_logging_httpx_client() if self._log_requests else None
        return get_client(room=room, http_client=http_client)

    @staticmethod
    def _message_to_blocks(*, role: str, content: Any) -> dict:
        if isinstance(content, str):
            return {"role": role, "content": [_text_block(content)]}
        if isinstance(content, list):
            return {"role": role, "content": content}
        return {"role": role, "content": [_text_block(str(content))]}

    def _convert_message_list(
        self, *, raw_messages: list[dict[str, Any]]
    ) -> list[dict]:
        messages: list[dict] = []
        pending_tool_use_ids: set[str] = set()

        for m in raw_messages:
            role = m.get("role")
            if role not in {"user", "assistant"}:
                continue

            msg = self._message_to_blocks(role=role, content=m.get("content"))

            # Anthropic requires that tool_result blocks appear in the *immediately next*
            # user message after an assistant tool_use.
            if pending_tool_use_ids:
                if role == "assistant":
                    # Drop any assistant chatter that appears between tool_use and tool_result.
                    logger.warning(
                        "dropping assistant message between tool_use and tool_result"
                    )
                    continue

                # role == user
                content_blocks = msg.get("content") or []
                tool_results = [
                    b
                    for b in content_blocks
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                tool_result_ids = {
                    b.get("tool_use_id") for b in tool_results if b.get("tool_use_id")
                }

                if not pending_tool_use_ids.issubset(tool_result_ids):
                    # If we can't satisfy the ordering contract, it's better to fail early
                    # with a clear error than to send an invalid request.
                    raise RoomException(
                        "invalid transcript: tool_use blocks must be followed by a user message "
                        "containing tool_result blocks for all tool_use ids"
                    )

                pending_tool_use_ids.clear()

            # Track tool_use ids introduced by assistant messages.
            if role == "assistant":
                content_blocks = msg.get("content") or []
                for b in content_blocks:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tool_id = b.get("id")
                        if tool_id:
                            pending_tool_use_ids.add(tool_id)

            messages.append(msg)

        return messages

    def _convert_messages(
        self, *, context: AgentSessionContext
    ) -> tuple[list[dict], Optional[str]]:
        system_parts = [
            part
            for part in (
                context.get_system_instructions(),
                self.get_additional_instructions(),
            )
            if isinstance(part, str) and part.strip() != ""
        ]
        system = "\n\n".join(system_parts) if len(system_parts) > 0 else None
        messages = self._convert_message_list(raw_messages=context.messages)
        return messages, system

    @staticmethod
    def _strip_trailing_user_messages(*, messages: list[dict[str, Any]]) -> list[dict]:
        trimmed_messages = list(messages)
        while len(trimmed_messages) > 0 and trimmed_messages[-1].get("role") == "user":
            trimmed_messages.pop()
        return trimmed_messages

    def _messages_api(self, *, client: Any, request: dict) -> Any:
        # The MCP connector requires `client.beta.messages.*`.
        if request.get("betas") is not None:
            return client.beta.messages
        extra_headers = request.get("extra_headers")
        if isinstance(extra_headers, dict) and extra_headers.get("anthropic-beta"):
            return client.beta.messages
        return client.messages

    async def _create_with_optional_headers(self, *, client: Any, request: dict) -> Any:
        api = self._messages_api(client=client, request=request)
        try:
            return await api.create(**request)
        except TypeError:
            request = dict(request)
            request.pop("extra_headers", None)
            return await api.create(**request)

    def _ensure_beta(self, *, request: dict[str, Any], beta: str) -> None:
        betas = request.get("betas")
        if betas is None:
            request["betas"] = [beta]
            return
        if isinstance(betas, str):
            normalized_betas = [betas]
        else:
            normalized_betas = [*betas]
        if beta not in normalized_betas:
            normalized_betas.append(beta)
        request["betas"] = normalized_betas

    def _ensure_context_management_betas(self, *, request: dict[str, Any]) -> None:
        context_management = request.get("context_management")
        if context_management is None:
            return
        if not isinstance(context_management, dict):
            self._ensure_beta(request=request, beta=_CONTEXT_MANAGEMENT_BETA)
            return
        edits = context_management.get("edits")
        if edits is None:
            self._ensure_beta(request=request, beta=_CONTEXT_MANAGEMENT_BETA)
            return

        has_compaction_edit = False
        for edit in edits:
            if isinstance(edit, dict) and edit.get("type") == "compact_20260112":
                has_compaction_edit = True
                break

        if has_compaction_edit:
            self._ensure_beta(request=request, beta=_COMPACTION_BETA)
            return
        self._ensure_beta(request=request, beta=_CONTEXT_MANAGEMENT_BETA)

    def _build_compaction_edit(self) -> dict[str, Any]:
        edit: dict[str, Any] = {
            "type": "compact_20260112",
            "trigger": {"type": "input_tokens", "value": self._compaction_threshold},
            "pause_after_compaction": self._compaction_pause_after,
        }
        if self._compaction_instructions is not None:
            edit["instructions"] = self._compaction_instructions
        return edit

    @staticmethod
    def _supports_auto_compaction_model(*, model: str) -> bool:
        normalized_model = model.strip().lower()
        if normalized_model == "":
            return False

        direct_match = re.match(r"^claude-(sonnet|opus)-(.+)$", normalized_model)
        if direct_match is not None:
            version_parts = direct_match.group(2).split("-")
            if len(version_parts) == 0 or not version_parts[0].isdigit():
                return False

            major = int(version_parts[0])
            minor = 0
            if (
                len(version_parts) > 1
                and version_parts[1].isdigit()
                and len(version_parts[1]) <= 2
            ):
                minor = int(version_parts[1])
            return (major, minor) > (4, 5)

        legacy_match = re.match(
            r"^claude-(\d+)-(\d+)-(sonnet|opus)(?:-|$)",
            normalized_model,
        )
        if legacy_match is not None:
            major = int(legacy_match.group(1))
            minor = int(legacy_match.group(2))
            return (major, minor) > (4, 5)

        return False

    def _add_auto_compaction_context_management(
        self, *, request: dict[str, Any]
    ) -> None:
        if self._context_management_mode != "auto":
            return
        model = request.get("model")
        if not isinstance(model, str) or not self._supports_auto_compaction_model(
            model=model
        ):
            return

        compaction_edit = self._build_compaction_edit()
        context_management = request.get("context_management")

        if context_management is None:
            request["context_management"] = {"edits": [compaction_edit]}
            return
        if not isinstance(context_management, dict):
            raise ValueError("context_management must be an object")

        edits_value = context_management.get("edits")
        if edits_value is None:
            edits: list[Any] = []
        else:
            edits = [*edits_value]

        has_compaction_edit = False
        normalized_edits = list[Any]()
        for edit in edits:
            if isinstance(edit, dict) and edit.get("type") == "compact_20260112":
                has_compaction_edit = True
            normalized_edits.append(edit)
        if not has_compaction_edit:
            normalized_edits.append(compaction_edit)

        request["context_management"] = {
            **context_management,
            "edits": normalized_edits,
        }

    def _is_tool_schema_grammar_complexity_error(self, *, error: Exception) -> bool:
        message = str(error).lower()
        return (
            "compiled grammar is too large" in message
            or "reduce the number of strict tools" in message
        )

    def _store_usage(
        self, *, context: AgentSessionContext, response: dict[str, Any], model: str
    ) -> None:
        usage = normalize_anthropic_usage(response.get("usage"))
        if usage is not None:
            context.metadata["last_response_usage"] = usage
            context.metadata["last_response_model"] = model

            flattened_usage = preprocess_anthropic_usage(model=model, usage=usage)
            if flattened_usage is not None:
                add_usage_metrics(totals=context.usage, usage=flattened_usage)
        context_management = response.get("context_management")
        if isinstance(context_management, dict):
            context.metadata["last_context_management"] = context_management

    async def _stream_message(
        self,
        *,
        client: Any,
        request: dict,
        event_handler: Callable[[dict], None],
    ) -> Any:
        """Stream text deltas and return the final message.

        Uses the official Anthropic SDK streaming helper:

        ```py
        async with client.messages.stream(...) as stream:
            async for text in stream.text_stream:
                ...
        message = await stream.get_final_message()
        ```
        """

        api = self._messages_api(client=client, request=request)
        stream_mgr = api.stream(**request)

        async with stream_mgr as stream:
            async for event in stream:
                event_handler({"type": event.type, "event": _as_jsonable(event)})

        final_message = await stream.get_final_message()
        event_handler(
            {"type": "message.completed", "message": _as_jsonable(final_message)}
        )
        return final_message

    def _split_toolkits(
        self, *, toolkits: list[Toolkit]
    ) -> tuple[list[Toolkit], list[BaseTool]]:
        """Split toolkits into executable tools and request middleware tools."""

        executable_toolkits: list[Toolkit] = []
        middleware: list[BaseTool] = []

        for toolkit in toolkits:
            executable_tools: list[FunctionTool] = []

            for t in toolkit.tools:
                if isinstance(t, MCPConnectorTool):
                    middleware.append(t)
                elif isinstance(t, FunctionTool):
                    executable_tools.append(t)
                elif isinstance(t, BaseTool):
                    if hasattr(t, "apply") and callable(getattr(t, "apply")):
                        middleware.append(t)
                    # Non-executable tool types are ignored.
                    continue
                else:
                    raise RoomException(f"unsupported tool type {type(t)}")

            if executable_tools:
                executable_toolkits.append(
                    Toolkit(
                        name=toolkit.name,
                        title=getattr(toolkit, "title", None),
                        description=getattr(toolkit, "description", None),
                        thumbnail_url=getattr(toolkit, "thumbnail_url", None),
                        rules=getattr(toolkit, "rules", []),
                        tools=executable_tools,
                    )
                )

        return executable_toolkits, middleware

    def _apply_request_middleware(
        self, *, request: dict, middleware: list[BaseTool]
    ) -> dict:
        headers = request.get("extra_headers") or {}
        for m in middleware:
            apply = getattr(m, "apply", None)
            if callable(apply):
                try:
                    apply(request=request, headers=headers)
                except TypeError:
                    apply(request=request)
        request["extra_headers"] = headers or None
        return request

    async def next(
        self,
        *,
        context: AgentSessionContext,
        room: RoomClient,
        toolkits: list[Toolkit],
        output_schema: Optional[dict] = None,
        event_handler: Optional[Callable[[dict], None]] = None,
        steering_callback: SteeringCallback | None = None,
        model: Optional[str] = None,
        on_behalf_of: Optional[RemoteParticipant] = None,
        options: Optional[dict] = None,
    ) -> Any:
        del options

        if model is None:
            model = self.default_model()

        context.turn_count += 1

        tool_adapter = AnthropicMessagesToolResponseAdapter()

        client = self.get_anthropic_client(room=room)

        validation_attempts = 0
        pause_turn_continuations = 0
        context_messages_snapshot = copy.deepcopy(context.messages)
        context_metadata_snapshot = copy.deepcopy(context.metadata)
        iteration_committed = False

        try:

            async def apply_steering() -> bool:
                if steering_callback is None:
                    return False

                message_count_before = len(context.messages)
                steered = await steering_callback()
                if steered:
                    if len(context.messages) > message_count_before:
                        trailing_messages = context.messages[message_count_before:]
                        del context.messages[message_count_before:]
                    else:
                        trailing_messages = []
                    self.on_turn_steer(context=context, interrupted=False)
                    context.messages.extend(trailing_messages)
                return steered

            while True:
                context_messages_snapshot = copy.deepcopy(context.messages)
                context_metadata_snapshot = copy.deepcopy(context.metadata)
                iteration_committed = False
                executable_toolkits, middleware = self._split_toolkits(
                    toolkits=toolkits
                )
                tool_bundle = MessagesToolBundle(
                    toolkits=executable_toolkits,
                    tool_calling_state=self._tool_calling_state,
                )
                self._set_function_tool_name_resolver(
                    event_handler=event_handler,
                    resolver=tool_bundle.resolve_function_tool_name,
                )

                messages, system = self._convert_messages(context=context)

                extra_headers = {}
                if on_behalf_of is not None:
                    extra_headers["Meshagent-On-Behalf-Of"] = (
                        on_behalf_of.get_attribute("name")
                    )

                message_options = dict(self._message_options or {})

                tools_list: list[dict] = tool_bundle.to_json() or []
                extra_tools = message_options.pop("tools", None)
                if isinstance(extra_tools, list):
                    tools_list.extend(extra_tools)

                request = {
                    "model": model,
                    "max_tokens": self._resolved_max_tokens(model=model),
                    "messages": messages,
                    "system": system,
                    "tools": tools_list,
                    "extra_headers": extra_headers or None,
                    **message_options,
                }

                if output_schema is not None:
                    request["output_config"] = {
                        "format": {
                            "type": "json_schema",
                            "schema": _transform_strict_anthropic_schema(
                                ensure_strict_json_schema(output_schema)
                            ),
                        }
                    }

                request = self._apply_request_middleware(
                    request=request,
                    middleware=middleware,
                )
                self._add_auto_compaction_context_management(request=request)
                self._ensure_context_management_betas(request=request)

                # Normalize empty lists to None for Anthropic.
                if (
                    isinstance(request.get("tools"), list)
                    and len(request["tools"]) == 0
                ):
                    request["tools"] = None
                if (
                    isinstance(request.get("mcp_servers"), list)
                    and len(request["mcp_servers"]) == 0
                ):
                    request["mcp_servers"] = None
                if (
                    isinstance(request.get("betas"), list)
                    and len(request["betas"]) == 0
                ):
                    request["betas"] = None

                # remove None fields
                request = {k: v for k, v in request.items() if v is not None}

                logger.info("requesting response from anthropic with model: %s", model)

                try:
                    if event_handler is not None:
                        final_message = await self._stream_message(
                            client=client,
                            request=request,
                            event_handler=event_handler,
                        )
                        response_dict = _as_jsonable(final_message)
                    else:
                        response = await self._create_with_optional_headers(
                            client=client,
                            request=request,
                        )
                        response_dict = _as_jsonable(response)
                except APIStatusError as e:
                    if (
                        self._tool_calling_state.mode == "adaptive"
                        and tool_bundle.uses_strict_tools()
                        and self._is_tool_schema_grammar_complexity_error(error=e)
                    ):
                        logger.warning(
                            "anthropic strict tool grammar is too large; disabling strict tool calling and retrying"
                        )
                        tool_bundle.disable_all_strict()
                        continue
                    raise

                self._store_usage(context=context, response=response_dict, model=model)

                content_blocks = []
                raw_content = response_dict.get("content")
                if isinstance(raw_content, list):
                    content_blocks = raw_content

                tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]
                stop_reason = response_dict.get("stop_reason")
                if stop_reason == "max_tokens" and len(tool_uses) > 0:
                    if event_handler is not None:
                        for tool_use in tool_uses:
                            event_handler(
                                {
                                    "type": "meshagent.handler.done",
                                    "item_id": tool_use.get("id"),
                                    "error": (
                                        "Anthropic response hit max_tokens before completing tool calls. "
                                        "Increase max_tokens and retry."
                                    ),
                                }
                            )
                    raise RoomException(
                        "Anthropic response hit max_tokens before completing tool calls. "
                        "Increase max_tokens and retry."
                    )

                assistant_message = {"role": "assistant", "content": content_blocks}

                if stop_reason == "pause_turn":
                    context.messages.append(assistant_message)
                    iteration_committed = True
                    await apply_steering()
                    pause_turn_continuations += 1
                    if pause_turn_continuations > _MAX_PAUSE_TURN_CONTINUATIONS:
                        raise RoomException(
                            "Anthropic response paused too many times while processing server tools."
                        )
                    continue

                if tool_uses:
                    completed_results: list[list[dict]] = []
                    steering_applied = False

                    async def do_tool(tool_use: dict) -> list[dict]:
                        tool_context = ToolContext(
                            room=room,
                            caller=room.local_participant,
                            on_behalf_of=on_behalf_of,
                            caller_context={"chat": context.to_json()},
                            validation_mode=tool_bundle.validation_mode_for_tool_use(
                                tool_use=tool_use
                            ),
                        )
                        try:
                            if event_handler is not None:
                                event_handler(
                                    {
                                        "type": "meshagent.handler.added",
                                        "item": {
                                            "type": "function_call",
                                            "id": tool_use.get("id"),
                                            "call_id": tool_use.get("id"),
                                            "name": tool_use.get("name"),
                                            "arguments": json.dumps(
                                                tool_use.get("input") or {}
                                            ),
                                        },
                                    }
                                )
                            tool_response = await tool_bundle.execute(
                                context=tool_context,
                                tool_use=tool_use,
                            )
                            if isinstance(tool_response, AsyncIterable):
                                tool_response = await _consume_streaming_tool_result(
                                    stream=tool_response,
                                    event_handler=event_handler,
                                )
                            else:
                                tool_response = ensure_content(tool_response)
                            if event_handler is not None:
                                handler_result = None
                                if isinstance(tool_response, JsonContent):
                                    handler_result = tool_response.json
                                elif isinstance(tool_response, TextContent):
                                    handler_result = tool_response.text
                                event_handler(
                                    {
                                        "type": "meshagent.handler.done",
                                        "item_id": tool_use.get("id"),
                                        "result": handler_result,
                                    }
                                )
                            return await tool_adapter.create_messages(
                                context=context,
                                tool_call=tool_use,
                                room=room,
                                response=tool_response,
                            )
                        except asyncio.CancelledError:
                            if event_handler is not None:
                                event_handler(
                                    {
                                        "type": "meshagent.handler.done",
                                        "item_id": tool_use.get("id"),
                                        "error": "cancelled",
                                    }
                                )
                            raise
                        except Exception as ex:
                            tool_bundle.record_tool_failure(tool_use=tool_use)
                            logger.error(
                                f"error in tool call {json.dumps(tool_use)}:",
                                exc_info=ex,
                            )
                            if event_handler is not None:
                                event_handler(
                                    {
                                        "type": "meshagent.handler.done",
                                        "item_id": tool_use.get("id"),
                                        "error": f"{ex}",
                                    }
                                )
                            return [
                                _tool_result_message(
                                    tool_use_id=tool_use.get("id"),
                                    content=[_text_block(f"Error: {ex}")],
                                )
                            ]

                    pending_tasks: dict[asyncio.Task[list[dict]], dict] = {
                        asyncio.create_task(do_tool(tool_use)): tool_use
                        for tool_use in tool_uses
                    }

                    while pending_tasks:
                        done, _ = await asyncio.wait(
                            list(pending_tasks.keys()),
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for task in done:
                            tool_use = pending_tasks.pop(task)
                            del tool_use
                            completed_results.append(await task)

                        if steering_callback is None:
                            continue

                        provisional_tool_result_blocks, _ = (
                            _split_tool_execution_messages(completed_results)
                        )
                        if pending_tasks:
                            for tool_use in pending_tasks.values():
                                provisional_tool_result_blocks.extend(
                                    _cancelled_tool_result_message(tool_use=tool_use)[
                                        "content"
                                    ]
                                )

                        context.messages.append(assistant_message)
                        tool_result_message_index: int | None = None
                        if provisional_tool_result_blocks:
                            tool_result_message_index = len(context.messages)
                            context.messages.append(
                                {
                                    "role": "user",
                                    "content": provisional_tool_result_blocks,
                                }
                            )

                        if await apply_steering():
                            if pending_tasks:
                                tasks_to_cancel = list(pending_tasks.keys())
                                cancelled_tool_uses = [
                                    pending_tasks[task] for task in tasks_to_cancel
                                ]
                                for task in tasks_to_cancel:
                                    task.cancel()
                                cancelled_results = await asyncio.gather(
                                    *tasks_to_cancel,
                                    return_exceptions=True,
                                )
                                for tool_use, cancelled_result in zip(
                                    cancelled_tool_uses, cancelled_results
                                ):
                                    if isinstance(cancelled_result, list):
                                        completed_results.append(cancelled_result)
                                    else:
                                        completed_results.append(
                                            [
                                                _cancelled_tool_result_message(
                                                    tool_use=tool_use
                                                )
                                            ]
                                        )
                                final_tool_result_blocks, _ = (
                                    _split_tool_execution_messages(completed_results)
                                )
                                if (
                                    final_tool_result_blocks
                                    and tool_result_message_index is not None
                                    and tool_result_message_index
                                    < len(context.messages)
                                ):
                                    context.messages[tool_result_message_index] = {
                                        "role": "user",
                                        "content": final_tool_result_blocks,
                                    }
                                pending_tasks.clear()
                            iteration_committed = True
                            steering_applied = True
                            break

                        context.messages[:] = copy.deepcopy(context_messages_snapshot)

                    if steering_applied:
                        continue

                    tool_result_blocks, trailing_messages = (
                        _split_tool_execution_messages(completed_results)
                    )

                    context.messages.append(assistant_message)
                    if tool_result_blocks:
                        context.messages.append(
                            {"role": "user", "content": tool_result_blocks}
                        )
                    iteration_committed = True

                    if await apply_steering():
                        continue

                    for msg in trailing_messages:
                        context.messages.append(msg)
                    continue

                # no tool calls; return final content
                context.messages.append(assistant_message)
                iteration_committed = True
                text = "".join(
                    [
                        b.get("text", "")
                        for b in content_blocks
                        if b.get("type") == "text"
                    ]
                )

                stop_reason_error = (
                    self._stop_reason_error_message(stop_reason=stop_reason)
                    if isinstance(stop_reason, str)
                    else None
                )
                if stop_reason_error is not None:
                    raise RoomException(stop_reason_error)

                if output_schema is None:
                    return text

                # Schema-mode: parse and validate JSON.
                validation_attempts += 1
                try:
                    parsed = json.loads(text)
                    return parsed
                except Exception as e:
                    if validation_attempts >= 3:
                        raise RoomException(
                            f"Invalid JSON response from Anthropic: {e}"
                        )
                    context.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous response did not match the required JSON schema. "
                                f"Error: {e}. Please try again and return only valid JSON."
                            ),
                        }
                    )

        except asyncio.CancelledError:
            if not iteration_committed:
                context.messages.clear()
                context.messages.extend(copy.deepcopy(context_messages_snapshot))
                context.metadata.clear()
                context.metadata.update(copy.deepcopy(context_metadata_snapshot))
            raise
        except APIStatusError as e:
            raise RoomException(f"Error from Anthropic: {e}")
        finally:
            self._set_function_tool_name_resolver(
                event_handler=event_handler,
                resolver=None,
            )

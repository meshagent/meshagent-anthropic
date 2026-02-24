from __future__ import annotations

from meshagent.agents.agent import AgentSessionContext
from meshagent.api import RoomClient, RoomException, RemoteParticipant
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
from meshagent.agents.adapter import ToolResponseAdapter, LLMAdapter

import json
from typing import Any, Optional, Callable
from collections.abc import AsyncIterable
import os
import logging
import re
import asyncio
import base64
from html_to_markdown import convert

from meshagent.anthropic.proxy import get_client, get_logging_httpx_client
from meshagent.anthropic.mcp import MCPTool as MCPConnectorTool

try:
    from anthropic import APIStatusError
except Exception:  # pragma: no cover
    APIStatusError = Exception  # type: ignore

logger = logging.getLogger("anthropic_agent")


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


class MessagesToolBundle:
    def __init__(self, toolkits: list[Toolkit]):
        self._executors: dict[str, Toolkit] = {}
        self._safe_names: dict[str, str] = {}
        self._tools_by_safe_name: dict[str, FunctionTool] = {}

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

                tools.append(
                    {
                        "name": safe_name,
                        "description": v.description,
                        "input_schema": schema,
                    }
                )

        self._tools = tools or None

    def to_json(self) -> list[dict] | None:
        return None if self._tools is None else self._tools.copy()

    def get_tool(self, safe_name: str) -> FunctionTool | None:
        return self._tools_by_safe_name.get(safe_name)

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
        max_tokens: int = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192")),
        client: Optional[Any] = None,
        message_options: Optional[dict] = None,
        provider: str = "anthropic",
        log_requests: bool = False,
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._client = client
        self._message_options = message_options or {}
        self._provider = provider
        self._log_requests = log_requests

    def default_model(self) -> str:
        return self._model

    def create_session(self) -> AgentSessionContext:
        return AnthropicMessagesChatContext(system_role=None)

    def get_anthropic_client(self, *, room: RoomClient) -> Any:
        if self._client is not None:
            return self._client
        http_client = get_logging_httpx_client() if self._log_requests else None
        return get_client(room=room, http_client=http_client)

    def _convert_messages(
        self, *, context: AgentSessionContext
    ) -> tuple[list[dict], Optional[str]]:
        system = context.get_system_instructions()

        def as_blocks(role: str, content: Any) -> dict:
            if isinstance(content, str):
                return {"role": role, "content": [_text_block(content)]}
            if isinstance(content, list):
                return {"role": role, "content": content}
            return {"role": role, "content": [_text_block(str(content))]}

        messages: list[dict] = []
        pending_tool_use_ids: set[str] = set()

        for m in context.messages:
            role = m.get("role")
            if role not in {"user", "assistant"}:
                continue

            msg = as_blocks(role, m.get("content"))

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

        return messages, system

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
        model: Optional[str] = None,
        on_behalf_of: Optional[RemoteParticipant] = None,
    ) -> Any:
        if model is None:
            model = self.default_model()

        tool_adapter = AnthropicMessagesToolResponseAdapter()

        client = self.get_anthropic_client(room=room)

        validation_attempts = 0

        try:
            while True:
                executable_toolkits, middleware = self._split_toolkits(
                    toolkits=toolkits
                )
                tool_bundle = MessagesToolBundle(toolkits=executable_toolkits)

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
                    "max_tokens": self._max_tokens,
                    "messages": messages,
                    "system": system,
                    "tools": tools_list,
                    "extra_headers": extra_headers or None,
                    **message_options,
                }

                if output_schema is not None:
                    request["output_config"] = {
                        "format": {"type": "json_schema", "schema": output_schema}
                    }

                request = self._apply_request_middleware(
                    request=request,
                    middleware=middleware,
                )

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

                content_blocks = []
                raw_content = response_dict.get("content")
                if isinstance(raw_content, list):
                    content_blocks = raw_content

                tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]

                # Keep the assistant message in context.
                assistant_message = {"role": "assistant", "content": content_blocks}
                context.messages.append(assistant_message)

                if tool_uses:
                    tasks = []

                    async def do_tool(tool_use: dict) -> list[dict]:
                        tool_context = ToolContext(
                            room=room,
                            caller=room.local_participant,
                            on_behalf_of=on_behalf_of,
                            caller_context={"chat": context.to_json()},
                        )
                        try:
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
                            return await tool_adapter.create_messages(
                                context=context,
                                tool_call=tool_use,
                                room=room,
                                response=tool_response,
                            )
                        except Exception as ex:
                            logger.error(
                                f"error in tool call {json.dumps(tool_use)}:",
                                exc_info=ex,
                            )
                            tool_result_content = [_text_block(f"Error: {ex}")]
                            message = {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": tool_use.get("id"),
                                        "content": tool_result_content,
                                    }
                                ],
                            }
                            return [message]

                    for tool_use in tool_uses:
                        tasks.append(asyncio.create_task(do_tool(tool_use)))

                    results = await asyncio.gather(*tasks)

                    # Anthropic requires tool_result blocks for *all* tool_use ids to appear in the
                    # *immediately next* user message after the assistant tool_use message.
                    tool_result_blocks: list[dict] = []
                    trailing_messages: list[dict] = []

                    for msgs in results:
                        for msg in msgs:
                            if (
                                isinstance(msg, dict)
                                and msg.get("role") == "user"
                                and isinstance(msg.get("content"), list)
                                and all(
                                    isinstance(b, dict)
                                    and b.get("type") == "tool_result"
                                    for b in msg["content"]
                                )
                            ):
                                tool_result_blocks.extend(msg["content"])
                            else:
                                trailing_messages.append(msg)

                    if tool_result_blocks:
                        context.messages.append(
                            {"role": "user", "content": tool_result_blocks}
                        )

                    for msg in trailing_messages:
                        context.messages.append(msg)

                    continue

                # no tool calls; return final content
                text = "".join(
                    [
                        b.get("text", "")
                        for b in content_blocks
                        if b.get("type") == "text"
                    ]
                )

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

        except APIStatusError as e:
            raise RoomException(f"Error from Anthropic: {e}")

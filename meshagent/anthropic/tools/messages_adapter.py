from __future__ import annotations

from meshagent.agents.agent import AgentChatContext
from meshagent.api import RoomClient, RoomException, RemoteParticipant
from meshagent.tools import Toolkit, ToolContext, Tool
from meshagent.api.messaging import (
    Response,
    LinkResponse,
    FileResponse,
    JsonResponse,
    TextResponse,
    EmptyResponse,
    RawOutputs,
    ensure_response,
)
from meshagent.agents.adapter import ToolResponseAdapter, LLMAdapter

import json
from typing import Any, Optional, Callable
import os
import logging
import re
import asyncio

from meshagent.anthropic.proxy import get_client, get_logging_httpx_client

try:
    from anthropic import APIStatusError
except Exception:  # pragma: no cover
    APIStatusError = Exception  # type: ignore

logger = logging.getLogger("anthropic_agent")


def _replace_non_matching(text: str, allowed_chars: str, replacement: str) -> str:
    pattern = rf"[^{allowed_chars}]"
    return re.sub(pattern, replacement, text)


def safe_tool_name(name: str) -> str:
    return _replace_non_matching(name, "a-zA-Z0-9_-", "_")


def _as_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return obj
    return obj.model_dump(mode="json")


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


class MessagesToolBundle:
    def __init__(self, toolkits: list[Toolkit]):
        self._executors: dict[str, Toolkit] = {}
        self._safe_names: dict[str, str] = {}
        self._tools_by_safe_name: dict[str, Tool] = {}

        tools: list[dict] = []

        for toolkit in toolkits:
            for v in toolkit.tools:
                if not isinstance(v, Tool):
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

    def get_tool(self, safe_name: str) -> Tool | None:
        return self._tools_by_safe_name.get(safe_name)

    async def execute(self, *, context: ToolContext, tool_use: dict) -> Response:
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
        result = await proxy.execute(context=context, name=name, arguments=arguments)
        return ensure_response(result)


class AnthropicMessagesToolResponseAdapter(ToolResponseAdapter):
    async def to_plain_text(self, *, room: RoomClient, response: Response) -> str:
        if isinstance(response, LinkResponse):
            return json.dumps({"name": response.name, "url": response.url})
        if isinstance(response, JsonResponse):
            return json.dumps(response.json)
        if isinstance(response, TextResponse):
            return response.text
        if isinstance(response, FileResponse):
            return response.name
        if isinstance(response, EmptyResponse):
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
        context: AgentChatContext,
        tool_call: Any,
        room: RoomClient,
        response: Response,
    ) -> list:
        tool_use = tool_call if isinstance(tool_call, dict) else _as_jsonable(tool_call)
        tool_use_id = tool_use.get("id")
        if tool_use_id is None:
            raise RoomException("anthropic tool_use block was missing an id")

        if isinstance(response, RawOutputs):
            # Allow advanced tools to return pre-built Anthropic blocks.
            return [{"role": "user", "content": response.outputs}]

        output = await self.to_plain_text(room=room, response=response)

        message = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [_text_block(output)],
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
        max_tokens: int = int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024")),
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

    def create_chat_context(self) -> AgentChatContext:
        return AgentChatContext(system_role=None)

    def get_anthropic_client(self, *, room: RoomClient) -> Any:
        if self._client is not None:
            return self._client
        http_client = get_logging_httpx_client() if self._log_requests else None
        return get_client(room=room, http_client=http_client)

    def _convert_messages(
        self, *, context: AgentChatContext
    ) -> tuple[list[dict], Optional[str]]:
        system = context.get_system_instructions()

        messages: list[dict] = []
        for m in context.messages:
            if m.get("role") in {"user", "assistant"}:
                content = m.get("content")
                if isinstance(content, str):
                    messages.append(
                        {"role": m["role"], "content": [_text_block(content)]}
                    )
                elif isinstance(content, list):
                    # Allow passing through OpenAI-style image/file blocks if already present.
                    # Tool adapters will also insert Anthropic blocks here.
                    messages.append({"role": m["role"], "content": content})
                else:
                    messages.append(
                        {"role": m["role"], "content": [_text_block(str(content))]}
                    )

        return messages, system

    async def _create_with_optional_headers(self, client: Any, **kwargs) -> Any:
        try:
            return await client.messages.create(**kwargs)
        except TypeError:
            kwargs.pop("extra_headers", None)
            return await client.messages.create(**kwargs)

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

        stream = client.messages.stream(**request)

        async with stream:
            async for event in stream:
                event_handler({"type": event.type, "event": _as_jsonable(event)})

        final_message = await stream.get_final_message()
        event_handler(
            {"type": "message.completed", "message": _as_jsonable(final_message)}
        )
        return final_message

    async def next(
        self,
        *,
        context: AgentChatContext,
        room: RoomClient,
        toolkits: list[Toolkit],
        tool_adapter: Optional[ToolResponseAdapter] = None,
        output_schema: Optional[dict] = None,
        event_handler: Optional[Callable[[dict], None]] = None,
        model: Optional[str] = None,
        on_behalf_of: Optional[RemoteParticipant] = None,
    ) -> Any:
        if model is None:
            model = self.default_model()

        if tool_adapter is None:
            tool_adapter = AnthropicMessagesToolResponseAdapter()

        client = self.get_anthropic_client(room=room)

        validation_attempts = 0

        try:
            while True:
                tool_bundle = MessagesToolBundle(toolkits=toolkits)
                tools = tool_bundle.to_json()

                messages, system = self._convert_messages(context=context)

                if output_schema is not None:
                    schema_hint = json.dumps(output_schema)
                    schema_system = (
                        "Return ONLY valid JSON that matches this JSON Schema. "
                        "Do not wrap in markdown. Schema: " + schema_hint
                    )
                    system = (
                        (system + "\n" + schema_system) if system else schema_system
                    )

                extra_headers = {}
                if on_behalf_of is not None:
                    extra_headers["Meshagent-On-Behalf-Of"] = (
                        on_behalf_of.get_attribute("name")
                    )

                request = {
                    "model": model,
                    "max_tokens": self._max_tokens,
                    "messages": messages,
                    "system": system,
                    "tools": tools,
                    "extra_headers": extra_headers or None,
                    **(self._message_options or {}),
                }

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
                        client, **request
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
                        tool_response = await tool_bundle.execute(
                            context=tool_context,
                            tool_use=tool_use,
                        )
                        return await tool_adapter.create_messages(
                            context=context,
                            tool_call=tool_use,
                            room=room,
                            response=tool_response,
                        )

                    for tool_use in tool_uses:
                        tasks.append(asyncio.create_task(do_tool(tool_use)))

                    results = await asyncio.gather(*tasks)
                    for msgs in results:
                        for msg in msgs:
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
                    self.validate(response=parsed, output_schema=output_schema)
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

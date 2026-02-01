from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from meshagent.api import RoomClient, RemoteParticipant
from meshagent.agents.agent import AgentChatContext
from meshagent.tools import Toolkit

from .messages_adapter import AnthropicMessagesAdapter


@dataclass
class _OutputBlockState:
    kind: str  # "message" | "function_call" | "reasoning"
    item_id: str
    output_index: int
    content_index: int
    name: Optional[str] = None
    call_id: Optional[str] = None
    text: str = ""
    arguments: str = ""


class AnthropicOpenAIResponsesStreamAdapter(AnthropicMessagesAdapter):
    """Anthropic adapter that emits OpenAI Responses-style stream events.

    This is useful when you have downstream consumers that already understand the
    OpenAI Responses streaming event schema (e.g. UI code), but want to run the
    underlying inference on Anthropic.

    Notes:
    - This adapter only affects the *streaming* event shape.
    - Tool execution still uses MeshAgent toolkits and the Anthropic tool loop.
    """

    async def _stream_message(
        self,
        *,
        client: Any,
        request: dict,
        event_handler: Callable[[dict], None],
    ) -> Any:
        seq = 0
        response_id: Optional[str] = None
        response_model: str = str(request.get("model"))

        output: list[dict] = []
        blocks: dict[int, _OutputBlockState] = {}

        created_at = int(time.time())

        def emit(payload: dict) -> None:
            nonlocal seq
            if "sequence_number" not in payload:
                payload["sequence_number"] = seq
            seq += 1
            event_handler(payload)

        def output_message_item(*, item_id: str, status: str, text: str) -> dict:
            return {
                "type": "message",
                "id": item_id,
                "role": "assistant",
                "status": status,
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                        "logprobs": None,
                    }
                ],
            }

        def output_function_call_item(
            *,
            item_id: str,
            call_id: str,
            name: str,
            status: Optional[str],
            arguments: str,
        ) -> dict:
            return {
                "type": "function_call",
                "id": item_id,
                "call_id": call_id,
                "name": name,
                "arguments": arguments,
                "status": status,
            }

        def output_reasoning_item(*, item_id: str, status: str, text: str) -> dict:
            return {
                "type": "reasoning",
                "id": item_id,
                "status": status,
                "summary": [],
                "content": [{"type": "reasoning_text", "text": text}],
                "encrypted_content": None,
            }

        response_obj: dict = {
            "id": None,
            "object": "response",
            "created_at": created_at,
            "model": response_model,
            "output": output,
            "status": "in_progress",
            "error": None,
            "usage": None,
        }

        async with client.messages.stream(**request) as stream:
            async for event in stream:
                data = event.model_dump(
                    mode="json",
                    exclude_none=True,
                    exclude_unset=True,
                )
                etype = data.get("type")

                if etype == "message_start":
                    message = data.get("message") or {}
                    response_id = message.get("id") or response_id
                    response_obj["id"] = response_id
                    if message.get("model") is not None:
                        response_obj["model"] = message.get("model")

                    emit({"type": "response.created", "response": dict(response_obj)})

                elif etype == "content_block_start":
                    idx = int(data.get("index"))
                    block = data.get("content_block") or {}
                    btype = block.get("type")

                    output_index = len(output)
                    base_item_id = response_id or "anthropic"
                    item_id = f"{base_item_id}_out_{output_index}"

                    if btype == "text":
                        item = output_message_item(
                            item_id=item_id, status="in_progress", text=""
                        )
                        output.append(item)
                        blocks[idx] = _OutputBlockState(
                            kind="message",
                            item_id=item_id,
                            output_index=output_index,
                            content_index=0,
                        )

                    elif btype == "tool_use":
                        call_id = str(block.get("id"))
                        name = str(block.get("name"))
                        item = output_function_call_item(
                            item_id=item_id,
                            call_id=call_id,
                            name=name,
                            status="in_progress",
                            arguments="",
                        )
                        output.append(item)
                        blocks[idx] = _OutputBlockState(
                            kind="function_call",
                            item_id=item_id,
                            output_index=output_index,
                            content_index=0,
                            name=name,
                            call_id=call_id,
                        )

                    elif btype == "thinking":
                        item = output_reasoning_item(
                            item_id=item_id, status="in_progress", text=""
                        )
                        output.append(item)
                        blocks[idx] = _OutputBlockState(
                            kind="reasoning",
                            item_id=item_id,
                            output_index=output_index,
                            content_index=0,
                        )

                    else:
                        # Unknown block type: ignore for OpenAI compatibility.
                        continue

                    emit(
                        {
                            "type": "response.output_item.added",
                            "output_index": output_index,
                            "item": output[output_index],
                        }
                    )

                    # OpenAI-style content part events (text + reasoning content).
                    state = blocks.get(idx)
                    if state is not None and state.kind in {"message", "reasoning"}:
                        emit(
                            {
                                "type": "response.content_part.added",
                                "output_index": state.output_index,
                                "item_id": state.item_id,
                                "content_index": state.content_index,
                                "part": output[state.output_index]["content"][
                                    state.content_index
                                ],
                            }
                        )

                elif etype == "content_block_delta":
                    idx = int(data.get("index"))
                    state = blocks.get(idx)
                    if state is None:
                        continue

                    delta = data.get("delta") or {}
                    dtype = delta.get("type")

                    if dtype == "text_delta" and state.kind == "message":
                        piece = str(delta.get("text") or "")
                        state.text += piece
                        output[state.output_index]["content"][0]["text"] = state.text

                        emit(
                            {
                                "type": "response.output_text.delta",
                                "output_index": state.output_index,
                                "item_id": state.item_id,
                                "content_index": state.content_index,
                                "delta": piece,
                                "logprobs": None,
                            }
                        )

                    elif dtype == "input_json_delta" and state.kind == "function_call":
                        piece = str(delta.get("partial_json") or "")
                        state.arguments += piece
                        output[state.output_index]["arguments"] = state.arguments

                        emit(
                            {
                                "type": "response.function_call_arguments.delta",
                                "output_index": state.output_index,
                                "item_id": state.item_id,
                                "delta": piece,
                            }
                        )

                    elif dtype == "thinking_delta" and state.kind == "reasoning":
                        piece = str(delta.get("thinking") or "")
                        state.text += piece
                        output[state.output_index]["content"][0]["text"] = state.text

                        emit(
                            {
                                "type": "response.reasoning_text.delta",
                                "output_index": state.output_index,
                                "item_id": state.item_id,
                                "content_index": state.content_index,
                                "delta": piece,
                            }
                        )

                    else:
                        # Ignore signature_delta and unknown deltas.
                        continue

                elif etype == "content_block_stop":
                    idx = int(data.get("index"))
                    state = blocks.get(idx)
                    if state is None:
                        continue

                    if state.kind == "message":
                        emit(
                            {
                                "type": "response.output_text.done",
                                "output_index": state.output_index,
                                "item_id": state.item_id,
                                "content_index": state.content_index,
                                "text": state.text,
                                "logprobs": None,
                            }
                        )
                        output[state.output_index] = output_message_item(
                            item_id=state.item_id, status="completed", text=state.text
                        )
                        emit(
                            {
                                "type": "response.content_part.done",
                                "output_index": state.output_index,
                                "item_id": state.item_id,
                                "content_index": state.content_index,
                                "part": output[state.output_index]["content"][
                                    state.content_index
                                ],
                            }
                        )

                    elif state.kind == "function_call":
                        emit(
                            {
                                "type": "response.function_call_arguments.done",
                                "output_index": state.output_index,
                                "item_id": state.item_id,
                                "name": state.name,
                                "arguments": state.arguments,
                            }
                        )
                        output[state.output_index] = output_function_call_item(
                            item_id=state.item_id,
                            call_id=str(state.call_id),
                            name=str(state.name),
                            status="completed",
                            arguments=state.arguments,
                        )

                    elif state.kind == "reasoning":
                        emit(
                            {
                                "type": "response.reasoning_text.done",
                                "output_index": state.output_index,
                                "item_id": state.item_id,
                                "content_index": state.content_index,
                                "text": state.text,
                            }
                        )
                        output[state.output_index] = output_reasoning_item(
                            item_id=state.item_id, status="completed", text=state.text
                        )
                        emit(
                            {
                                "type": "response.content_part.done",
                                "output_index": state.output_index,
                                "item_id": state.item_id,
                                "content_index": state.content_index,
                                "part": output[state.output_index]["content"][
                                    state.content_index
                                ],
                            }
                        )

                    emit(
                        {
                            "type": "response.output_item.done",
                            "output_index": state.output_index,
                            "item": output[state.output_index],
                        }
                    )

                elif etype == "message_stop":
                    # Defer `response.completed` until after we have final usage.
                    pass

                else:
                    # ping, message_delta, etc.
                    continue

        final_message = await stream.get_final_message()
        final_dict = final_message.model_dump(
            mode="json",
            exclude_none=True,
            exclude_unset=True,
        )
        usage = final_dict.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            response_obj["usage"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_tokens_details": None,
                "output_tokens_details": None,
            }

        response_obj["status"] = "completed"
        response_obj["output"] = output

        emit({"type": "response.completed", "response": dict(response_obj)})
        return final_message

    async def next(
        self,
        *,
        context: AgentChatContext,
        room: RoomClient,
        toolkits: list[Toolkit],
        tool_adapter: Any = None,
        output_schema: Optional[dict] = None,
        event_handler: Optional[Callable[[dict], None]] = None,
        model: Optional[str] = None,
        on_behalf_of: Optional[RemoteParticipant] = None,
    ) -> Any:
        # Keep the same behavior; only streaming shape changes.
        return await super().next(
            context=context,
            room=room,
            toolkits=toolkits,
            tool_adapter=tool_adapter,
            output_schema=output_schema,
            event_handler=event_handler,
            model=model,
            on_behalf_of=on_behalf_of,
        )

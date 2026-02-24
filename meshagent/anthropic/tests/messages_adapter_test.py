import pytest

from meshagent.anthropic.messages_adapter import (
    AnthropicMessagesAdapter,
    _consume_streaming_tool_result,
)
from meshagent.agents.agent import AgentSessionContext
from meshagent.api.messaging import JsonContent, TextContent
from meshagent.tools import FunctionTool, Toolkit
from meshagent.api import RoomException


class _DummyParticipant:
    def __init__(self):
        self.id = "p1"

    def get_attribute(self, name: str):
        if name == "name":
            return "tester"
        return None


class _DummyRoom:
    def __init__(self):
        self.local_participant = _DummyParticipant()
        self.developer = self

    def log_nowait(self, **kwargs):
        del kwargs


class _AnyArgsTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="test tool",
        )

    async def execute(self, context, **kwargs):
        return {"ok": True, "args": kwargs}


class _FakeAdapter(AnthropicMessagesAdapter):
    def __init__(self, responses: list[dict]):
        super().__init__(client=object())
        self._responses = responses
        self._idx = 0

    async def _create_with_optional_headers(self, *, client, request):
        if self._idx >= len(self._responses):
            raise AssertionError("unexpected extra request")
        resp = self._responses[self._idx]
        self._idx += 1
        return resp


class _StreamingTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="streaming test tool",
        )

    async def execute(self, context, **kwargs):
        del context
        del kwargs

        async def _run():
            yield JsonContent(json={"type": "agent.event", "headline": "working"})
            yield TextContent(text="tool-final")

        return _run()


def test_convert_messages_drops_assistant_between_tool_use_and_tool_result():
    ctx = AgentSessionContext(
        system_role=None,
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "calling tool"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "tool_a",
                        "input": {},
                    },
                ],
            },
            {"role": "assistant", "content": "stray assistant message"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "ok"}],
                    }
                ],
            },
        ],
    )

    adapter = AnthropicMessagesAdapter(client=object())
    msgs, _system = adapter._convert_messages(context=ctx)

    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][1]["type"] == "tool_use"


def test_convert_messages_raises_if_tool_result_not_immediately_next():
    ctx = AgentSessionContext(
        system_role=None,
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "tool_a",
                        "input": {},
                    },
                ],
            },
            {"role": "user", "content": "not a tool_result"},
        ],
    )

    adapter = AnthropicMessagesAdapter(client=object())

    with pytest.raises(RoomException):
        adapter._convert_messages(context=ctx)


@pytest.mark.asyncio
async def test_next_batches_multiple_tool_results_into_single_user_message():
    responses = [
        {
            "content": [
                {"type": "text", "text": "calling tools"},
                {"type": "tool_use", "id": "toolu_1", "name": "tool_a", "input": {}},
                {"type": "tool_use", "id": "toolu_2", "name": "tool_b", "input": {}},
            ]
        },
        {"content": [{"type": "text", "text": "done"}]},
    ]

    adapter = _FakeAdapter(responses=responses)
    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message("run tools")

    toolkit = Toolkit(
        name="test",
        tools=[_AnyArgsTool("tool_a"), _AnyArgsTool("tool_b")],
    )

    result = await adapter.next(
        context=ctx,
        room=_DummyRoom(),
        toolkits=[toolkit],
    )

    assert result == "done"

    # Expect: user -> assistant(tool_use) -> user(tool_results batched) -> assistant(final)
    assert [m["role"] for m in ctx.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    tool_results_msg = ctx.messages[2]
    assert tool_results_msg["role"] == "user"
    assert len(tool_results_msg["content"]) == 2
    assert {b["tool_use_id"] for b in tool_results_msg["content"]} == {
        "toolu_1",
        "toolu_2",
    }


@pytest.mark.asyncio
async def test_next_uses_final_stream_item_as_tool_result() -> None:
    responses = [
        {
            "content": [
                {"type": "text", "text": "calling tool"},
                {"type": "tool_use", "id": "toolu_1", "name": "tool_a", "input": {}},
            ]
        },
        {"content": [{"type": "text", "text": "done"}]},
    ]

    adapter = _FakeAdapter(responses=responses)
    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message("run tool")

    toolkit = Toolkit(name="test", tools=[_StreamingTool("tool_a")])
    result = await adapter.next(
        context=ctx,
        room=_DummyRoom(),
        toolkits=[toolkit],
    )

    assert result == "done"
    assert [m["role"] for m in ctx.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    tool_result_content = ctx.messages[2]["content"][0]["content"][0]
    assert tool_result_content["type"] == "text"
    assert tool_result_content["text"] == "tool-final"


def test_create_chat_context_supports_images_and_files() -> None:
    adapter = AnthropicMessagesAdapter(client=object())
    context = adapter.create_session()

    assert context.supports_images is True
    assert context.supports_files is True

    image_message = context.append_image_message(mime_type="image/png", data=b"png")
    assert image_message["content"][0]["type"] == "image"

    file_message = context.append_file_message(
        filename="file.pdf",
        mime_type="application/pdf",
        data=b"%PDF-1.7",
    )
    assert file_message["content"][0]["type"] == "document"


class _ToolItemStream:
    def __init__(self, *, items: list[object]):
        self._items = items

    def __aiter__(self):
        return self._run()

    async def _run(self):
        for item in self._items:
            yield item


@pytest.mark.asyncio
async def test_consume_streaming_tool_result_emits_intermediate_json_chunk_events():
    events: list[dict] = []
    result = await _consume_streaming_tool_result(
        stream=_ToolItemStream(
            items=[
                JsonContent(json={"type": "agent.event", "headline": "working"}),
                TextContent(text="done"),
            ]
        ),
        event_handler=events.append,
    )

    assert events == [{"type": "agent.event", "headline": "working"}]
    assert isinstance(result, TextContent)
    assert result.text == "done"


@pytest.mark.asyncio
async def test_consume_streaming_tool_result_uses_final_item_as_result():
    events: list[dict] = []
    result = await _consume_streaming_tool_result(
        stream=_ToolItemStream(
            items=[
                JsonContent(json={"progress": 1}),
                JsonContent(json={"ok": True}),
            ]
        ),
        event_handler=events.append,
    )

    assert events == [{"progress": 1}]
    assert isinstance(result, JsonContent)
    assert result.json == {"ok": True}

import pytest
from pydantic import BaseModel

from meshagent.anthropic.openai_responses_stream_adapter import (
    AnthropicOpenAIResponsesStreamAdapter,
)
from meshagent.anthropic.messages_adapter import AnthropicMessagesAdapter
from meshagent.agents.agent import AgentSessionContext


class _Event(BaseModel):
    type: str
    index: int | None = None
    message: dict | None = None
    content_block: dict | None = None
    delta: dict | None = None


class _FinalMessage(BaseModel):
    id: str = "msg_1"
    usage: dict = {"input_tokens": 3, "output_tokens": 5}


class _FakeStream:
    def __init__(self, events: list[BaseModel], final: BaseModel):
        self._events = events
        self._final = final

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e

        return gen()

    async def get_final_message(self):
        return self._final


class _FakeMessages:
    def __init__(self, stream: _FakeStream):
        self._stream = stream
        self.calls: list[dict] = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return self._stream


class _FakeClient:
    def __init__(
        self,
        *,
        messages_stream: _FakeStream,
        beta_stream: _FakeStream | None = None,
    ):
        self.messages = _FakeMessages(messages_stream)
        self.beta = type(
            "_FakeBeta",
            (),
            {
                "messages": _FakeMessages(
                    messages_stream if beta_stream is None else beta_stream
                )
            },
        )()


def test_openai_responses_stream_adapter_inherits_tool_truncation_limits() -> None:
    adapter = AnthropicOpenAIResponsesStreamAdapter(
        client=object(),
        max_tool_call_length=444,
        max_tool_call_lines=12,
    )

    tool_adapter = adapter._make_tool_response_adapter()

    assert tool_adapter.max_tool_call_length == 444
    assert tool_adapter.max_tool_call_lines == 12


@pytest.mark.asyncio
async def test_openai_responses_stream_emits_content_part_events():
    events = [
        _Event(type="message_start", message={"id": "msg_1", "model": "claude"}),
        _Event(type="content_block_start", index=0, content_block={"type": "text"}),
        _Event(
            type="content_block_delta",
            index=0,
            delta={"type": "text_delta", "text": "hi"},
        ),
        _Event(type="content_block_stop", index=0),
        _Event(type="message_stop"),
    ]

    stream = _FakeStream(
        events=events,
        final=_FinalMessage(usage={"input_tokens": 3, "output_tokens": 5}),
    )
    client = _FakeClient(messages_stream=stream)

    adapter = AnthropicOpenAIResponsesStreamAdapter(client=client)

    emitted: list[dict] = []

    def handler(e: dict):
        emitted.append(e)

    await adapter._stream_message(
        client=client,
        request={"model": "x", "max_tokens": 5, "messages": []},
        event_handler=handler,
    )

    types = [e["type"] for e in emitted]

    assert "response.created" in types
    assert "response.output_item.added" in types
    assert "response.content_part.added" in types
    assert "response.output_text.delta" in types
    assert "response.output_text.done" in types
    assert "response.content_part.done" in types
    assert "response.output_item.done" in types
    assert "response.completed" in types

    # Sanity-check completed response contains usage.
    completed = next(e for e in emitted if e["type"] == "response.completed")
    assert completed["response"]["usage"]["input_tokens"] == 3
    assert completed["response"]["usage"]["output_tokens"] == 5
    assert completed["response"]["usage"]["total_tokens"] == 8


@pytest.mark.asyncio
async def test_openai_responses_stream_uses_tool_use_id_for_function_call_items():
    events = [
        _Event(type="message_start", message={"id": "msg_1", "model": "claude"}),
        _Event(
            type="content_block_start",
            index=0,
            content_block={
                "type": "tool_use",
                "id": "toolu_123",
                "name": "read_file",
            },
        ),
        _Event(
            type="content_block_delta",
            index=0,
            delta={"type": "input_json_delta", "partial_json": '{"path":"src/app.py"}'},
        ),
        _Event(type="content_block_stop", index=0),
        _Event(type="message_stop"),
    ]

    stream = _FakeStream(
        events=events,
        final=_FinalMessage(usage={"input_tokens": 3, "output_tokens": 5}),
    )
    client = _FakeClient(messages_stream=stream)
    adapter = AnthropicOpenAIResponsesStreamAdapter(client=client)

    emitted: list[dict] = []

    await adapter._stream_message(
        client=client,
        request={"model": "x", "max_tokens": 5, "messages": []},
        event_handler=emitted.append,
    )

    added = next(e for e in emitted if e["type"] == "response.output_item.added")
    done = next(e for e in emitted if e["type"] == "response.output_item.done")
    args_done = next(
        e for e in emitted if e["type"] == "response.function_call_arguments.done"
    )

    assert added["item"]["id"] == "toolu_123"
    assert added["item"]["call_id"] == "toolu_123"
    assert done["item"]["id"] == "toolu_123"
    assert done["item"]["call_id"] == "toolu_123"
    assert args_done["item_id"] == "toolu_123"


@pytest.mark.asyncio
async def test_openai_responses_stream_uses_beta_messages_api_when_betas_present():
    messages_stream = _FakeStream(events=[], final=_FinalMessage())
    beta_stream = _FakeStream(
        events=[
            _Event(type="message_start", message={"id": "msg_1", "model": "claude"}),
            _Event(type="message_stop"),
        ],
        final=_FinalMessage(),
    )
    client = _FakeClient(messages_stream=messages_stream, beta_stream=beta_stream)
    adapter = AnthropicOpenAIResponsesStreamAdapter(client=client)

    emitted: list[dict] = []

    await adapter._stream_message(
        client=client,
        request={
            "model": "x",
            "max_tokens": 5,
            "messages": [],
            "betas": ["context-management-2025-06-27"],
        },
        event_handler=emitted.append,
    )

    assert client.messages.calls == []
    assert len(client.beta.messages.calls) == 1
    assert client.beta.messages.calls[0]["betas"] == ["context-management-2025-06-27"]
    assert any(event["type"] == "response.completed" for event in emitted)


@pytest.mark.asyncio
async def test_openai_responses_stream_create_response_forwards_options(monkeypatch):
    called: dict = {}

    async def _fake_create_response(
        self,
        *,
        context,
        caller,
        toolkits,
        output_schema=None,
        event_handler=None,
        steering_callback=None,
        model=None,
        on_behalf_of=None,
        tool_choice=None,
        options=None,
    ):
        del self
        del context
        del caller
        del toolkits
        del output_schema
        del event_handler
        del steering_callback
        del model
        del on_behalf_of
        del tool_choice
        called["options"] = options
        return "ok"

    monkeypatch.setattr(
        AnthropicMessagesAdapter, "create_response", _fake_create_response
    )

    adapter = AnthropicOpenAIResponsesStreamAdapter(client=object())

    result = await adapter.create_response(
        context=AgentSessionContext(system_role=None),
        caller=object(),
        toolkits=[],
        options={"reasoning": {"effort": "none"}},
    )

    assert result == "ok"
    assert called["options"] == {"reasoning": {"effort": "none"}}


@pytest.mark.asyncio
async def test_openai_responses_stream_create_response_forwards_steering_callback(
    monkeypatch,
):
    called: dict = {"steering_callback": None}

    async def _fake_create_response(
        self,
        *,
        context,
        caller,
        toolkits,
        output_schema=None,
        event_handler=None,
        steering_callback=None,
        model=None,
        on_behalf_of=None,
        tool_choice=None,
        options=None,
    ):
        del self
        del context
        del caller
        del toolkits
        del output_schema
        del event_handler
        del model
        del on_behalf_of
        del tool_choice
        del options
        called["steering_callback"] = steering_callback
        if steering_callback is None:
            return False
        return await steering_callback()

    monkeypatch.setattr(
        AnthropicMessagesAdapter, "create_response", _fake_create_response
    )

    adapter = AnthropicOpenAIResponsesStreamAdapter(client=object())
    steering_calls = 0

    async def _steer() -> bool:
        nonlocal steering_calls
        steering_calls += 1
        return True

    result = await adapter.create_response(
        context=AgentSessionContext(system_role=None),
        caller=object(),
        toolkits=[],
        steering_callback=_steer,
    )

    assert result is True
    assert called["steering_callback"] is _steer
    assert steering_calls == 1

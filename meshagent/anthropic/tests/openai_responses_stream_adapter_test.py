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

    def stream(self, **kwargs):
        return self._stream


class _FakeClient:
    def __init__(self, stream: _FakeStream):
        self.messages = _FakeMessages(stream)


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
    client = _FakeClient(stream)

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
async def test_openai_responses_stream_next_forwards_options(monkeypatch):
    called: dict = {}

    async def _fake_next(
        self,
        *,
        context,
        room,
        toolkits,
        output_schema=None,
        event_handler=None,
        model=None,
        on_behalf_of=None,
        options=None,
    ):
        del self
        del context
        del room
        del toolkits
        del output_schema
        del event_handler
        del model
        del on_behalf_of
        called["options"] = options
        return "ok"

    monkeypatch.setattr(AnthropicMessagesAdapter, "next", _fake_next)

    adapter = AnthropicOpenAIResponsesStreamAdapter(client=object())

    result = await adapter.next(
        context=AgentSessionContext(system_role=None),
        room=object(),
        toolkits=[],
        options={"reasoning": {"effort": "none"}},
    )

    assert result == "ok"
    assert called["options"] == {"reasoning": {"effort": "none"}}

import asyncio
import pytest
from typing import Any

from meshagent.agents.messages import (
    AgentFileContentDelta,
    AgentFileContentEnded,
    AgentFileContentStarted,
    AgentReasoningContentDelta,
    AgentReasoningContentEnded,
    AgentReasoningContentStarted,
    AgentTextContentDelta,
    AgentTextContentEnded,
    AgentTextContentStarted,
    AgentToolCallLogDelta,
    AgentToolCallPending,
    AgentToolCallEnded,
    AgentToolCallStarted,
)
from meshagent.agents.event_publisher import (
    _AgentMessageEmitter,
    _AnthropicAgentEventPublisher,
)
from meshagent.anthropic.messages_adapter import (
    AnthropicMessagesAdapter,
    AnthropicMessagesToolResponseAdapter,
    MessagesToolBundle,
    _AnthropicToolCallingState,
    _default_max_tokens_for_model,
    _consume_streaming_tool_result,
    safe_tool_name,
)
from meshagent.agents.agent import AgentSessionContext
from meshagent.api.messaging import JsonContent, TextContent
from meshagent.tools import FunctionTool, Toolkit, ToolContext
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


class _StrictOptOutTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={"type": "object", "properties": {}, "required": []},
            description="test tool",
            strict=False,
        )

    async def execute(self, context, **kwargs):
        del context
        del kwargs
        return {"ok": True}


class _NumericBoundsTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    }
                },
                "required": ["count"],
                "additionalProperties": False,
            },
            description="numeric bounds test tool",
        )

    async def execute(self, context, **kwargs):
        del context
        return {"ok": True, "args": kwargs}


class _BoundedExecutionTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    }
                },
                "required": ["count"],
                "additionalProperties": False,
            },
            description="bounded execution tool",
        )

    async def execute(self, context: ToolContext, count: int) -> dict[str, Any]:
        del context
        return {"count": count}


class _NullableStringTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={
                "type": "object",
                "properties": {
                    "value": {
                        "type": ["string", "null"],
                    }
                },
                "required": ["value"],
                "additionalProperties": False,
            },
            description="nullable string test tool",
        )

    async def execute(self, context, **kwargs):
        del context
        return {"ok": True, "args": kwargs}


class _WrappedAnyOfObjectTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={
                "type": "object",
                "properties": {
                    "tools": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "anyOf": [
                                        {
                                            "type": "object",
                                            "properties": {
                                                "a": {"type": "string"},
                                            },
                                            "required": ["a"],
                                            "additionalProperties": False,
                                        },
                                        {
                                            "type": "object",
                                            "properties": {
                                                "b": {"type": "integer"},
                                            },
                                            "required": ["b"],
                                            "additionalProperties": False,
                                        },
                                    ],
                                },
                            },
                            {"type": "null"},
                        ]
                    }
                },
                "required": ["tools"],
                "additionalProperties": False,
            },
            description="wrapped anyOf tool",
        )

    async def execute(self, context, **kwargs):
        del context
        return {"ok": True, "args": kwargs}


class _FakeAdapter(AnthropicMessagesAdapter):
    def __init__(self, responses: list[dict], **kwargs):
        super().__init__(client=object(), **kwargs)
        self._responses = responses
        self._idx = 0
        self.requests: list[dict] = []

    async def _create_with_optional_headers(self, *, client, request):
        if self._idx >= len(self._responses):
            raise AssertionError("unexpected extra request")
        self.requests.append(request)
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


class _FailingTool(FunctionTool):
    def __init__(self, name: str, *, strict: bool = True):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="failing test tool",
            strict=strict,
        )

    async def execute(self, context, **kwargs):
        del context
        del kwargs
        raise RoomException("tool failed")


class _BlockingTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="blocking test tool",
        )
        self.started = asyncio.Event()

    async def execute(self, context, **kwargs):
        del context
        del kwargs
        self.started.set()
        await asyncio.Future()


class _GateTool(FunctionTool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="gated test tool",
        )
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, context, **kwargs):
        del context
        del kwargs
        self.started.set()
        await self.release.wait()
        return {"ok": True, "tool": self.name}


class _WriteFileLikeTool(FunctionTool):
    def __init__(self) -> None:
        super().__init__(
            name="write_file",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "text", "overwrite"],
                "properties": {
                    "path": {"type": "string"},
                    "text": {"type": "string"},
                    "overwrite": {"type": "boolean"},
                },
            },
            description="write a file",
        )

    async def execute(self, context, **kwargs):
        del context
        return {"ok": True, "args": kwargs}


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


def test_convert_messages_keeps_real_user_turn_after_tool_result() -> None:
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
            {"role": "user", "content": "stop"},
        ],
    )

    adapter = AnthropicMessagesAdapter(client=object())
    msgs, system = adapter._convert_messages(context=ctx)

    assert system == adapter.get_additional_instructions()
    assert msgs == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
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
        {"role": "user", "content": [{"type": "text", "text": "stop"}]},
    ]


def test_convert_messages_keeps_trailing_user_turn_without_tool_result() -> None:
    ctx = AgentSessionContext(
        system_role=None,
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "working"}],
            },
            {"role": "user", "content": "stop"},
        ],
    )

    adapter = AnthropicMessagesAdapter(client=object())
    msgs, system = adapter._convert_messages(context=ctx)

    assert system == adapter.get_additional_instructions()
    assert msgs == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "working"}],
        },
        {"role": "user", "content": [{"type": "text", "text": "stop"}]},
    ]


def test_convert_messages_appends_additional_instructions_to_existing_system() -> None:
    adapter = AnthropicMessagesAdapter(client=object())
    ctx = AgentSessionContext(system_role=None, instructions="base instructions")

    _, system = adapter._convert_messages(context=ctx)

    assert system == ("base instructions\n\n" + adapter.get_additional_instructions())


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


def test_make_agent_event_publisher_emits_tool_log_delta() -> None:
    adapter = AnthropicMessagesAdapter(client=object())
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher(
        {
            "type": "meshagent.handler.output",
            "item_id": "toolu_1",
            "lines": [
                {"source": "stdout", "text": "line-1"},
                {"source": "stdout", "text": "line-2"},
            ],
        }
    )

    assert len(published) == 1
    log_delta = published[0]
    assert isinstance(log_delta, AgentToolCallLogDelta)
    assert log_delta.item_id == "toolu_1"
    assert [(line.source, line.text) for line in log_delta.lines] == [
        ("stdout", "line-1"),
        ("stdout", "line-2"),
    ]


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


def test_messages_tool_bundle_marks_function_tools_strict_by_default() -> None:
    toolkit = Toolkit(name="test", tools=[_AnyArgsTool("tool_a")])

    tool_bundle = MessagesToolBundle(toolkits=[toolkit])

    assert tool_bundle.to_json() == [
        {
            "name": "tool_a",
            "description": "test tool",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]


def test_messages_tool_bundle_allows_tools_to_opt_out_of_strict_mode() -> None:
    toolkit = Toolkit(name="test", tools=[_StrictOptOutTool("tool_a")])

    tool_bundle = MessagesToolBundle(toolkits=[toolkit])

    assert tool_bundle.to_json() == [
        {
            "name": "tool_a",
            "description": "test tool",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
            "strict": False,
        }
    ]


def test_messages_tool_bundle_transforms_unsupported_numeric_constraints() -> None:
    toolkit = Toolkit(name="test", tools=[_NumericBoundsTool("tool_a")])

    tool_bundle = MessagesToolBundle(toolkits=[toolkit])

    assert tool_bundle.to_json() == [
        {
            "name": "tool_a",
            "description": "numeric bounds test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "{minimum: 1, maximum: 5}",
                    }
                },
                "required": ["count"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]


@pytest.mark.asyncio
async def test_messages_tool_bundle_execute_respects_context_validation_mode():
    toolkit = Toolkit(name="test", tools=[_BoundedExecutionTool("tool_a")])
    tool_bundle = MessagesToolBundle(toolkits=[toolkit])

    result = await tool_bundle.execute(
        context=ToolContext(
            room=_DummyRoom(),
            caller=_DummyParticipant(),
            validation_mode="content_types",
        ),
        tool_use={"name": "tool_a", "id": "toolu_1", "input": {"count": 9}},
    )

    assert isinstance(result, JsonContent)
    assert result.json == {"count": 9}


def test_messages_tool_bundle_uses_content_types_validation_for_strict_tools() -> None:
    toolkit = Toolkit(name="test", tools=[_NumericBoundsTool("tool_a")])
    tool_bundle = MessagesToolBundle(toolkits=[toolkit])

    assert (
        tool_bundle.validation_mode_for_tool_use(
            tool_use={"name": "tool_a", "id": "toolu_1", "input": {}}
        )
        == "content_types"
    )


def test_messages_tool_bundle_normalizes_nullable_union_types() -> None:
    toolkit = Toolkit(name="test", tools=[_NullableStringTool("tool_a")])

    tool_bundle = MessagesToolBundle(toolkits=[toolkit])

    assert tool_bundle.to_json() == [
        {
            "name": "tool_a",
            "description": "nullable string test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                    }
                },
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]


def test_messages_tool_bundle_strips_empty_object_wrapper_around_anyof() -> None:
    toolkit = Toolkit(name="test", tools=[_WrappedAnyOfObjectTool("tool_a")])

    tool_bundle = MessagesToolBundle(toolkits=[toolkit])

    assert tool_bundle.to_json() == [
        {
            "name": "tool_a",
            "description": "wrapped anyOf tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "tools": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "a": {"type": "string"},
                                    },
                                    "additionalProperties": False,
                                    "required": ["a"],
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "b": {"type": "integer"},
                                    },
                                    "additionalProperties": False,
                                    "required": ["b"],
                                },
                            ]
                        },
                    }
                },
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]


def test_messages_tool_bundle_loose_mode_disables_strict_tools() -> None:
    toolkit = Toolkit(name="test", tools=[_AnyArgsTool("tool_a")])
    state = _AnthropicToolCallingState(mode="loose")

    tool_bundle = MessagesToolBundle(
        toolkits=[toolkit],
        tool_calling_state=state,
    )

    assert tool_bundle.to_json() == [
        {
            "name": "tool_a",
            "description": "test tool",
            "input_schema": {
                "type": "object",
                "additionalProperties": True,
            },
            "strict": False,
        }
    ]


def test_messages_tool_bundle_strict_mode_forces_strict_on_non_strict_tools() -> None:
    toolkit = Toolkit(name="test", tools=[_StrictOptOutTool("tool_a")])
    state = _AnthropicToolCallingState(mode="strict")

    tool_bundle = MessagesToolBundle(
        toolkits=[toolkit],
        tool_calling_state=state,
    )

    assert tool_bundle.to_json() == [
        {
            "name": "tool_a",
            "description": "test tool",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
                "required": [],
            },
            "strict": True,
        }
    ]


def test_messages_tool_bundle_adaptive_mode_enables_strict_after_failure() -> None:
    tool = _NumericBoundsTool("tool_a")
    toolkit = Toolkit(name="test", tools=[tool])
    state = _AnthropicToolCallingState(mode="adaptive")

    initial_bundle = MessagesToolBundle(
        toolkits=[toolkit],
        tool_calling_state=state,
    )
    assert initial_bundle.to_json() == [
        {
            "name": "tool_a",
            "description": "numeric bounds test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    }
                },
                "required": ["count"],
                "additionalProperties": False,
            },
            "strict": False,
        }
    ]

    initial_bundle.record_tool_failure(
        tool_use={"name": "tool_a", "id": "toolu_1", "input": {}}
    )

    retried_bundle = MessagesToolBundle(
        toolkits=[toolkit],
        tool_calling_state=state,
    )
    assert retried_bundle.to_json() == [
        {
            "name": "tool_a",
            "description": "numeric bounds test tool",
            "input_schema": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "{minimum: 1, maximum: 5}",
                    }
                },
                "required": ["count"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]


def test_messages_tool_bundle_adaptive_mode_does_not_enable_non_strict_tools() -> None:
    toolkit = Toolkit(name="test", tools=[_FailingTool("tool_a", strict=False)])
    state = _AnthropicToolCallingState(mode="adaptive")

    initial_bundle = MessagesToolBundle(
        toolkits=[toolkit],
        tool_calling_state=state,
    )
    initial_bundle.record_tool_failure(
        tool_use={"name": "tool_a", "id": "toolu_1", "input": {}}
    )

    retried_bundle = MessagesToolBundle(
        toolkits=[toolkit],
        tool_calling_state=state,
    )
    assert retried_bundle.to_json() == [
        {
            "name": "tool_a",
            "description": "failing test tool",
            "input_schema": {
                "type": "object",
                "additionalProperties": True,
            },
            "strict": False,
        }
    ]


def test_default_max_tokens_for_model_uses_model_family_defaults() -> None:
    assert _default_max_tokens_for_model("claude-sonnet-4-6") == 64_000
    assert _default_max_tokens_for_model("claude-sonnet-4-5") == 64_000
    assert _default_max_tokens_for_model("claude-3-7-sonnet-latest") == 64_000
    assert _default_max_tokens_for_model("claude-opus-4-6") == 128_000
    assert _default_max_tokens_for_model("claude-opus-4-1") == 32_000
    assert _default_max_tokens_for_model("claude-3-5-sonnet-latest") == 8_192


@pytest.mark.asyncio
async def test_next_uses_model_specific_max_tokens_default() -> None:
    adapter = _FakeAdapter(
        responses=[{"content": [{"type": "text", "text": "ok"}]}],
        model="claude-3-5-sonnet-latest",
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
        model="claude-sonnet-4-6",
    )

    assert result == "ok"
    assert adapter.requests[0]["max_tokens"] == 64_000


@pytest.mark.asyncio
async def test_next_uses_higher_opus_4_6_model_specific_max_tokens_default() -> None:
    adapter = _FakeAdapter(
        responses=[{"content": [{"type": "text", "text": "ok"}]}],
        model="claude-3-5-sonnet-latest",
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
        model="claude-opus-4-6",
    )

    assert result == "ok"
    assert adapter.requests[0]["max_tokens"] == 128_000


@pytest.mark.asyncio
async def test_next_prefers_explicit_max_tokens_over_model_default() -> None:
    adapter = _FakeAdapter(
        responses=[{"content": [{"type": "text", "text": "ok"}]}],
        model="claude-sonnet-4-6",
        max_tokens=1234,
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
    )

    assert result == "ok"
    assert adapter.requests[0]["max_tokens"] == 1234


@pytest.mark.asyncio
async def test_next_prefers_env_max_tokens_over_model_default(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "4096")
    adapter = _FakeAdapter(
        responses=[{"content": [{"type": "text", "text": "ok"}]}],
        model="claude-sonnet-4-6",
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
    )

    assert result == "ok"
    assert adapter.requests[0]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_next_uses_native_output_config_with_strict_schema() -> None:
    adapter = _FakeAdapter(
        responses=[
            {"content": [{"type": "text", "text": '{"answer":"done"}'}]},
        ]
    )
    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message("answer")

    result = await adapter.next(
        context=ctx,
        room=_DummyRoom(),
        toolkits=[],
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        },
    )

    assert result == {"answer": "done"}
    assert adapter.requests[0]["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        }
    }
    assert "betas" not in adapter.requests[0]


@pytest.mark.asyncio
async def test_next_transforms_unsupported_numeric_output_constraints() -> None:
    adapter = _FakeAdapter(
        responses=[
            {"content": [{"type": "text", "text": '{"count":3}'}]},
        ]
    )
    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message("answer")

    result = await adapter.next(
        context=ctx,
        room=_DummyRoom(),
        toolkits=[],
        output_schema={
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                }
            },
            "required": ["count"],
            "additionalProperties": False,
        },
    )

    assert result == {"count": 3}
    assert adapter.requests[0]["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "{minimum: 1, maximum: 5}",
                    }
                },
                "required": ["count"],
                "additionalProperties": False,
            },
        }
    }
    assert "betas" not in adapter.requests[0]


@pytest.mark.asyncio
async def test_next_normalizes_nullable_union_types_in_output_schema() -> None:
    adapter = _FakeAdapter(
        responses=[
            {"content": [{"type": "text", "text": '{"value":null}'}]},
        ]
    )
    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message("answer")

    result = await adapter.next(
        context=ctx,
        room=_DummyRoom(),
        toolkits=[],
        output_schema={
            "type": "object",
            "properties": {
                "value": {
                    "type": ["string", "null"],
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        },
    )

    assert result == {"value": None}
    assert adapter.requests[0]["output_config"] == {
        "format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                    }
                },
                "additionalProperties": False,
            },
        }
    }
    assert "betas" not in adapter.requests[0]


@pytest.mark.asyncio
async def test_next_adaptive_mode_retries_with_loose_tools_after_grammar_error(
    monkeypatch,
) -> None:
    class _FakeAPIStatusError(Exception):
        pass

    monkeypatch.setattr(
        "meshagent.anthropic.messages_adapter.APIStatusError",
        _FakeAPIStatusError,
    )

    adapter = AnthropicMessagesAdapter(client=object(), tool_calling_mode="adaptive")
    tool = _AnyArgsTool("tool_a")
    adapter._tool_calling_state.record_tool_failure(tool_name="tool_a", tool=tool)
    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message("run tool")

    requests: list[dict] = []

    async def _fake_create_with_optional_headers(*, client, request):
        del client
        requests.append(request)
        if len(requests) == 1:
            raise _FakeAPIStatusError(
                "The compiled grammar is too large, which would cause performance issues. "
                "Simplify your tool schemas or reduce the number of strict tools."
            )
        return {"content": [{"type": "text", "text": "done"}]}

    monkeypatch.setattr(
        adapter,
        "_create_with_optional_headers",
        _fake_create_with_optional_headers,
    )

    result = await adapter.next(
        context=ctx,
        room=_DummyRoom(),
        toolkits=[Toolkit(name="test", tools=[tool])],
    )

    assert result == "done"
    assert requests[0]["tools"][0]["strict"] is True
    assert requests[1]["tools"][0]["strict"] is False
    assert "betas" not in requests[0]
    assert "betas" not in requests[1]


@pytest.mark.asyncio
async def test_next_adaptive_mode_enables_strict_after_local_tool_validation_failure(
    monkeypatch,
) -> None:
    tool = _WriteFileLikeTool()
    adapter = AnthropicMessagesAdapter(client=object(), tool_calling_mode="adaptive")
    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message("write the file")

    requests: list[dict[str, Any]] = []
    responses = [
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "write_file",
                    "input": {"path": "/tmp/test.txt"},
                }
            ]
        },
        {"content": [{"type": "text", "text": "done"}]},
    ]

    async def _fake_create_with_optional_headers(*, client, request):
        del client
        requests.append(request)
        if len(responses) == 0:
            raise AssertionError("unexpected extra request")
        return responses.pop(0)

    monkeypatch.setattr(
        adapter,
        "_create_with_optional_headers",
        _fake_create_with_optional_headers,
    )

    result = await adapter.next(
        context=ctx,
        room=_DummyRoom(),
        toolkits=[Toolkit(name="storage", tools=[tool])],
    )

    assert result == "done"
    assert requests[0]["tools"][0]["name"] == "write_file"
    assert requests[0]["tools"][0]["strict"] is False
    assert requests[1]["tools"][0]["name"] == "write_file"
    assert requests[1]["tools"][0]["strict"] is True
    assert "betas" not in requests[0]
    assert "betas" not in requests[1]


@pytest.mark.asyncio
async def test_next_inserts_steering_messages_after_tool_results() -> None:
    adapter = _FakeAdapter(
        responses=[
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "tool_a",
                        "input": {"path": "/tmp/test.txt"},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            },
        ],
        model="claude-sonnet-4-6",
    )
    context = adapter.create_session()
    context.append_user_message("run tool")
    steering_calls = 0

    async def _steer() -> bool:
        nonlocal steering_calls
        steering_calls += 1
        context.append_user_message("steer now")
        return True

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[Toolkit(name="test", tools=[_AnyArgsTool("tool_a")])],
        steering_callback=_steer,
    )

    assert result == "done"
    assert steering_calls == 1
    assert len(adapter.requests) == 2
    second_messages = adapter.requests[1]["messages"]
    assert second_messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "tool_a",
                "input": {"path": "/tmp/test.txt"},
            }
        ],
    }
    assert second_messages[2]["role"] == "user"
    assert second_messages[2]["content"][0]["type"] == "tool_result"
    assert second_messages[2]["content"][0]["tool_use_id"] == "toolu_1"
    assert second_messages[3] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "TURN INTERRUPTED"}],
    }
    assert second_messages[4] == {
        "role": "user",
        "content": [{"type": "text", "text": "steer now"}],
    }
    assert len(second_messages) == 5
    assert adapter.requests[1].get("system") == adapter.get_additional_instructions()


@pytest.mark.asyncio
async def test_steering_followup_request_matches_plain_transcript_shape() -> None:
    adapter = _FakeAdapter(
        responses=[
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "tool_a",
                        "input": {"path": "/tmp/test.txt"},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            },
        ],
        model="claude-sonnet-4-6",
    )
    context = adapter.create_session()
    context.append_user_message("run tool")

    async def _steer() -> bool:
        context.append_user_message("steer now")
        return True

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[Toolkit(name="test", tools=[_AnyArgsTool("tool_a")])],
        steering_callback=_steer,
    )

    assert result == "done"

    manual_context = AgentSessionContext(
        system_role=None,
        messages=[
            {"role": "user", "content": "run tool"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "tool_a",
                        "input": {"path": "/tmp/test.txt"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {
                                "type": "text",
                                "text": '{"ok": true, "args": {"path": "/tmp/test.txt"}}',
                            }
                        ],
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "TURN INTERRUPTED"}],
            },
            {"role": "user", "content": "steer now"},
        ],
    )

    manual_messages, manual_system = adapter._convert_messages(context=manual_context)

    assert adapter.requests[1]["messages"] == manual_messages
    assert adapter.requests[1].get("system") == manual_system


@pytest.mark.asyncio
async def test_next_inserts_steering_before_trailing_tool_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(
        responses=[
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "tool_a",
                        "input": {"path": "/tmp/test.txt"},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            },
        ],
        model="claude-sonnet-4-6",
    )
    context = adapter.create_session()
    context.append_user_message("run tool")

    original_create_messages = AnthropicMessagesToolResponseAdapter.create_messages

    async def _create_messages_with_trailing(
        self,
        *,
        context,
        tool_call,
        room,
        response,
    ):
        messages = await original_create_messages(
            self,
            context=context,
            tool_call=tool_call,
            room=room,
            response=response,
        )
        return [
            *messages,
            {"role": "assistant", "content": [{"type": "text", "text": "too late"}]},
        ]

    monkeypatch.setattr(
        AnthropicMessagesToolResponseAdapter,
        "create_messages",
        _create_messages_with_trailing,
    )

    async def _steer() -> bool:
        context.append_user_message("steer now")
        return True

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[Toolkit(name="test", tools=[_AnyArgsTool("tool_a")])],
        steering_callback=_steer,
    )

    assert result == "done"
    second_messages = adapter.requests[1]["messages"]
    assert second_messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "tool_a",
                "input": {"path": "/tmp/test.txt"},
            }
        ],
    }
    assert second_messages[2]["role"] == "user"
    assert second_messages[2]["content"][0]["type"] == "tool_result"
    assert second_messages[2]["content"][0]["tool_use_id"] == "toolu_1"
    assert second_messages[3] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "TURN INTERRUPTED"}],
    }
    assert second_messages[4] == {
        "role": "user",
        "content": [{"type": "text", "text": "steer now"}],
    }
    assert len(second_messages) == 5
    assert adapter.requests[1].get("system") == adapter.get_additional_instructions()
    assert not any(
        message.get("role") == "assistant"
        and message.get("content") == [{"type": "text", "text": "too late"}]
        for message in second_messages
    )


@pytest.mark.asyncio
async def test_next_inserts_steering_after_first_completed_tool_when_multiple_tools_are_pending() -> (
    None
):
    adapter = _FakeAdapter(
        responses=[
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "tool_a",
                        "input": {},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "tool_b",
                        "input": {},
                    },
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            },
        ],
        model="claude-sonnet-4-6",
    )
    tool_a = _GateTool("tool_a")
    tool_b = _BlockingTool("tool_b")
    context = adapter.create_session()
    context.append_user_message("run tools")
    pending_steer = False

    async def _steer() -> bool:
        nonlocal pending_steer
        if not pending_steer:
            return False
        pending_steer = False
        context.append_user_message("steer now")
        return True

    task = asyncio.create_task(
        adapter.next(
            context=context,
            room=_DummyRoom(),
            toolkits=[Toolkit(name="test", tools=[tool_a, tool_b])],
            steering_callback=_steer,
        )
    )

    await asyncio.wait_for(tool_a.started.wait(), timeout=1)
    await asyncio.wait_for(tool_b.started.wait(), timeout=1)
    pending_steer = True
    tool_a.release.set()
    result = await asyncio.wait_for(task, timeout=1)

    assert result == "done"
    assert len(adapter.requests) == 2
    second_messages = adapter.requests[1]["messages"]
    assert second_messages[1] == {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "tool_a", "input": {}},
            {"type": "tool_use", "id": "toolu_2", "name": "tool_b", "input": {}},
        ],
    }
    assert second_messages[2]["role"] == "user"
    assert any(
        isinstance(block, dict)
        and block.get("type") == "tool_result"
        and block.get("tool_use_id") == "toolu_1"
        for block in second_messages[2]["content"]
    )
    assert second_messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": [{"type": "text", "text": '{"ok": true, "tool": "tool_a"}'}],
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Cancelled because queued steering took priority over the remaining tool calls."
                        ),
                    }
                ],
            },
        ],
    }
    assert second_messages[3] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "TURN INTERRUPTED"}],
    }
    assert second_messages[4] == {
        "role": "user",
        "content": [{"type": "text", "text": "steer now"}],
    }
    assert adapter.requests[1].get("system") == adapter.get_additional_instructions()
    assert len(second_messages) == 5


@pytest.mark.asyncio
async def test_next_restores_context_when_cancelled_during_tool_call() -> None:
    blocking_tool = _BlockingTool("tool_a")
    adapter = _FakeAdapter(
        responses=[
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "tool_a",
                        "input": {"path": "/tmp/test.txt"},
                    }
                ],
            }
        ],
        model="claude-sonnet-4-6",
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("run tool")

    task = asyncio.create_task(
        adapter.next(
            context=context,
            room=_DummyRoom(),
            toolkits=[Toolkit(name="test", tools=[blocking_tool])],
        )
    )

    await asyncio.wait_for(blocking_tool.started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert context.messages == [{"role": "user", "content": "run tool"}]


@pytest.mark.asyncio
async def test_next_raises_on_truncated_tool_calls_before_appending_assistant_message():
    adapter = _FakeAdapter(
        responses=[
            {
                "stop_reason": "max_tokens",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "write_file",
                        "input": {"path": "/tmp/test.txt"},
                    }
                ],
            }
        ],
        model="claude-sonnet-4-6",
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("write the file")

    with pytest.raises(
        RoomException,
        match="Anthropic response hit max_tokens before completing tool calls",
    ):
        await adapter.next(
            context=context,
            room=_DummyRoom(),
            toolkits=[Toolkit(name="storage", tools=[_WriteFileLikeTool()])],
        )

    assert context.messages == [{"role": "user", "content": "write the file"}]


@pytest.mark.asyncio
async def test_next_marks_truncated_streamed_tool_calls_as_failed(
    monkeypatch,
) -> None:
    adapter = AnthropicMessagesAdapter(client=object(), model="claude-sonnet-4-6")
    context = AgentSessionContext(system_role=None)
    context.append_user_message("write the file")
    events: list[dict[str, Any]] = []

    async def _fake_stream_message(*, client, request, event_handler):
        del client
        del request
        event_handler(
            {
                "type": "content_block_start",
                "event": {
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "write_file",
                        "input": {"path": "/tmp/test.txt"},
                    },
                },
            }
        )
        return {
            "stop_reason": "max_tokens",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "write_file",
                    "input": {"path": "/tmp/test.txt"},
                }
            ],
        }

    monkeypatch.setattr(adapter, "_stream_message", _fake_stream_message)

    with pytest.raises(
        RoomException,
        match="Anthropic response hit max_tokens before completing tool calls",
    ):
        await adapter.next(
            context=context,
            room=_DummyRoom(),
            toolkits=[Toolkit(name="storage", tools=[_WriteFileLikeTool()])],
            event_handler=events.append,
        )

    assert {
        "type": "meshagent.handler.done",
        "item_id": "toolu_1",
        "error": (
            "Anthropic response hit max_tokens before completing tool calls. "
            "Increase max_tokens and retry."
        ),
    } in events


@pytest.mark.asyncio
async def test_next_continues_on_pause_turn() -> None:
    adapter = _FakeAdapter(
        responses=[
            {
                "stop_reason": "pause_turn",
                "content": [
                    {
                        "type": "mcp_tool_use",
                        "id": "mcpu_1",
                        "server_name": "web",
                        "name": "search",
                        "input": {"query": "anthropic"},
                    }
                ],
            },
            {
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "done"}],
            },
        ],
        model="claude-sonnet-4-6",
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("search")

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
    )

    assert result == "done"
    assert len(adapter.requests) == 2
    assert context.messages == [
        {"role": "user", "content": "search"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "mcp_tool_use",
                    "id": "mcpu_1",
                    "server_name": "web",
                    "name": "search",
                    "input": {"query": "anthropic"},
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
    ]


@pytest.mark.asyncio
async def test_next_raises_on_truncated_text_response_after_appending_assistant_message():
    adapter = _FakeAdapter(
        responses=[
            {
                "stop_reason": "max_tokens",
                "content": [{"type": "text", "text": "partial answer"}],
            }
        ],
        model="claude-sonnet-4-6",
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("answer")

    with pytest.raises(
        RoomException,
        match="Anthropic response hit max_tokens before completing the turn",
    ):
        await adapter.next(
            context=context,
            room=_DummyRoom(),
            toolkits=[],
        )

    assert context.messages == [
        {"role": "user", "content": "answer"},
        {"role": "assistant", "content": [{"type": "text", "text": "partial answer"}]},
    ]


@pytest.mark.asyncio
async def test_next_raises_on_model_context_window_exceeded_after_appending_assistant_message():
    adapter = _FakeAdapter(
        responses=[
            {
                "stop_reason": "model_context_window_exceeded",
                "content": [{"type": "text", "text": "partial answer"}],
            }
        ],
        model="claude-sonnet-4-6",
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("answer")

    with pytest.raises(
        RoomException,
        match="Anthropic response hit the model context window before completing the turn",
    ):
        await adapter.next(
            context=context,
            room=_DummyRoom(),
            toolkits=[],
        )

    assert context.messages == [
        {"role": "user", "content": "answer"},
        {"role": "assistant", "content": [{"type": "text", "text": "partial answer"}]},
    ]


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


def test_create_chat_context_supports_remote_image_and_file_urls() -> None:
    adapter = AnthropicMessagesAdapter(client=object())
    context = adapter.create_session()

    image_message = context.append_image_url(url="https://example.com/image.png")
    file_message = context.append_file_url(url="https://example.com/report.pdf")

    assert image_message["content"][0] == {
        "type": "image",
        "source": {
            "type": "url",
            "url": "https://example.com/image.png",
        },
    }
    assert file_message["content"][0] == {
        "type": "document",
        "title": "report.pdf",
        "source": {
            "type": "url",
            "url": "https://example.com/report.pdf",
        },
    }


def test_constructor_rejects_invalid_context_management_mode() -> None:
    with pytest.raises(
        ValueError, match="context_management must be one of 'auto' or 'none'"
    ):
        AnthropicMessagesAdapter(client=object(), context_management="invalid")


def test_constructor_rejects_invalid_tool_calling_mode() -> None:
    with pytest.raises(
        ValueError,
        match="tool_calling_mode must be one of 'loose', 'strict', 'explicit', or 'adaptive'",
    ):
        AnthropicMessagesAdapter(client=object(), tool_calling_mode="invalid")


def test_constructor_rejects_invalid_compaction_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="compaction_threshold must be greater than or equal to 50000",
    ):
        AnthropicMessagesAdapter(client=object(), compaction_threshold=49999)


@pytest.mark.asyncio
async def test_next_adds_auto_compaction_request_fields() -> None:
    adapter = _FakeAdapter(
        responses=[{"content": [{"type": "text", "text": "ok"}]}],
        model="claude-sonnet-4-6",
        context_management="auto",
        compaction_threshold=50001,
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
    )

    assert result == "ok"
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert request["context_management"] == {
        "edits": [
            {
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 50001},
                "pause_after_compaction": False,
            }
        ]
    }
    assert "compact-2026-01-12" in request["betas"]


@pytest.mark.asyncio
async def test_next_preserves_existing_betas_when_adding_compaction_beta() -> None:
    adapter = _FakeAdapter(
        responses=[{"content": [{"type": "text", "text": "ok"}]}],
        model="claude-opus-4-6",
        context_management="auto",
        message_options={"betas": ["mcp-client-2025-11-20"]},
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
    )

    assert result == "ok"
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert "mcp-client-2025-11-20" in request["betas"]
    assert "compact-2026-01-12" in request["betas"]


@pytest.mark.asyncio
async def test_next_skips_auto_compaction_for_legacy_model_versions() -> None:
    adapter = _FakeAdapter(
        responses=[{"content": [{"type": "text", "text": "ok"}]}],
        model="claude-sonnet-4-5",
        context_management="auto",
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
    )

    assert result == "ok"
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert "context_management" not in request
    assert "betas" not in request


@pytest.mark.asyncio
async def test_next_uses_context_management_beta_for_non_compaction_edits() -> None:
    adapter = _FakeAdapter(
        responses=[{"content": [{"type": "text", "text": "ok"}]}],
        message_options={
            "context_management": {
                "edits": [{"type": "clear_tool_uses_20250919", "trigger": "auto"}]
            }
        },
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
    )

    assert result == "ok"
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert "context-management-2025-06-27" in request["betas"]


@pytest.mark.asyncio
async def test_next_stores_usage_metadata() -> None:
    adapter = _FakeAdapter(
        responses=[
            {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "iterations": [
                        {
                            "type": "compaction",
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                        }
                    ],
                },
            }
        ]
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
    )

    assert result == "ok"
    assert context.turn_count == 1
    assert context.metadata["last_response_model"] == adapter.default_model()
    assert context.metadata["last_response_usage"]["input_tokens"] == 10
    assert context.metadata["last_response_usage"]["output_tokens"] == 5
    assert context.usage == {
        "input_tokens": 10.0,
        "output_tokens": 5.0,
    }


@pytest.mark.asyncio
async def test_next_stores_usage_for_streaming_response(monkeypatch) -> None:
    adapter = _FakeAdapter(responses=[])
    context = AgentSessionContext(system_role=None)
    context.append_user_message("hello")
    events: list[dict] = []

    async def _fake_stream_message(*, client, request, event_handler):
        del client
        del request
        event_handler({"type": "message.delta"})
        return {
            "content": [{"type": "text", "text": "ok"}],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 4,
            },
        }

    monkeypatch.setattr(adapter, "_stream_message", _fake_stream_message)

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
        event_handler=events.append,
    )

    assert result == "ok"
    assert events[0]["type"] == "message.delta"
    assert context.turn_count == 1
    assert context.metadata["last_response_usage"]["input_tokens"] == 7
    assert context.usage == {
        "input_tokens": 7.0,
        "output_tokens": 4.0,
    }


@pytest.mark.asyncio
async def test_next_accepts_options_keyword_argument() -> None:
    adapter = _FakeAdapter(
        responses=[{"content": [{"type": "text", "text": "ok"}]}],
    )
    context = AgentSessionContext(system_role=None)
    context.append_user_message("hello")

    result = await adapter.next(
        context=context,
        room=_DummyRoom(),
        toolkits=[],
        options={"reasoning": {"effort": "none"}},
    )

    assert result == "ok"
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert "reasoning" not in request


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


def test_make_agent_event_publisher_emits_native_anthropic_messages() -> None:
    adapter = AnthropicMessagesAdapter(client=object())
    published = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher({"type": "message_start", "event": {"message": {"id": "msg_1"}}})
    publisher(
        {
            "type": "content_block_start",
            "event": {"index": 0, "content_block": {"type": "thinking"}},
        }
    )
    publisher(
        {
            "type": "content_block_delta",
            "event": {
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "trace"},
            },
        }
    )
    publisher({"type": "content_block_stop", "event": {"index": 0}})
    publisher(
        {
            "type": "content_block_start",
            "event": {"index": 1, "content_block": {"type": "text", "text": ""}},
        }
    )
    publisher(
        {
            "type": "content_block_delta",
            "event": {
                "index": 1,
                "delta": {"type": "text_delta", "text": "hello"},
            },
        }
    )
    publisher({"type": "content_block_stop", "event": {"index": 1}})
    publisher(
        {
            "type": "content_block_start",
            "event": {
                "index": 2,
                "content_block": {
                    "type": "document",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/report.pdf",
                    },
                },
            },
        }
    )
    publisher({"type": "content_block_stop", "event": {"index": 2}})
    publisher(
        {
            "type": "content_block_start",
            "event": {
                "index": 3,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": {"q": "meshagent"},
                },
            },
        }
    )
    publisher({"type": "content_block_stop", "event": {"index": 3}})
    publisher(
        {
            "type": "meshagent.handler.added",
            "item": {
                "type": "function_call",
                "id": "toolu_1",
                "call_id": "toolu_1",
                "name": "lookup",
                "arguments": '{"q":"meshagent"}',
            },
        }
    )
    publisher({"type": "meshagent.handler.done", "item_id": "toolu_1"})
    publisher(
        {
            "type": "content_block_start",
            "event": {
                "index": 4,
                "content_block": {
                    "type": "mcp_tool_use",
                    "id": "mcpu_1",
                    "server_name": "deepwiki",
                    "name": "search",
                    "input": {"query": "meshagent"},
                },
            },
        }
    )
    publisher({"type": "content_block_stop", "event": {"index": 4}})
    publisher(
        {
            "type": "meshagent.handler.added",
            "item": {
                "type": "mcp_call",
                "id": "mcpu_1",
                "call_id": "mcpu_1",
                "server_label": "deepwiki",
                "name": "search",
                "arguments": {"query": "meshagent"},
            },
        }
    )
    publisher({"type": "meshagent.handler.done", "item_id": "mcpu_1"})

    assert [type(event) for event in published] == [
        AgentReasoningContentStarted,
        AgentReasoningContentDelta,
        AgentReasoningContentEnded,
        AgentTextContentStarted,
        AgentTextContentDelta,
        AgentTextContentEnded,
        AgentFileContentStarted,
        AgentFileContentDelta,
        AgentFileContentEnded,
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]
    for event in published:
        assert event.thread_id == "thread-1"

    text_delta = published[4]
    assert isinstance(text_delta, AgentTextContentDelta)
    assert text_delta.turn_id == "turn-1"
    assert text_delta.item_id == "msg_1:content:1"
    assert text_delta.text == "hello"

    file_started = published[6]
    assert isinstance(file_started, AgentFileContentStarted)
    assert file_started.item_id == "msg_1:content:2"

    file_delta = published[7]
    assert isinstance(file_delta, AgentFileContentDelta)
    assert file_delta.url == "https://example.com/report.pdf"

    function_pending = published[9]
    assert isinstance(function_pending, AgentToolCallPending)
    assert function_pending.toolkit == "function"
    assert function_pending.tool == "lookup"
    assert function_pending.arguments == {"q": "meshagent"}

    function_started = published[10]
    assert isinstance(function_started, AgentToolCallStarted)
    assert function_started.toolkit == "function"
    assert function_started.tool == "lookup"
    assert function_started.arguments == {"q": "meshagent"}

    mcp_pending = published[12]
    assert isinstance(mcp_pending, AgentToolCallPending)
    assert mcp_pending.toolkit == "deepwiki"
    assert mcp_pending.tool == "search"

    mcp_started = published[13]
    assert isinstance(mcp_started, AgentToolCallStarted)
    assert mcp_started.toolkit == "deepwiki"
    assert mcp_started.tool == "search"


def test_make_agent_event_publisher_preserves_native_text_delta_whitespace() -> None:
    adapter = AnthropicMessagesAdapter(client=object())
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher({"type": "message_start", "event": {"message": {"id": "msg_1"}}})
    publisher(
        {
            "type": "content_block_start",
            "event": {"index": 0, "content_block": {"type": "text", "text": ""}},
        }
    )
    publisher(
        {
            "type": "content_block_delta",
            "event": {
                "index": 0,
                "delta": {"type": "text_delta", "text": "hello"},
            },
        }
    )
    publisher(
        {
            "type": "content_block_delta",
            "event": {
                "index": 0,
                "delta": {"type": "text_delta", "text": " world"},
            },
        }
    )
    publisher({"type": "content_block_stop", "event": {"index": 0}})

    deltas = [
        event.text for event in published if isinstance(event, AgentTextContentDelta)
    ]

    assert deltas == ["hello", " world"]
    assert "".join(deltas) == "hello world"


def test_make_agent_event_publisher_preserves_anthropic_tool_json_delta_whitespace():
    published: list[object] = []
    publisher = _AnthropicAgentEventPublisher(
        emitter=_AgentMessageEmitter(
            turn_id="turn-1",
            thread_id="thread-1",
            callback=published.append,
        )
    )

    publisher({"type": "message_start", "event": {"message": {"id": "msg_1"}}})
    publisher(
        {
            "type": "content_block_start",
            "event": {
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "lookup",
                    "input": None,
                },
            },
        }
    )
    publisher(
        {
            "type": "content_block_delta",
            "event": {
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"q":"hello',
                },
            },
        }
    )
    publisher(
        {
            "type": "content_block_delta",
            "event": {
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": ' world"}',
                },
            },
        }
    )

    assert publisher._blocks[0].arguments_text == '{"q":"hello world"}'

    publisher({"type": "content_block_stop", "event": {"index": 0}})
    publisher(
        {
            "type": "meshagent.handler.added",
            "item": {
                "type": "function_call",
                "id": "toolu_1",
                "call_id": "toolu_1",
                "name": "lookup",
                "arguments": '{"q":"hello world"}',
            },
        }
    )
    publisher({"type": "meshagent.handler.done", "item_id": "toolu_1"})

    assert [type(event) for event in published] == [
        AgentToolCallPending,
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]
    updated_pending = published[1]
    assert isinstance(updated_pending, AgentToolCallPending)
    assert updated_pending.arguments == {"q": "hello world"}
    updated_started = published[2]
    assert isinstance(updated_started, AgentToolCallStarted)
    assert updated_started.arguments == {"q": "hello world"}
    assert isinstance(published[-1], AgentToolCallEnded)


def test_make_agent_event_publisher_unmangles_function_tool_names_from_tool_bundle():
    adapter = AnthropicMessagesAdapter(client=object())
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )
    toolkit = Toolkit(
        name="search",
        tools=[_AnyArgsTool("lookup/web")],
    )
    tool_bundle = MessagesToolBundle(toolkits=[toolkit])
    adapter._set_function_tool_name_resolver(
        event_handler=publisher,
        resolver=tool_bundle.resolve_function_tool_name,
    )

    safe_name = safe_tool_name("lookup/web")
    publisher({"type": "message_start", "event": {"message": {"id": "msg_1"}}})
    publisher(
        {
            "type": "content_block_start",
            "event": {
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": safe_name,
                    "input": {"q": "meshagent"},
                },
            },
        }
    )
    publisher({"type": "content_block_stop", "event": {"index": 0}})
    publisher(
        {
            "type": "meshagent.handler.added",
            "item": {
                "type": "function_call",
                "id": "toolu_1",
                "call_id": "toolu_1",
                "name": safe_name,
                "arguments": '{"q":"meshagent"}',
            },
        }
    )
    publisher({"type": "meshagent.handler.done", "item_id": "toolu_1"})

    assert [type(event) for event in published] == [
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]
    pending = published[0]
    assert isinstance(pending, AgentToolCallPending)
    assert pending.toolkit == "search"
    assert pending.tool == "lookup/web"
    assert pending.arguments == {"q": "meshagent"}

    started = published[1]
    assert isinstance(started, AgentToolCallStarted)
    assert started.toolkit == "search"
    assert started.tool == "lookup/web"
    assert started.arguments == {"q": "meshagent"}


def test_make_agent_event_publisher_marks_anthropic_tool_failure() -> None:
    adapter = AnthropicMessagesAdapter(client=object())
    published: list[object] = []
    publisher = adapter.make_agent_event_publisher(
        turn_id="turn-1",
        thread_id="thread-1",
        callback=published.append,
    )

    publisher({"type": "message_start", "event": {"message": {"id": "msg_1"}}})
    publisher(
        {
            "type": "content_block_start",
            "event": {
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "write_file",
                    "input": {"path": "src/app.py"},
                },
            },
        }
    )
    publisher({"type": "content_block_stop", "event": {"index": 0}})
    publisher(
        {
            "type": "meshagent.handler.added",
            "item": {
                "type": "function_call",
                "id": "toolu_1",
                "call_id": "toolu_1",
                "name": "write_file",
                "arguments": '{"path":"src/app.py"}',
            },
        }
    )
    publisher(
        {
            "type": "meshagent.handler.done",
            "item_id": "toolu_1",
            "error": "'text' is a required property",
        }
    )

    assert [type(event) for event in published] == [
        AgentToolCallPending,
        AgentToolCallStarted,
        AgentToolCallEnded,
    ]
    ended = published[-1]
    assert isinstance(ended, AgentToolCallEnded)
    assert ended.error is not None
    assert ended.error.message == "'text' is a required property"

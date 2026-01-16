import pytest

from meshagent.anthropic.messages_adapter import AnthropicMessagesAdapter
from meshagent.agents.agent import AgentChatContext
from meshagent.tools import Tool, Toolkit
from meshagent.api import RoomException
from meshagent.agents.adapter import ToolResponseAdapter


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


class _AnyArgsTool(Tool):
    def __init__(self, name: str):
        super().__init__(
            name=name,
            input_schema={"type": "object", "additionalProperties": True},
            description="test tool",
        )

    async def execute(self, context, **kwargs):
        return {"ok": True, "args": kwargs}


class _ToolResultAdapter(ToolResponseAdapter):
    async def to_plain_text(self, *, room, response):
        return "ok"

    async def create_messages(self, *, context, tool_call, room, response):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": [{"type": "text", "text": "ok"}],
                    }
                ],
            }
        ]


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


def test_convert_messages_drops_assistant_between_tool_use_and_tool_result():
    ctx = AgentChatContext(
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
    ctx = AgentChatContext(
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
    ctx = AgentChatContext(system_role=None)
    ctx.append_user_message("run tools")

    toolkit = Toolkit(
        name="test",
        tools=[_AnyArgsTool("tool_a"), _AnyArgsTool("tool_b")],
    )

    result = await adapter.next(
        context=ctx,
        room=_DummyRoom(),
        toolkits=[toolkit],
        tool_adapter=_ToolResultAdapter(),
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

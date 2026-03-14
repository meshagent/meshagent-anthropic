import os
import sys
import asyncio
import copy

import pytest

from meshagent.anthropic.messages_adapter import AnthropicMessagesAdapter
from meshagent.anthropic.mcp import MCPConfig, MCPServer, MCPTool
from meshagent.agents.agent import AgentSessionContext
from meshagent.tools import FunctionTool, Toolkit


def _import_real_anthropic_sdk():
    """Import the external `anthropic` SDK without shadowing.

    If `pytest` is run from inside `.../meshagent/`, Python may resolve
    `import anthropic` to the local `meshagent/anthropic` package directory.
    """

    cwd = os.getcwd()

    if os.path.isdir(os.path.join(cwd, "anthropic")):
        sys.path = [p for p in sys.path if p not in ("", cwd)]

    import importlib

    mod = importlib.import_module("anthropic")

    mod_file = getattr(mod, "__file__", "") or ""
    if mod_file.endswith("/meshagent/anthropic/__init__.py"):
        raise RuntimeError(
            "Imported local `meshagent/anthropic` instead of the Anthropic SDK. "
            "Run pytest from the repo root."
        )

    return mod


a = _import_real_anthropic_sdk()


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


class _SteeringProbeTool(FunctionTool):
    def __init__(self):
        super().__init__(
            name="steering_probe",
            input_schema={
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                },
                "required": ["note"],
                "additionalProperties": False,
            },
            description="A probe tool used to verify steering order across tool calls.",
        )
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[str] = []

    async def execute(self, context, note: str) -> dict[str, object]:
        del context
        self.calls.append(note)
        self.started.set()
        await self.release.wait()
        return {"ok": True, "note": note}


class _RecordingAnthropicMessagesAdapter(AnthropicMessagesAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.recorded_requests: list[dict[str, object]] = []

    async def _stream_message(self, *, client, request: dict, event_handler):
        self.recorded_requests.append(copy.deepcopy(request))
        return await super()._stream_message(
            client=client,
            request=request,
            event_handler=event_handler,
        )


@pytest.mark.asyncio
async def test_live_anthropic_adapter_messages_create_if_key_set():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    model = os.getenv("ANTHROPIC_TEST_MODEL", "claude-sonnet-4-5")

    client = a.AsyncAnthropic(api_key=api_key)
    adapter = AnthropicMessagesAdapter(model=model, client=client, max_tokens=64)

    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message("Say hello in one word.")

    text = await adapter.next(context=ctx, room=_DummyRoom(), toolkits=[])

    assert isinstance(text, str)
    assert len(text.strip()) > 0


@pytest.mark.asyncio
async def test_live_anthropic_adapter_streaming_if_key_set():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    model = os.getenv("ANTHROPIC_TEST_MODEL", "claude-sonnet-4-5")

    client = a.AsyncAnthropic(api_key=api_key)
    adapter = AnthropicMessagesAdapter(model=model, client=client, max_tokens=64)

    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message("Count from 1 to 3.")

    seen_types: list[str] = []

    def handler(event: dict):
        if isinstance(event, dict) and "type" in event:
            seen_types.append(event["type"])

    text = await adapter.next(
        context=ctx,
        room=_DummyRoom(),
        toolkits=[],
        event_handler=handler,
    )

    assert isinstance(text, str)
    assert len(text.strip()) > 0
    # These are best-effort; event types depend on Anthropic SDK.
    assert len(seen_types) > 0


@pytest.mark.asyncio
async def test_live_anthropic_mcp_deepwiki_if_key_set():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    model = os.getenv("ANTHROPIC_TEST_MODEL", "claude-sonnet-4-5")

    client = a.AsyncAnthropic(api_key=api_key)
    adapter = AnthropicMessagesAdapter(model=model, client=client, max_tokens=256)

    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message(
        "Use the DeepWiki MCP toolset and make at least one tool call. "
        "Then reply with a one-sentence summary of what you learned."
    )

    mcp_toolkit = Toolkit(
        name="mcp",
        tools=[
            MCPTool(
                config=MCPConfig(
                    mcp_servers=[
                        MCPServer(url="https://mcp.deepwiki.com/mcp", name="deepwiki")
                    ]
                )
            )
        ],
    )

    seen_mcp_blocks = False

    def handler(event: dict):
        nonlocal seen_mcp_blocks
        if not isinstance(event, dict):
            return

        # Adapter forwards Anthropic SDK stream events:
        # {"type": "content_block_start", "event": {...}}
        if event.get("type") == "content_block_start":
            payload = event.get("event") or {}
            content_block = payload.get("content_block") or {}
            if content_block.get("type") in {"mcp_tool_use", "mcp_tool_result"}:
                seen_mcp_blocks = True

    text = await adapter.next(
        context=ctx,
        room=_DummyRoom(),
        toolkits=[mcp_toolkit],
        event_handler=handler,
    )

    assert isinstance(text, str)
    assert len(text.strip()) > 0

    # This asserts the connector actually engaged (best-effort, but should be stable
    # for DeepWiki).
    assert seen_mcp_blocks


@pytest.mark.asyncio
async def test_live_anthropic_adapter_compaction_if_key_set():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    model = os.getenv(
        "ANTHROPIC_COMPACTION_TEST_MODEL",
        os.getenv("ANTHROPIC_TEST_MODEL", "claude-sonnet-4-6"),
    )

    client = a.AsyncAnthropic(api_key=api_key)
    adapter = AnthropicMessagesAdapter(
        model=model,
        client=client,
        max_tokens=64,
        context_management="auto",
        compaction_threshold=50000,
    )

    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message(
        "Reply with the single word 'ready'. Then wait for the next message."
    )

    try:
        first = await adapter.next(context=ctx, room=_DummyRoom(), toolkits=[])
    except Exception as ex:
        message = str(ex)
        if "does not support the 'compact_20260112'" in message:
            pytest.skip(f"model {model} does not support compact_20260112")
        raise
    assert isinstance(first, str)

    ctx.append_user_message("Now reply with the single word 'done'.")
    second = await adapter.next(context=ctx, room=_DummyRoom(), toolkits=[])
    assert isinstance(second, str)
    assert len(second.strip()) > 0
    assert ctx.metadata.get("last_response_model") == model
    usage = ctx.metadata.get("last_response_usage")
    assert isinstance(usage, dict)
    assert int(usage.get("input_tokens", 0)) > 0


@pytest.mark.asyncio
async def test_live_anthropic_inserts_steer_immediately_after_tool_results_if_key_set():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    model = os.getenv("ANTHROPIC_STEERING_TEST_MODEL", "claude-sonnet-4-6")

    client = a.AsyncAnthropic(api_key=api_key)
    adapter = _RecordingAnthropicMessagesAdapter(model=model, client=client)
    tool = _SteeringProbeTool()

    ctx = AgentSessionContext(system_role=None)
    ctx.append_user_message(
        "You must call the steering_probe tool exactly once with any note string. "
        "After the tool result is available, reply with exactly ORIGINAL and nothing else."
    )

    pending_steer = False

    async def _steer() -> bool:
        nonlocal pending_steer
        if not pending_steer:
            return False
        pending_steer = False
        ctx.append_user_message(
            "New instruction: after the tool result, reply with exactly STEERED and nothing else."
        )
        return True

    task = asyncio.create_task(
        adapter.next(
            context=ctx,
            room=_DummyRoom(),
            toolkits=[Toolkit(name="test", tools=[tool])],
            event_handler=lambda event: None,
            steering_callback=_steer,
        )
    )

    await asyncio.wait_for(tool.started.wait(), timeout=30.0)
    pending_steer = True
    tool.release.set()
    result = await asyncio.wait_for(task, timeout=90.0)

    assert tool.calls
    assert "STEERED" in result
    assert "ORIGINAL" not in result
    assert len(adapter.recorded_requests) >= 2

    second_messages = adapter.recorded_requests[1]["messages"]
    assert isinstance(second_messages, list)
    assert len(second_messages) >= 4
    assert second_messages[-3]["role"] == "assistant"
    assert second_messages[-2]["role"] == "user"
    assert second_messages[-2]["content"][0]["type"] == "tool_result"
    assert second_messages[-1] == {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "New instruction: after the tool result, reply with exactly "
                    "STEERED and nothing else."
                ),
            }
        ],
    }
    assert not any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("content"), list)
        and any(
            isinstance(block, dict)
            and block.get("type") == "text"
            and "ORIGINAL" in block.get("text", "")
            for block in message["content"]
        )
        for message in second_messages[:-1]
    )

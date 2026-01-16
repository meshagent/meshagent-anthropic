import os
import sys

import pytest

from meshagent.anthropic.messages_adapter import AnthropicMessagesAdapter
from meshagent.anthropic.mcp import MCPConfig, MCPServer, MCPTool
from meshagent.agents.agent import AgentChatContext
from meshagent.tools import Toolkit


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


class _DummyRoom:
    # Adapter won't touch room when no tools.
    pass


@pytest.mark.asyncio
async def test_live_anthropic_adapter_messages_create_if_key_set():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    model = os.getenv("ANTHROPIC_TEST_MODEL", "claude-sonnet-4-5")

    client = a.AsyncAnthropic(api_key=api_key)
    adapter = AnthropicMessagesAdapter(model=model, client=client, max_tokens=64)

    ctx = AgentChatContext(system_role=None)
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

    ctx = AgentChatContext(system_role=None)
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

    ctx = AgentChatContext(system_role=None)
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

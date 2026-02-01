from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from meshagent.tools import Toolkit, ToolkitBuilder, ToolkitConfig

from .request_tool import AnthropicRequestTool


# This module wraps Anthropic's official MCP connector support:
# https://platform.claude.com/docs/en/agents-and-tools/mcp-connector


MCP_CONNECTOR_BETA = "mcp-client-2025-11-20"


class MCPServer(BaseModel):
    """Anthropic `mcp_servers` entry."""

    type: Literal["url"] = "url"
    url: str
    name: str
    authorization_token: Optional[str] = None


class MCPToolConfig(BaseModel):
    enabled: Optional[bool] = None
    defer_loading: Optional[bool] = None


class MCPToolset(BaseModel):
    """Anthropic `tools` entry for MCP connector."""

    type: Literal["mcp_toolset"] = "mcp_toolset"
    mcp_server_name: str
    default_config: Optional[MCPToolConfig] = None
    configs: Optional[dict[str, MCPToolConfig]] = None

    # Pass-through cache control, if desired.
    cache_control: Optional[dict] = None


class MCPConfig(ToolkitConfig):
    """MeshAgent toolkit config that injects MCP connector params.

    This is intentionally modeled after the OpenAI adapter's MCP config pattern
    (a toolkit config that can be provided via `tools=[...]` in chat messages),
    but it produces Anthropic-specific request parameters: `mcp_servers` and
    `mcp_toolset` entries.
    """

    name: Literal["mcp"] = "mcp"

    mcp_servers: list[MCPServer]
    toolsets: Optional[list[MCPToolset]] = None
    betas: list[str] = [MCP_CONNECTOR_BETA]


class MCPTool(AnthropicRequestTool):
    """Non-executable tool that augments the Anthropic request."""

    def __init__(self, *, config: MCPConfig):
        super().__init__(name="mcp")
        self.config = config

    def apply(self, *, request: dict, headers: dict) -> None:
        """Mutate an Anthropic Messages request in-place."""

        self.apply_betas(headers=headers, betas=self.config.betas)

        toolsets = self.config.toolsets
        if toolsets is None:
            toolsets = [
                MCPToolset(mcp_server_name=s.name) for s in self.config.mcp_servers
            ]

        # Merge/dedupe servers by name.
        existing_servers = request.setdefault("mcp_servers", [])
        dedup: dict[str, dict] = {
            s["name"]: s
            for s in existing_servers
            if isinstance(s, dict) and isinstance(s.get("name"), str)
        }
        for server in self.config.mcp_servers:
            dedup[server.name] = server.model_dump(mode="json", exclude_none=True)
        request["mcp_servers"] = list(dedup.values())

        # Anthropic MCP toolsets live inside the top-level `tools` array.
        tools = request.setdefault("tools", [])
        for toolset in toolsets:
            tools.append(toolset.model_dump(mode="json", exclude_none=True))


class MCPToolkitBuilder(ToolkitBuilder):
    def __init__(self):
        super().__init__(name="mcp", type=MCPConfig)

    async def make(self, *, room, model: str, config: MCPConfig) -> Toolkit:
        return Toolkit(name="mcp", tools=[MCPTool(config=config)])

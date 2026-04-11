from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

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


class MCPTool(AnthropicRequestTool):
    """Non-executable tool that augments the Anthropic request."""

    def __init__(
        self,
        *,
        mcp_servers: list[MCPServer],
        toolsets: list[MCPToolset] | None = None,
        betas: list[str] | None = None,
        name: str = "mcp",
    ):
        super().__init__(name=name)
        self.mcp_servers = mcp_servers
        self.toolsets = toolsets
        self.betas = [MCP_CONNECTOR_BETA] if betas is None else list(betas)

    def apply(self, *, request: dict, headers: dict) -> None:
        """Mutate an Anthropic Messages request in-place."""

        self.apply_betas(headers=headers, betas=self.betas)

        toolsets = self.toolsets
        if toolsets is None:
            toolsets = [MCPToolset(mcp_server_name=s.name) for s in self.mcp_servers]

        # Merge/dedupe servers by name.
        existing_servers = request.setdefault("mcp_servers", [])
        dedup: dict[str, dict] = {
            s["name"]: s
            for s in existing_servers
            if isinstance(s, dict) and isinstance(s.get("name"), str)
        }
        for server in self.mcp_servers:
            dedup[server.name] = server.model_dump(mode="json", exclude_none=True)
        request["mcp_servers"] = list(dedup.values())

        # Anthropic MCP toolsets live inside the top-level `tools` array.
        tools = request.setdefault("tools", [])
        for toolset in toolsets:
            tools.append(toolset.model_dump(mode="json", exclude_none=True))

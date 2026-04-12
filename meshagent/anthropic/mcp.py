from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from meshagent.agents.mcp import MCPServerConfig, MCPToolkitClientOptions
from meshagent.tools import BaseTool, Toolkit

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


def _merge_mcp_server_configs(
    *,
    static_servers: list[MCPServerConfig],
    client_options: dict | None,
) -> list[MCPServerConfig]:
    merged_by_label: dict[str, MCPServerConfig] = {
        server.server_label: server for server in static_servers
    }
    if client_options is None:
        return list(merged_by_label.values())

    options = MCPToolkitClientOptions.model_validate(client_options)
    for server in options.servers:
        merged_by_label[server.server_label] = server
    return list(merged_by_label.values())


class AnthropicMessagesMCPToolkit(Toolkit):
    def __init__(
        self,
        *,
        servers: list[MCPServerConfig] | None = None,
        title: str | None = None,
        description: str | None = None,
        hidden: bool = False,
    ) -> None:
        super().__init__(
            name="mcp",
            title=title,
            description=description,
            tools=[],
            client_options=MCPToolkitClientOptions.model_json_schema(),
            hidden=hidden,
        )
        self._servers = [*(servers or [])]

    def get_tools(self, *, client_options: dict | None = None) -> list[BaseTool]:
        servers = _merge_mcp_server_configs(
            static_servers=self._servers,
            client_options=client_options,
        )
        resolved_tools: list[BaseTool] = []
        for server in servers:
            if server.server_url is None or server.server_url.strip() == "":
                continue
            resolved_tools.append(
                MCPTool(
                    name=server.server_label,
                    mcp_servers=[
                        MCPServer(
                            url=server.server_url,
                            name=server.server_label,
                            authorization_token=server.authorization,
                        )
                    ],
                )
            )
        return resolved_tools

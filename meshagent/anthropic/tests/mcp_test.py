from meshagent.anthropic.mcp import MCPConfig, MCPServer, MCPTool, MCPToolset


def test_mcp_tool_apply_injects_servers_toolsets_and_beta():
    cfg = MCPConfig(
        mcp_servers=[
            MCPServer(url="https://mcp.example.com/sse", name="example-mcp"),
        ],
        toolsets=[MCPToolset(mcp_server_name="example-mcp")],
    )

    tool = MCPTool(config=cfg)
    request: dict = {"tools": []}
    tool.apply(request=request)

    assert request["betas"] == ["mcp-client-2025-11-20"]
    assert request["mcp_servers"] == [
        {
            "type": "url",
            "url": "https://mcp.example.com/sse",
            "name": "example-mcp",
        }
    ]

    assert request["tools"][0]["type"] == "mcp_toolset"
    assert request["tools"][0]["mcp_server_name"] == "example-mcp"


def test_mcp_tool_apply_dedupes_servers_by_name_and_preserves_existing():
    cfg = MCPConfig(
        mcp_servers=[
            MCPServer(url="https://mcp.example.com/sse", name="example-mcp"),
            MCPServer(url="https://mcp.other.com/sse", name="other"),
        ]
    )
    tool = MCPTool(config=cfg)

    request: dict = {
        "tools": [{"type": "tool", "name": "some_tool"}],
        "mcp_servers": [
            {
                "type": "url",
                "url": "https://old.example.com/sse",
                "name": "example-mcp",
                "authorization_token": "OLD",
            }
        ],
        "betas": ["some-other-beta"],
    }

    tool.apply(request=request)

    # Keeps existing betas and appends MCP beta.
    assert "some-other-beta" in request["betas"]
    assert "mcp-client-2025-11-20" in request["betas"]

    # Dedupes by name; cfg overwrites the existing server entry.
    by_name = {s["name"]: s for s in request["mcp_servers"]}
    assert set(by_name.keys()) == {"example-mcp", "other"}
    assert by_name["example-mcp"]["url"] == "https://mcp.example.com/sse"

    # If toolsets omitted, it creates one per server.
    mcp_toolsets = [t for t in request["tools"] if t.get("type") == "mcp_toolset"]
    assert {t["mcp_server_name"] for t in mcp_toolsets} == {"example-mcp", "other"}

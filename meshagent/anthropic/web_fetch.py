from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from meshagent.tools import Toolkit, ToolkitBuilder, ToolkitConfig

from .request_tool import AnthropicRequestTool


class WebFetchCitations(BaseModel):
    enabled: bool = True


class WebFetchConfig(ToolkitConfig):
    name: Literal["web_fetch"] = "web_fetch"
    max_uses: Optional[int] = None
    allowed_domains: Optional[list[str]] = None
    blocked_domains: Optional[list[str]] = None
    citations: Optional[WebFetchCitations] = None
    max_content_tokens: Optional[int] = None
    betas: Optional[list[str]] = Field(default_factory=lambda: ["web-fetch-2025-09-10"])


class WebFetchTool(AnthropicRequestTool):
    def __init__(self, *, config: Optional[WebFetchConfig] = None):
        if config is None:
            config = WebFetchConfig(name="web_fetch")
        super().__init__(name=config.name)
        self.config = config

    def apply(self, *, request: dict, headers: dict) -> None:
        if self.config.allowed_domains and self.config.blocked_domains:
            raise ValueError(
                "web_fetch cannot set both allowed_domains and blocked_domains"
            )

        tools = request.setdefault("tools", [])
        tool_def: dict[str, object] = {
            "type": "web_fetch_20250910",
            "name": self.config.name,
        }
        if self.config.max_uses is not None:
            tool_def["max_uses"] = self.config.max_uses
        if self.config.allowed_domains is not None:
            tool_def["allowed_domains"] = self.config.allowed_domains
        if self.config.blocked_domains is not None:
            tool_def["blocked_domains"] = self.config.blocked_domains
        if self.config.citations is not None:
            tool_def["citations"] = self.config.citations.model_dump(
                mode="json", exclude_none=True
            )
        if self.config.max_content_tokens is not None:
            tool_def["max_content_tokens"] = self.config.max_content_tokens

        if not any(
            isinstance(t, dict)
            and t.get("type") == tool_def["type"]
            and t.get("name") == tool_def["name"]
            for t in tools
        ):
            tools.append(tool_def)

        self.apply_betas(headers=headers, betas=self.config.betas)


class WebFetchToolkitBuilder(ToolkitBuilder):
    def __init__(self):
        super().__init__(name="web_fetch", type=WebFetchConfig)

    async def make(self, *, room, model: str, config: WebFetchConfig) -> Toolkit:
        return Toolkit(name="web_fetch", tools=[WebFetchTool(config=config)])

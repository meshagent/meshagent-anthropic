from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from meshagent.tools import BaseTool, Toolkit, ToolkitBuilder, ToolkitConfig


class WebSearchConfig(ToolkitConfig):
    name: Literal["web_search"] = "web_search"
    max_uses: Optional[int] = None
    allowed_domains: Optional[list[str]] = None
    blocked_domains: Optional[list[str]] = None
    user_location: Optional["WebSearchUserLocation"] = None
    betas: Optional[list[str]] = None


class WebSearchUserLocation(BaseModel):
    type: Literal["approximate"] = "approximate"
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None


class WebSearchTool(BaseTool):
    def __init__(self, *, config: Optional[WebSearchConfig] = None):
        if config is None:
            config = WebSearchConfig(name="web_search")
        super().__init__(name=config.name)
        self.config = config

    def apply(self, *, request: dict) -> None:
        if self.config.allowed_domains and self.config.blocked_domains:
            raise ValueError(
                "web_search cannot set both allowed_domains and blocked_domains"
            )

        tools = request.setdefault("tools", [])
        tool_def: dict[str, object] = {
            "type": "web_search_20250305",
            "name": self.config.name,
        }
        if self.config.max_uses is not None:
            tool_def["max_uses"] = self.config.max_uses
        if self.config.allowed_domains is not None:
            tool_def["allowed_domains"] = self.config.allowed_domains
        if self.config.blocked_domains is not None:
            tool_def["blocked_domains"] = self.config.blocked_domains
        if self.config.user_location is not None:
            tool_def["user_location"] = self.config.user_location.model_dump(
                mode="json", exclude_none=True
            )
        if not any(
            isinstance(t, dict)
            and t.get("type") == tool_def["type"]
            and t.get("name") == tool_def["name"]
            for t in tools
        ):
            tools.append(tool_def)

        if self.config.betas:
            betas = request.setdefault("betas", [])
            for b in self.config.betas:
                if b not in betas:
                    betas.append(b)


class WebSearchToolkitBuilder(ToolkitBuilder):
    def __init__(self):
        super().__init__(name="web_search", type=WebSearchConfig)

    async def make(self, *, room, model: str, config: WebSearchConfig) -> Toolkit:
        return Toolkit(name="web_search", tools=[WebSearchTool(config=config)])

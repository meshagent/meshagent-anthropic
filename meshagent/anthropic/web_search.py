from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from .request_tool import AnthropicRequestTool


class WebSearchUserLocation(BaseModel):
    type: Literal["approximate"] = "approximate"
    city: Optional[str] = None
    region: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None


class WebSearchTool(AnthropicRequestTool):
    def __init__(
        self,
        *,
        name: str = "web_search",
        max_uses: int | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        user_location: WebSearchUserLocation | None = None,
        betas: list[str] | None = None,
    ):
        super().__init__(name=name)
        self.max_uses = max_uses
        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        self.user_location = user_location
        self.betas = betas

    def apply(self, *, request: dict, headers: dict) -> None:
        if self.allowed_domains and self.blocked_domains:
            raise ValueError(
                "web_search cannot set both allowed_domains and blocked_domains"
            )

        tools = request.setdefault("tools", [])
        tool_def: dict[str, object] = {
            "type": "web_search_20250305",
            "name": self.name,
        }
        if self.max_uses is not None:
            tool_def["max_uses"] = self.max_uses
        if self.allowed_domains is not None:
            tool_def["allowed_domains"] = self.allowed_domains
        if self.blocked_domains is not None:
            tool_def["blocked_domains"] = self.blocked_domains
        if self.user_location is not None:
            tool_def["user_location"] = self.user_location.model_dump(
                mode="json", exclude_none=True
            )
        if not any(
            isinstance(t, dict)
            and t.get("type") == tool_def["type"]
            and t.get("name") == tool_def["name"]
            for t in tools
        ):
            tools.append(tool_def)

        self.apply_betas(headers=headers, betas=self.betas)

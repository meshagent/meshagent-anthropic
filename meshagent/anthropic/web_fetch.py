from __future__ import annotations

from pydantic import BaseModel

from .request_tool import AnthropicRequestTool


class WebFetchCitations(BaseModel):
    enabled: bool = True


class WebFetchTool(AnthropicRequestTool):
    def __init__(
        self,
        *,
        name: str = "web_fetch",
        max_uses: int | None = None,
        allowed_domains: list[str] | None = None,
        blocked_domains: list[str] | None = None,
        citations: WebFetchCitations | None = None,
        max_content_tokens: int | None = None,
        betas: list[str] | None = None,
    ):
        super().__init__(name=name)
        self.max_uses = max_uses
        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains
        self.citations = citations
        self.max_content_tokens = max_content_tokens
        self.betas = ["web-fetch-2025-09-10"] if betas is None else list(betas)

    def apply(self, *, request: dict, headers: dict) -> None:
        if self.allowed_domains and self.blocked_domains:
            raise ValueError(
                "web_fetch cannot set both allowed_domains and blocked_domains"
            )

        tools = request.setdefault("tools", [])
        tool_def: dict[str, object] = {
            "type": "web_fetch_20250910",
            "name": self.name,
        }
        if self.max_uses is not None:
            tool_def["max_uses"] = self.max_uses
        if self.allowed_domains is not None:
            tool_def["allowed_domains"] = self.allowed_domains
        if self.blocked_domains is not None:
            tool_def["blocked_domains"] = self.blocked_domains
        if self.citations is not None:
            tool_def["citations"] = self.citations.model_dump(
                mode="json", exclude_none=True
            )
        if self.max_content_tokens is not None:
            tool_def["max_content_tokens"] = self.max_content_tokens

        if not any(
            isinstance(t, dict)
            and t.get("type") == tool_def["type"]
            and t.get("name") == tool_def["name"]
            for t in tools
        ):
            tools.append(tool_def)

        self.apply_betas(headers=headers, betas=self.betas)

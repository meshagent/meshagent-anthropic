from __future__ import annotations

from typing import Optional

from meshagent.tools import BaseTool


class AnthropicRequestTool(BaseTool):
    beta_header = "anthropic-beta"

    @staticmethod
    def apply_betas(*, headers: dict, betas: Optional[list[str]]) -> None:
        if not betas:
            return

        existing = headers.get(AnthropicRequestTool.beta_header)
        existing_betas: list[str] = []

        if isinstance(existing, str):
            existing_betas = [b.strip() for b in existing.split(",") if b.strip()]
        elif isinstance(existing, list):
            existing_betas = [str(b) for b in existing if str(b)]

        for beta in betas:
            if beta not in existing_betas:
                existing_betas.append(beta)

        headers[AnthropicRequestTool.beta_header] = ",".join(existing_betas)

import logging
import json
import httpx
import os
from typing import Optional, Any

try:
    from anthropic import AsyncAnthropic
except Exception:  # pragma: no cover
    AsyncAnthropic = None  # type: ignore

logger = logging.getLogger("anthropic.client")


def _redact_headers(headers: httpx.Headers) -> dict:
    h = dict(headers)
    if "x-api-key" in {k.lower() for k in h.keys()}:
        for k in list(h.keys()):
            if k.lower() == "x-api-key":
                h[k] = "***REDACTED***"
    if "authorization" in {k.lower() for k in h.keys()}:
        for k in list(h.keys()):
            if k.lower() == "authorization":
                h[k] = "***REDACTED***"
    return h


def _truncate_bytes(b: bytes, limit: int = 1024 * 100) -> str:
    s = b.decode("utf-8", errors="replace")
    return (
        s
        if len(s) <= limit
        else (s[:limit] + f"\n... (truncated, {len(s)} chars total)")
    )


async def log_request(request: httpx.Request):
    logging.info("==> %s %s", request.method, request.url)
    logging.info("headers=%s", json.dumps(_redact_headers(request.headers), indent=2))
    if request.content:
        logging.info("body=%s", _truncate_bytes(request.content))


async def log_response(response: httpx.Response):
    body = await response.aread()
    logging.info("<== %s %s", response.status_code, response.request.url)
    logging.info("headers=%s", json.dumps(_redact_headers(response.headers), indent=2))
    if body:
        logging.info("body=%s", _truncate_bytes(body))


def get_logging_httpx_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        event_hooks={"request": [log_request], "response": [log_response]},
        timeout=60.0,
    )


def get_client(
    *,
    base_url: str | None = None,
    http_client: Optional[httpx.AsyncClient] = None,
    api_key: str | None = None,
) -> Any:
    if AsyncAnthropic is None:  # pragma: no cover
        raise RuntimeError(
            "anthropic is not installed. Install `meshagent-anthropic` extras/deps."
        )
    if base_url is None:
        base_url = os.getenv("ANTHROPIC_BASE_URL")
    if base_url is not None:
        base_url = base_url.strip() or None
    kwargs: dict[str, object] = {}
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        kwargs["base_url"] = base_url
    if http_client is not None:
        kwargs["http_client"] = http_client
    return AsyncAnthropic(**kwargs)

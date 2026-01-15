from meshagent.api import RoomClient
import logging
import json
import httpx
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


def _truncate_bytes(b: bytes, limit: int = 4000) -> str:
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
    *, room: RoomClient, http_client: Optional[httpx.AsyncClient] = None
) -> Any:
    if AsyncAnthropic is None:  # pragma: no cover
        raise RuntimeError(
            "anthropic is not installed. Install `meshagent-anthropic` extras/deps."
        )

    token: str = room.protocol.token

    url = getattr(room.protocol, "url")
    if url is None:
        logger.debug(
            "protocol does not have url, anthropic client falling back to room url %s",
            room.room_url,
        )
        url = room.room_url
    else:
        logger.debug("protocol had url, anthropic client will use %s", url)

    room_proxy_url = f"{url}/anthropic"

    if room_proxy_url.startswith("ws:") or room_proxy_url.startswith("wss:"):
        room_proxy_url = room_proxy_url.replace("ws", "http", 1)

    # The MeshAgent room proxy validates `x-api-key` and `Meshagent-Session`.
    return AsyncAnthropic(
        api_key=token,
        base_url=room_proxy_url,
        http_client=http_client,
        default_headers={"Meshagent-Session": room.session_id},
    )

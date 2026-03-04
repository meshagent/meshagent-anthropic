from __future__ import annotations

from typing import Any
import logging

from pydantic import BaseModel

logger = logging.getLogger("anthropic_agent")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def _flatten_usage(usage: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}

    for key, value in usage.items():
        try:
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, dict):
                        continue
                    nested_number = _to_float(nested_value)
                    if nested_number is not None:
                        flattened[nested_key] = nested_number
                continue

            number = _to_float(value)
            if number is not None:
                flattened[key] = number
        except Exception as error:
            logger.warning("unexpected usage key %s:%s", key, value, exc_info=error)

    return flattened


def split_anthropic_usage_by_tier(
    *, model: str, usage: dict[str, float]
) -> dict[str, float]:
    def total_input_tokens(current: dict[str, float]) -> float:
        return (
            float(current.get("input_tokens", 0))
            + float(current.get("cache_creation_input_tokens", 0))
            + float(current.get("cache_read_input_tokens", 0))
        )

    tiered_prefixes = (
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4",
        "claude-sonnet-4-0",
    )

    if not model.startswith(tiered_prefixes):
        return dict(usage)

    if total_input_tokens(usage) <= 200_000:
        return dict(usage)

    split_usage = dict(usage)

    def move(token_key: str) -> None:
        if token_key in split_usage:
            split_usage[f"{token_key}_long"] = float(split_usage.pop(token_key))

    move("input_tokens")
    move("output_tokens")
    move("cache_creation_input_tokens")
    move("cache_read_input_tokens")

    return split_usage


def normalize_anthropic_usage(usage: object) -> dict[str, Any] | None:
    if usage is None:
        return None

    if isinstance(usage, BaseModel):
        try:
            return usage.model_dump(mode="json")
        except Exception:
            return None

    if isinstance(usage, dict):
        return usage

    return None


def preprocess_anthropic_usage(
    *, model: str, usage: dict[str, Any]
) -> dict[str, float] | None:
    if not isinstance(usage, dict):
        return None

    usage_copy = {**usage}

    service_tier = usage_copy.pop("service_tier", None)
    if service_tier is not None and service_tier != "standard":
        logger.warning("custom pricing tier is in use, ignoring custom tier")

    flattened = _flatten_usage(usage_copy)
    return split_anthropic_usage_by_tier(model=model, usage=flattened)


def add_usage_metrics(*, totals: dict[str, float], usage: dict[str, float]) -> None:
    for key, value in usage.items():
        totals[key] = float(totals.get(key, 0.0)) + float(value)


# Backwards compatibility for older imports.
_preprocess_anthropic = preprocess_anthropic_usage

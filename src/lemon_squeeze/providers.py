"""Provider discovery — pings LM Studio + OpenRouter to surface available models.

The /v1/models endpoint is part of the OpenAI-compatible spec; LM Studio
implements it as the list of models you've downloaded (and that are loadable
on the running server), OpenRouter as their full catalog with metadata.

This module exists so the user can answer "what can I actually call right now"
without browsing dashboards. `lemon providers list` shows the available models
side-by-side; `lemon providers sync` registers all locally-loaded LM Studio
models into the DB in one shot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from lemon_squeeze.config import settings


@dataclass
class DiscoveredModel:
    provider: str           # "lm_studio" | "openrouter"
    name: str               # the API id we'd use to call it
    family: str | None = None
    context_window: int | None = None
    cost_in_per_mtok: float | None = None
    cost_out_per_mtok: float | None = None
    size_params_b: float | None = None
    raw: dict[str, Any] | None = None


def list_lm_studio_models(
    *, base_url: str | None = None, timeout: float = 5.0
) -> list[DiscoveredModel]:
    """GET <lm_studio_base_url>/models. Returns currently-loaded LM Studio models.

    Raises httpx.HTTPError on connection failure — callers should catch and
    treat as "LM Studio not running."
    """
    url = (base_url or settings.lm_studio_base_url).rstrip("/") + "/models"
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    out: list[DiscoveredModel] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str):
            continue
        out.append(
            DiscoveredModel(
                provider="lm_studio",
                name=model_id,
                family=_guess_family(model_id),
                # LM Studio's /models doesn't advertise context window or size,
                # so leave those None — the user can fill them in with
                # `lemon models register --size-b ... --ctx ...`.
                raw=entry,
            )
        )
    return out


def list_openrouter_models(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> list[DiscoveredModel]:
    """GET <openrouter_base_url>/models. Returns OpenRouter's full catalog."""
    url = (base_url or settings.openrouter_base_url).rstrip("/") + "/models"
    headers = {}
    key = api_key or settings.openrouter_api_key
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with httpx.Client(timeout=timeout) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []

    out: list[DiscoveredModel] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not isinstance(model_id, str):
            continue
        # Pricing can come back as None, a dict, or (rarely) a string like
        # "free". Anything that isn't a dict has no useful per-axis pricing.
        pricing = entry.get("pricing")
        if not isinstance(pricing, dict):
            pricing = {}
        # OpenRouter reports per-token cost; convert to per-million-tokens.
        cost_in = _parse_price_per_mtok(pricing.get("prompt"))
        cost_out = _parse_price_per_mtok(pricing.get("completion"))
        ctx_raw = entry.get("context_length")
        # context_length is normally int but providers occasionally hand back
        # numeric strings or floats; coerce defensively.
        ctx: int | None = None
        if isinstance(ctx_raw, bool):
            pass  # bool is technically int subclass — reject
        elif isinstance(ctx_raw, int):
            ctx = ctx_raw
        elif isinstance(ctx_raw, float) and ctx_raw.is_integer():
            ctx = int(ctx_raw)
        elif isinstance(ctx_raw, str):
            try:
                ctx = int(ctx_raw)
            except ValueError:
                pass
        out.append(
            DiscoveredModel(
                provider="openrouter",
                name=model_id,
                family=_guess_family(model_id),
                context_window=ctx,
                cost_in_per_mtok=cost_in,
                cost_out_per_mtok=cost_out,
                raw=entry,
            )
        )
    return out


def _parse_price_per_mtok(value: Any) -> float | None:
    """OpenRouter reports pricing as a string per-token rate. Convert to per-mtok."""
    if value is None:
        return None
    try:
        per_token = float(value)
    except (TypeError, ValueError):
        return None
    return per_token * 1_000_000


def _guess_family(name: str) -> str | None:
    """`anthropic/claude-sonnet-4-6` → 'claude'. See utils.split_provider_family."""
    from lemon_squeeze.utils import split_provider_family

    return split_provider_family(name)[1]

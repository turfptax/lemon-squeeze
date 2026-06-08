"""Unified chat-completion client.

LM Studio and OpenRouter both speak the OpenAI chat-completions dialect, so a
single client class handles both — only the base URL, API key, and a few cost
defaults differ. The client returns a `ChatResult` with the response text plus
usage metadata (tokens, latency, cost estimate) ready to drop into a `Run` row.

Design choice: we deliberately don't depend on the `openai` Python package.
A thin httpx call is enough, avoids the ~300 line OpenAI SDK init dance, and
keeps LM Studio working even when the SDK adds incompatible defaults.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from lemon_squeeze.config import settings

Provider = Literal["lm_studio", "openrouter"]


@dataclass
class ChatResult:
    text: str
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class ChatClient:
    """OpenAI-compatible chat-completions client for LM Studio and OpenRouter."""

    def __init__(
        self,
        provider: Provider,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.provider = provider
        if provider == "lm_studio":
            self.base_url = base_url or settings.lm_studio_base_url
            self.api_key = api_key or settings.lm_studio_api_key
        elif provider == "openrouter":
            self.base_url = base_url or settings.openrouter_base_url
            self.api_key = api_key or settings.openrouter_api_key or ""
        else:
            raise ValueError(f"unknown provider: {provider}")
        self.timeout = timeout

    def chat(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        cost_in_per_mtok: float | None = None,
        cost_out_per_mtok: float | None = None,
    ) -> ChatResult:
        """Single-turn completion. Returns a ChatResult with usage filled in."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        start = time.perf_counter()
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
        latency_ms = int((time.perf_counter() - start) * 1000)

        text = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage") or {}
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        cost_usd = _estimate_cost(tokens_in, tokens_out, cost_in_per_mtok, cost_out_per_mtok)

        return ChatResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            raw=data,
        )


def _estimate_cost(
    tokens_in: int | None,
    tokens_out: int | None,
    cost_in_per_mtok: float | None,
    cost_out_per_mtok: float | None,
) -> float | None:
    if tokens_in is None or tokens_out is None:
        return None
    if cost_in_per_mtok is None and cost_out_per_mtok is None:
        return None
    cost = 0.0
    if cost_in_per_mtok is not None:
        cost += (tokens_in / 1_000_000) * cost_in_per_mtok
    if cost_out_per_mtok is not None:
        cost += (tokens_out / 1_000_000) * cost_out_per_mtok
    return cost

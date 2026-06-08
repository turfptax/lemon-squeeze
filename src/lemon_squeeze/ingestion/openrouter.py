"""Ingest prompts from OpenRouter's generation history API.

OpenRouter exposes `/api/v1/generation?id=...` and `/api/v1/generations` for
account-level history. Since the listing endpoint is paginated and rate-limited,
this ingester accepts either:
  - a JSON file you've already downloaded (offline mode), or
  - live mode where it pages through the API using `OPENROUTER_API_KEY`.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from lemon_squeeze.config import settings
from lemon_squeeze.ingestion.base import Ingester, RawPrompt


class OpenRouterIngester(Ingester):
    source_name = "openrouter"

    def __init__(
        self,
        *,
        history_file: Path | None = None,
        since: timedelta | None = None,
        page_size: int = 100,
        max_pages: int = 20,
    ) -> None:
        self.history_file = Path(history_file) if history_file else None
        self.since = since
        self.page_size = page_size
        self.max_pages = max_pages

    def iter_prompts(self) -> Iterator[RawPrompt]:
        records = self._load_records()
        cutoff = datetime.now(timezone.utc) - self.since if self.since else None
        for rec in records:
            created = _parse_ts(rec.get("created_at") or rec.get("created"))
            if cutoff and created and created < cutoff:
                continue
            prompt_text = self._extract_prompt(rec)
            if not prompt_text:
                continue
            yield RawPrompt(
                content=prompt_text,
                source=self.source_name,
                source_ref=str(rec.get("id") or rec.get("generation_id") or ""),
                created_at=created,
                metadata={
                    "model": rec.get("model"),
                    "tokens_prompt": rec.get("tokens_prompt") or rec.get("native_tokens_prompt"),
                    "tokens_completion": rec.get("tokens_completion")
                    or rec.get("native_tokens_completion"),
                    "total_cost": rec.get("total_cost"),
                },
            )

    def _load_records(self) -> list[dict[str, Any]]:
        if self.history_file:
            data = json.loads(self.history_file.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else data.get("data", [])
        if not settings.openrouter_api_key:
            return []
        return list(self._fetch_history())

    def _fetch_history(self) -> Iterator[dict[str, Any]]:
        headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
        url = f"{settings.openrouter_base_url}/generations"
        with httpx.Client(timeout=30.0) as client:
            offset = 0
            for _ in range(self.max_pages):
                resp = client.get(
                    url,
                    headers=headers,
                    params={"limit": self.page_size, "offset": offset},
                )
                resp.raise_for_status()
                payload = resp.json()
                records = payload.get("data") if isinstance(payload, dict) else payload
                if not records:
                    return
                yield from records
                if len(records) < self.page_size:
                    return
                offset += self.page_size

    @staticmethod
    def _extract_prompt(rec: dict[str, Any]) -> str | None:
        # OpenRouter generation records sometimes contain the original prompt,
        # sometimes only token counts. Try the obvious fields.
        for key in ("prompt", "input", "user_prompt"):
            v = rec.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        messages = rec.get("messages")
        if isinstance(messages, list):
            user_parts = [
                m.get("content")
                for m in messages
                if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
            ]
            joined = "\n".join(user_parts).strip()
            if joined:
                return joined
        return None


def _parse_ts(v: Any) -> datetime | None:
    if isinstance(v, (int, float)):
        seconds = v / 1000 if v > 10_000_000_000 else v
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None

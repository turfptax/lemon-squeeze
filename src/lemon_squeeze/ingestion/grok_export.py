"""Ingest prompts from a Grok export.

Grok has no fully standardized export format yet; common shapes seen in the wild:
  - JSON array of conversations with `messages: [{role, content, timestamp}]`
  - JSON with `conversations: [...]`
  - A directory of per-conversation JSON files

This ingester handles all three shapes leniently.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from lemon_squeeze.ingestion.base import Ingester, RawPrompt


class GrokExportIngester(Ingester):
    source_name = "grok_export"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def iter_prompts(self) -> Iterator[RawPrompt]:
        if self.path.is_dir():
            for fp in self.path.rglob("*.json"):
                yield from self._iter_file(fp)
        else:
            yield from self._iter_file(self.path)

    def _iter_file(self, fp: Path) -> Iterator[RawPrompt]:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        conversations = self._normalize_conversations(data)
        for conv in conversations:
            conv_id = conv.get("id") or conv.get("conversation_id") or fp.stem
            for idx, msg in enumerate(conv.get("messages", [])):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role") or msg.get("sender") or msg.get("author")
                if role not in ("user", "human"):
                    continue
                content = msg.get("content") or msg.get("text")
                if not isinstance(content, str) or not content.strip():
                    continue
                yield RawPrompt(
                    content=content.strip(),
                    source=self.source_name,
                    source_ref=f"{fp.name}#{conv_id}:{idx}",
                    created_at=_parse_ts(msg.get("timestamp") or msg.get("created_at")),
                    metadata={"file": str(fp), "conversation_id": conv_id},
                )

    @staticmethod
    def _normalize_conversations(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            if data and isinstance(data[0], dict) and "messages" in data[0]:
                return data
            # bare message list
            return [{"messages": data}]
        if isinstance(data, dict):
            if "conversations" in data and isinstance(data["conversations"], list):
                return data["conversations"]
            if "messages" in data:
                return [data]
        return []


def _parse_ts(v: Any) -> datetime | None:
    if isinstance(v, (int, float)):
        from datetime import timezone

        seconds = v / 1000 if v > 10_000_000_000 else v
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None

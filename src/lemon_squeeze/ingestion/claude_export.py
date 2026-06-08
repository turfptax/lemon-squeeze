"""Ingest prompts from a Claude.ai data export (conversations.json).

The Claude export is a JSON array of conversations; each conversation has a
`chat_messages` list where every message has `sender` ("human" | "assistant")
and either `text` or `content` (a list of typed parts).
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from lemon_squeeze.ingestion.base import Ingester, RawPrompt


class ClaudeExportIngester(Ingester):
    source_name = "claude_export"

    def __init__(self, export_path: Path) -> None:
        self.export_path = Path(export_path)

    def iter_prompts(self) -> Iterator[RawPrompt]:
        data = json.loads(self.export_path.read_text(encoding="utf-8"))
        conversations = data if isinstance(data, list) else data.get("conversations", [])

        for conv in conversations:
            if not isinstance(conv, dict):
                continue
            conv_id = conv.get("uuid") or conv.get("id") or "unknown"
            conv_name = conv.get("name") or conv.get("title")
            messages = conv.get("chat_messages") or conv.get("messages") or []
            for idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    continue
                sender = msg.get("sender") or msg.get("role")
                if sender not in ("human", "user"):
                    continue
                content = self._extract_text(msg)
                if not content:
                    continue
                yield RawPrompt(
                    content=content,
                    source=self.source_name,
                    source_ref=f"{conv_id}:{idx}",
                    created_at=_parse_ts(msg.get("created_at") or msg.get("timestamp")),
                    metadata={"conversation_name": conv_name, "conversation_id": conv_id},
                )

    @staticmethod
    def _extract_text(msg: dict[str, Any]) -> str | None:
        text = msg.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        parts.append(t)
            joined = "\n".join(parts).strip()
            return joined or None
        return None


def _parse_ts(v: Any) -> datetime | None:
    if not isinstance(v, str):
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None

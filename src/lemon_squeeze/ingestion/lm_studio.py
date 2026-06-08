"""Ingest prompts from LM Studio's local conversation logs.

LM Studio persists chats as JSON under (typically) ~/.cache/lm-studio/conversations
on macOS/Linux and %USERPROFILE%\\.cache\\lm-studio\\conversations on Windows.
The exact shape has shifted between LM Studio versions, so this ingester is
lenient: it walks any JSON files under the directory and pulls anything that
looks like a user-authored message.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lemon_squeeze.config import settings
from lemon_squeeze.ingestion.base import Ingester, RawPrompt


class LMStudioIngester(Ingester):
    source_name = "lm_studio"

    def __init__(self, logs_dir: Path | None = None) -> None:
        self.logs_dir = Path(logs_dir or settings.lm_studio_logs_dir or "")

    def iter_prompts(self) -> Iterator[RawPrompt]:
        if not self.logs_dir or not self.logs_dir.exists():
            return
        for path in self.logs_dir.rglob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            yield from self._extract(data, path)

    def _extract(self, data: Any, path: Path) -> Iterator[RawPrompt]:
        messages = self._find_messages(data)
        conversation_id = (
            data.get("id") or data.get("conversation_id") or path.stem
            if isinstance(data, dict)
            else path.stem
        )
        model = data.get("model") if isinstance(data, dict) else None

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or msg.get("sender")
            if role != "user":
                continue
            content = self._get_content(msg)
            if not content:
                continue
            ts = self._get_timestamp(msg)
            yield RawPrompt(
                content=content,
                source=self.source_name,
                source_ref=f"{path.name}#{conversation_id}:{idx}",
                created_at=ts,
                metadata={"file": str(path), "model": model, "role": role},
            )

    @staticmethod
    def _find_messages(data: Any) -> list[Any]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("messages", "turns", "history", "chat", "conversation"):
                val = data.get(key)
                if isinstance(val, list):
                    return val
        return []

    @staticmethod
    def _get_content(msg: dict[str, Any]) -> str | None:
        content = msg.get("content") or msg.get("text") or msg.get("message")
        if isinstance(content, str):
            return content.strip() or None
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    t = part.get("text") or part.get("content")
                    if isinstance(t, str):
                        parts.append(t)
            joined = "\n".join(parts).strip()
            return joined or None
        return None

    @staticmethod
    def _get_timestamp(msg: dict[str, Any]) -> datetime | None:
        for key in ("timestamp", "created_at", "createdAt", "time"):
            v = msg.get(key)
            if v is None:
                continue
            if isinstance(v, (int, float)):
                # ms vs s heuristic
                seconds = v / 1000 if v > 10_000_000_000 else v
                return datetime.fromtimestamp(seconds, tz=timezone.utc)
            if isinstance(v, str):
                try:
                    return datetime.fromisoformat(v.replace("Z", "+00:00"))
                except ValueError:
                    continue
        return None

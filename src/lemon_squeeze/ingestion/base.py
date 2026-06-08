"""Base ingester interface + shared persistence logic.

Each concrete ingester yields `RawPrompt` items. The base class handles
hashing, dedup, token-counting, and the insert. Concrete ingesters never
talk to the DB directly — they just produce records.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select

from lemon_squeeze.db import Prompt, get_session
from lemon_squeeze.utils import count_tokens, hash_prompt, normalize_prompt


@dataclass
class RawPrompt:
    """One unit of work emitted by an ingester."""

    content: str
    source: str
    source_ref: str | None = None
    created_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestResult:
    inserted: int = 0
    duplicates: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: IngestResult) -> None:
        self.inserted += other.inserted
        self.duplicates += other.duplicates
        self.skipped += other.skipped
        self.errors.extend(other.errors)


class Ingester(ABC):
    """Subclass and implement `iter_prompts`. Call `.run()` to persist."""

    source_name: str = "unknown"

    @abstractmethod
    def iter_prompts(self) -> Iterator[RawPrompt]:
        ...

    def run(self, batch_size: int = 200, *, dry_run: bool = False) -> IngestResult:
        """Drain `iter_prompts` and persist (or simulate persistence if `dry_run`).

        When `dry_run=True`, the dedup query still runs and the counts in the
        returned `IngestResult` reflect what would have happened — but no
        rows are written to the DB. Useful for previewing a large seed file:
        `lemon ingest seed huge.jsonl --dry-run`.
        """
        result = IngestResult()
        batch: list[RawPrompt] = []

        for raw in self.iter_prompts():
            if not raw.content or not raw.content.strip():
                result.skipped += 1
                continue
            batch.append(raw)
            if len(batch) >= batch_size:
                result.merge(self._persist(batch, dry_run=dry_run))
                batch.clear()

        if batch:
            result.merge(self._persist(batch, dry_run=dry_run))
        return result

    @staticmethod
    def _persist(batch: Iterable[RawPrompt], *, dry_run: bool = False) -> IngestResult:
        result = IngestResult()
        with get_session() as session:
            hashes = [hash_prompt(r.content) for r in batch]
            existing = set(
                session.scalars(
                    select(Prompt.content_hash).where(Prompt.content_hash.in_(hashes))
                ).all()
            )
            for raw, h in zip(batch, hashes, strict=True):
                if h in existing:
                    result.duplicates += 1
                    continue
                if not dry_run:
                    content = (
                        normalize_prompt(raw.content) if "\n" not in raw.content else raw.content
                    )
                    prompt = Prompt(
                        content=content,
                        content_hash=h,
                        token_count=count_tokens(raw.content),
                        char_count=len(raw.content),
                        source=raw.source,
                        source_ref=raw.source_ref,
                        source_metadata=raw.metadata or None,
                        created_at=raw.created_at,
                    )
                    session.add(prompt)
                # Track within-batch dupes even in dry-run so the count is right.
                existing.add(h)
                result.inserted += 1
        return result

"""Synthesize test prompts to round out coverage.

Two modes:
  - `from_seed_file`: load a JSONL/JSON file you've authored (prompt + optional tag).
    Useful for hand-curated regression suites.
  - `from_templates`: expand simple `{slot}`-style templates into many concrete prompts.
    Lets you generate broad coverage cheaply.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from lemon_squeeze.ingestion.base import Ingester, RawPrompt


def _template_ref(template: str) -> str:
    """Stable per-template identifier for `source_ref`. Python's built-in
    `hash()` is randomized per process for strings (PYTHONHASHSEED), so the
    same template would produce different source_refs across runs --
    breaking the implicit contract that source_refs are stable identifiers.
    Use a truncated SHA-256 instead."""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:8]


class SeedFileIngester(Ingester):
    """Ingest a hand-authored JSON/JSONL file of prompts."""

    source_name = "self_generated:seed"

    def __init__(
        self,
        path: Path,
        metadata_overlay: dict[str, Any] | None = None,
    ) -> None:
        """`metadata_overlay`, when given, is merged into every emitted
        prompt's source_metadata. Lets callers (e.g. `bench.load`) tag the
        prompts with bench-specific fields so downstream filters can scope
        to that bench rather than relying on the filename alone.
        """
        self.path = Path(path)
        self.metadata_overlay = metadata_overlay or {}

    def iter_prompts(self) -> Iterator[RawPrompt]:
        suffix = self.path.suffix.lower()
        text = self.path.read_text(encoding="utf-8")
        if suffix == ".jsonl":
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            data = json.loads(text)
            records = data if isinstance(data, list) else [data]

        content_keys = {"prompt", "content", "text"}
        for idx, rec in enumerate(records):
            metadata: dict[str, Any] = {}
            if isinstance(rec, str):
                content = rec
            elif isinstance(rec, dict):
                content = rec.get("prompt") or rec.get("content") or rec.get("text")
                # Normalize: `tag`/`category`/`intended_tag` all map to intended_tag.
                tag = rec.get("intended_tag") or rec.get("tag") or rec.get("category")
                if tag:
                    metadata["intended_tag"] = tag
                # Preserve all other fields (expected_contains, expected, notes, ...).
                for k, v in rec.items():
                    if k in content_keys or k in ("tag", "category", "intended_tag"):
                        continue
                    metadata[k] = v
            else:
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            # Caller-supplied overlay (bench_dir, etc) loses to per-record
            # metadata when keys collide — the file's own metadata is more
            # specific. Same merge order applies in the prompts code path.
            merged = {**self.metadata_overlay, **metadata}
            yield RawPrompt(
                content=content.strip(),
                source=self.source_name,
                source_ref=f"{self.path.name}:{idx}",
                metadata=merged,
            )


class TemplateIngester(Ingester):
    """Expand `{slot}` templates into concrete prompts via cartesian product over slot values."""

    source_name = "self_generated:template"

    def __init__(
        self,
        templates: Sequence[str],
        slots: dict[str, Sequence[str]],
        intended_tag: str | None = None,
    ) -> None:
        self.templates = list(templates)
        self.slots = {k: list(v) for k, v in slots.items()}
        self.intended_tag = intended_tag

    def iter_prompts(self) -> Iterator[RawPrompt]:
        keys = list(self.slots.keys())
        value_lists = [self.slots[k] for k in keys]
        for template in self.templates:
            for combo in itertools.product(*value_lists) if keys else [()]:
                values = dict(zip(keys, combo, strict=True))
                try:
                    content = template.format(**values)
                except KeyError:
                    continue
                meta: dict[str, Any] = {"template": template, "slots": values}
                if self.intended_tag:
                    meta["intended_tag"] = self.intended_tag
                yield RawPrompt(
                    content=content,
                    source=self.source_name,
                    source_ref=f"tmpl:{_template_ref(template)}",
                    metadata=meta,
                )

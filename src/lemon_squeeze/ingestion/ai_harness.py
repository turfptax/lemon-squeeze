"""Importer for the sibling AI Harness project (data/harness_logs.db).

Unlike the other ingesters (which only emit prompts), AI Harness has the entire
(prompt, model, run, evaluations) tuple per row, so this importer writes across
all four tables in one pass. It deliberately doesn't inherit from `Ingester` —
the base class is shaped around prompt-only sources, and forcing this through
that ABC would distort both.

Source of truth for the schema we're reading: AI Harness README "Data Model"
section. The relevant columns on `runs`:

    id (UUID), task, supervisor_model, worker_model, worker_sequence (JSON),
    success (0/1/NULL), total_tokens, num_loops, timestamp (ISO),
    estimated_api_cost_usd, run_duration_seconds,
    complexity_score, usefulness_score, data_value_score, scalability_score,
    conversation_messages (JSON), tools_used (JSON), tool_results (JSON)
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from lemon_squeeze.db import Evaluation, Model, Prompt, Run, get_session
from lemon_squeeze.utils import count_tokens, hash_prompt, normalize_prompt

SOURCE_NAME = "ai_harness"
SCORER_MODEL = "google/gemini-2.0-flash-001"
AUTO_RUBRICS = ("complexity", "usefulness", "data_value", "scalability")


@dataclass
class ImportResult:
    runs_seen: int = 0
    runs_imported: int = 0
    runs_skipped_existing: int = 0
    prompts_inserted: int = 0
    prompts_deduped: int = 0
    models_registered: int = 0
    evaluations_inserted: int = 0
    errors: list[str] = field(default_factory=list)


class AIHarnessImporter:
    """Pull historical runs from an AI Harness SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(self.db_path)

    def run(self, *, dry_run: bool = False) -> ImportResult:
        """Import the AI Harness DB. With `dry_run=True`, counts what would
        happen but rolls back the transaction at the end so nothing persists.

        Implementation note: unlike `Ingester._persist`, this importer can't
        simply skip `session.add()` — `_get_or_create_prompt/_get_or_create_model`
        rely on flushed IDs to link runs and evaluations. We let the work
        happen normally and call `session.rollback()` before exit so the
        contextmanager's commit becomes a no-op.
        """
        result = ImportResult()
        rows = self._read_rows()

        with get_session() as session:
            # Cache: AI-Harness run IDs already imported (stored in runs.run_metadata).
            already_imported: set[str] = set()
            for r in session.query(Run).all():
                meta = r.run_metadata or {}
                hid = meta.get("ai_harness_id")
                if hid:
                    already_imported.add(hid)

            # Cache: prompts by content_hash, models by name. Built lazily.
            prompt_cache: dict[str, Prompt] = {}
            model_cache: dict[str, Model] = {}

            for row in rows:
                result.runs_seen += 1
                if row["id"] in already_imported:
                    result.runs_skipped_existing += 1
                    continue

                task = (row.get("task") or "").strip()
                if not task:
                    result.errors.append(f"run {row['id']}: empty task; skipped")
                    continue

                prompt = self._get_or_create_prompt(session, prompt_cache, task, row, result)
                model_name = row.get("worker_model") or row.get("supervisor_model")
                if not model_name:
                    result.errors.append(f"run {row['id']}: no model recorded; skipped")
                    continue
                model = self._get_or_create_model(session, model_cache, model_name, result)

                run = Run(
                    prompt=prompt,
                    model=model,
                    response=self._extract_final_response(row.get("conversation_messages")),
                    tokens_in=None,
                    tokens_out=None,
                    latency_ms=int(row["run_duration_seconds"] * 1000)
                    if row.get("run_duration_seconds") is not None
                    else None,
                    cost_usd=row.get("estimated_api_cost_usd"),
                    run_metadata={
                        "ai_harness_id": row["id"],
                        "supervisor_model": row.get("supervisor_model"),
                        "worker_sequence": _maybe_json(row.get("worker_sequence")),
                        "num_loops": row.get("num_loops"),
                        "total_tokens": row.get("total_tokens"),
                        "tools_used": _maybe_json(row.get("tools_used")),
                        "tool_results": _maybe_json(row.get("tool_results")),
                        "coder_flagged_complete": bool(row.get("coder_flagged_complete")),
                    },
                    created_at=_parse_ts(row.get("timestamp")),
                )
                session.add(run)
                result.runs_imported += 1
                already_imported.add(row["id"])

                # Human pass/fail label (only if present).
                if row.get("success") is not None:
                    session.add(
                        Evaluation(
                            run=run,
                            rubric="human_pass",
                            score=float(row["success"]),
                            passed=bool(row["success"]),
                            scored_by="human",
                            notes="Imported from AI Harness `success` column.",
                        )
                    )
                    result.evaluations_inserted += 1

                # Auto-scored rubrics (1-5).
                for rubric in AUTO_RUBRICS:
                    val = row.get(f"{rubric}_score")
                    if val is None:
                        continue
                    session.add(
                        Evaluation(
                            run=run,
                            rubric=rubric,
                            score=float(val),
                            scored_by="llm",
                            judge_model=SCORER_MODEL,
                            notes="Imported from AI Harness Gemini-Flash scorer.",
                        )
                    )
                    result.evaluations_inserted += 1

            if dry_run:
                # Discard everything we just added — counts in `result` already
                # reflect what would have been written.
                session.rollback()
        return result

    def _read_rows(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            return [dict(r) for r in con.execute("SELECT * FROM runs")]

    @staticmethod
    def _get_or_create_prompt(
        session, cache: dict[str, Prompt], task: str, row: dict, result: ImportResult
    ) -> Prompt:
        h = hash_prompt(task)
        if h in cache:
            return cache[h]
        existing = session.scalar(select(Prompt).where(Prompt.content_hash == h))
        if existing is not None:
            cache[h] = existing
            result.prompts_deduped += 1
            return existing
        prompt = Prompt(
            content=normalize_prompt(task),
            content_hash=h,
            token_count=count_tokens(task),
            char_count=len(task),
            source=SOURCE_NAME,
            source_ref=row["id"],
            source_metadata={"project_id": row.get("project_id")},
            created_at=_parse_ts(row.get("timestamp")),
        )
        session.add(prompt)
        session.flush()  # populate id so child runs can reference it
        cache[h] = prompt
        result.prompts_inserted += 1
        return prompt

    @staticmethod
    def _get_or_create_model(
        session, cache: dict[str, Model], name: str, result: ImportResult
    ) -> Model:
        if name in cache:
            return cache[name]
        existing = session.scalar(select(Model).where(Model.name == name))
        if existing is not None:
            cache[name] = existing
            return existing

        provider, family = _split_provider(name)
        model = Model(
            name=name,
            provider=provider,
            family=family,
            local=False,
        )
        session.add(model)
        session.flush()
        cache[name] = model
        result.models_registered += 1
        return model

    @staticmethod
    def _extract_final_response(conversation_json: Any) -> str | None:
        msgs = _maybe_json(conversation_json)
        if not isinstance(msgs, list):
            return None
        # Take the last assistant message; falls through to any last message with content.
        for msg in reversed(msgs):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        for msg in reversed(msgs):
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return None


def _split_provider(model_name: str) -> tuple[str, str | None]:
    """`anthropic/claude-sonnet-4-6` -> ('anthropic', 'claude')."""
    from lemon_squeeze.utils import split_provider_family

    return split_provider_family(model_name)


def _maybe_json(v: Any) -> Any:
    if v is None or not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def _parse_ts(v: Any) -> datetime | None:
    if not v:
        return None
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None

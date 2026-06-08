"""Rubric = (name, judge_kind, judge_config). YAML-loadable, JSON-loadable.

Example rubric file (rubrics/coding_runs_python.yaml):

    name: produces_python_code
    description: Response contains a python code block
    judge: regex
    config:
      pattern: "```python"
      flags: ""
    applies_to:
      tags: [coding]

The `applies_to` filter is optional; without it, the rubric runs against
every run. With tags, it only runs against prompts that carry one of those
tags (from any classifier).
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

from lemon_squeeze.db import Evaluation, Prompt, PromptTag, Run, get_session
from lemon_squeeze.eval.judges import build_judge


@dataclass
class Rubric:
    name: str
    description: str
    judge_kind: str
    judge_config: dict[str, Any] = field(default_factory=dict)
    applies_to_tags: list[str] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rubric:
        applies = data.get("applies_to") or {}
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            judge_kind=data["judge"],
            judge_config=data.get("config") or {},
            applies_to_tags=applies.get("tags"),
        )

    def config_hash(self) -> str:
        """SHA-256 of the scoring-affecting parts of this rubric.

        `description` is deliberately excluded: editing prose shouldn't
        invalidate stored evaluations. What matters is (judge_kind,
        judge_config, applies_to_tags) — change any of those and previous
        scores are stale.
        """
        import hashlib

        payload = json.dumps(
            {
                "judge": self.judge_kind,
                "config": self.judge_config,
                "applies_to_tags": sorted(self.applies_to_tags) if self.applies_to_tags else None,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_file(cls, path: Path) -> Rubric:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".yaml", ".yml"):
            data = _load_yaml(text)
        else:
            data = json.loads(text)
        return cls.from_dict(data)


@dataclass
class EvalReport:
    runs_seen: int = 0
    runs_evaluated: int = 0
    evaluations_written: int = 0
    skipped_existing: int = 0
    skipped_no_response: int = 0
    skipped_tag_mismatch: int = 0
    replaced: int = 0
    stale_replaced: int = 0   # rows that had a different rubric_hash and got auto-re-scored
    errors: list[str] = field(default_factory=list)


def evaluate_runs(
    rubric: Rubric,
    *,
    run_ids: Iterable[int] | None = None,
    rescored_by: str | None = None,
    skip_existing: bool = True,
    replace_existing: bool = False,
) -> EvalReport:
    """Apply a rubric to runs and write Evaluation rows.

    Behavior knobs:
      * `skip_existing=True` (default) — leave existing (run, rubric) evals
        alone IF their `rubric_hash` matches the current rubric. Evals with a
        different hash (i.e. the YAML was edited since they were written) are
        treated as STALE and auto-replaced. NULL hashes (rows from before this
        feature existed) are treated as up-to-date — we don't churn legacy data.
      * `skip_existing=False` — append a new row alongside the old one (history).
      * `replace_existing=True` — delete old evals for this rubric first, then
        re-score. This is what `lemon eval replay` uses. Takes precedence over
        `skip_existing`.

    `rescored_by`: override the `scored_by` field. Defaults to "auto" for
    deterministic judges and "llm" for the LLM judge.
    """
    from sqlalchemy import delete

    judge = build_judge(rubric.judge_kind, rubric.judge_config)
    scored_by = rescored_by or ("llm" if rubric.judge_kind == "llm" else "auto")
    current_hash = rubric.config_hash()
    report = EvalReport()

    with get_session() as session:
        q = select(Run)
        if run_ids is not None:
            q = q.where(Run.id.in_(list(run_ids)))
        runs = list(session.scalars(q).all())

        # Track which run_ids have an UP-TO-DATE eval for this rubric (skip),
        # vs a STALE one (delete then re-score), vs none (write fresh).
        up_to_date: set[int] = set()
        stale_run_ids: list[int] = []

        if replace_existing:
            del_q = delete(Evaluation).where(Evaluation.rubric == rubric.name)
            if run_ids is not None:
                del_q = del_q.where(Evaluation.run_id.in_(list(run_ids)))
            result = session.execute(del_q)
            report.replaced = result.rowcount or 0
        elif skip_existing:
            # One query for existing (run_id, rubric_hash) pairs for this rubric.
            eq = select(Evaluation.run_id, Evaluation.rubric_hash).where(
                Evaluation.rubric == rubric.name
            )
            if run_ids is not None:
                eq = eq.where(Evaluation.run_id.in_(list(run_ids)))
            for rid, h in session.execute(eq).all():
                if h is None or h == current_hash:
                    # NULL (legacy) or matching: treat as up-to-date.
                    up_to_date.add(rid)
                else:
                    stale_run_ids.append(rid)

            # Delete stale rows so we can re-score cleanly.
            if stale_run_ids:
                del_stale = delete(Evaluation).where(
                    Evaluation.rubric == rubric.name,
                    Evaluation.run_id.in_(stale_run_ids),
                )
                result = session.execute(del_stale)
                report.stale_replaced = result.rowcount or 0

        # Back-compat with the rest of the function — `existing` is now the
        # set of (run_id, rubric_name) tuples we SHOULD skip (up-to-date only).
        existing: set[tuple[int, str]] = {(rid, rubric.name) for rid in up_to_date}

        # Precompute tag index for `applies_to_tags`.
        tag_index: dict[int, set[str]] = {}
        if rubric.applies_to_tags:
            for pt in session.scalars(select(PromptTag)).all():
                tag_index.setdefault(pt.prompt_id, set()).add(pt.tag)

        # Detach for after-session work. We carry prompt.source_metadata so
        # per-prompt judges (ExpectedContainsJudge) can read ground truth
        # without re-opening a session in the hot loop.
        runs_data: list[tuple[int, int, str | None, str, dict | None]] = []
        for r in runs:
            prompt = session.get(Prompt, r.prompt_id)
            runs_data.append(
                (
                    r.id,
                    r.prompt_id,
                    r.response,
                    prompt.content if prompt else "",
                    prompt.source_metadata if prompt else None,
                )
            )

    for run_id, prompt_id, response, prompt_text, prompt_meta in runs_data:
        report.runs_seen += 1
        if skip_existing and (run_id, rubric.name) in existing:
            report.skipped_existing += 1
            continue
        if response is None or not response.strip():
            report.skipped_no_response += 1
            continue
        if rubric.applies_to_tags:
            tags = tag_index.get(prompt_id, set())
            if not tags & set(rubric.applies_to_tags):
                report.skipped_tag_mismatch += 1
                continue

        try:
            verdict = judge.evaluate(prompt_text, response, metadata=prompt_meta)
        except Exception as e:
            report.errors.append(f"run {run_id}: {e!r}")
            continue

        with get_session() as session:
            session.add(
                Evaluation(
                    run_id=run_id,
                    rubric=rubric.name,
                    rubric_hash=current_hash,
                    score=verdict.score,
                    passed=verdict.passed,
                    scored_by=scored_by,
                    judge_model=verdict.judge_model,
                    notes=verdict.notes,
                    eval_metadata={"rubric_description": rubric.description, **verdict.extra},
                )
            )
        report.evaluations_written += 1
        report.runs_evaluated += 1
    return report


# ---------- YAML loader -----------------------------------------------------
# Uses PyYAML. We previously had a ~70-line hand-rolled parser to keep the
# dep count low; it tripped on backslash escapes (`\s` in regex rubrics) and
# block scalars (multi-line `description: |`). PyYAML handles both natively
# and is already a transitive dep of common ML libs, so the marginal cost is
# near zero. We use `safe_load` — no arbitrary Python objects.


def _load_yaml(text: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Rubric loading requires PyYAML. Install with: pip install pyyaml"
        ) from e

    result = yaml.safe_load(text)
    if not isinstance(result, dict):
        raise ValueError(
            f"top-level YAML must be a mapping, got {type(result).__name__}"
        )
    return result

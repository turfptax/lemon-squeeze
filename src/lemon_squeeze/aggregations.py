"""Single source of truth for per-(bucket, model) evaluation aggregations.

Four call sites used to compute pass_rate / avg_score / avg_cost / avg_latency
from runs+evaluations independently — router.stats_by_tag,
compare.compare's _summarize, bench._per_category_breakdown, and the
dashboard's heatmap query. They drifted: bench had `cost_per_pass`, router
had `size_params_b`, compare carried Wilson CIs nowhere else. Adding a new
metric (`tokens_per_pass`, p50 latency, whatever) meant a four-file patch.

This module is the consolidation. One SQL GROUP BY (`aggregate_by_tag_model`)
produces a uniform `Aggregate` row carrying the raw counters + model metadata;
all derived metrics (pass_rate, cost_per_pass) are `@property`s, so adding a
new one is a one-line change here, not a four-file change.

There are TWO entry points because two callers group by something different
than `PromptTag.tag`:
  - `aggregate_by_tag_model` — groups by (PromptTag.tag, Model.name). Used
    by router, compare, dashboard, report.
  - `aggregate_by_intended_tag_model` — groups by
    (Prompt.source_metadata->>"intended_tag", Model.name). Used by bench.
    SQLite's JSON GROUP BY is awkward, so this one buckets in Python after
    fetching prompt → intended_tag once.

Both return the same `Aggregate` shape, so downstream consumers don't care
which path was taken.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from sqlalchemy import case, func, select

from lemon_squeeze.cache import (
    _MISS,
    aggregations_cache,
    aggregations_key,
)
from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run, get_session


@dataclass
class Aggregate:
    """One per-bucket aggregate row.

    `key` is the tuple of grouping-key values in declaration order — for
    `aggregate_by_tag_model` that's (tag, model_name), for
    `aggregate_by_intended_tag_model` that's (intended_tag, model_name).

    `n_passed_known` counts only evaluations whose `passed` column is non-NULL
    (i.e. real pass/fail rubrics, not score-only LLM judges). `pass_rate` is
    computed against that denominator — otherwise an LLM-only scored prompt
    would drag the rate to zero unfairly.
    """

    key: tuple[str, str]
    n_evals: int
    n_passed: int
    n_passed_known: int
    avg_score: float
    avg_cost_usd: float | None
    avg_latency_ms: float | None
    # Optional model metadata; populated by both entry points.
    model_id: int | None = None
    model_size_b: float | None = None
    model_context_window: int | None = None

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_passed_known if self.n_passed_known else 0.0

    @property
    def cost_per_pass(self) -> float | None:
        """Expected cost to get one passing run on this bucket."""
        if self.avg_cost_usd is None or self.pass_rate <= 0:
            return None
        return self.avg_cost_usd / self.pass_rate

    @property
    def tag(self) -> str:
        """Convenience accessor for the bucket dimension (first key element)."""
        return self.key[0]

    @property
    def model_name(self) -> str:
        return self.key[1]


# ---------- Entry point 1: PromptTag.tag × Model.name ------------------------


def aggregate_by_tag_model(
    *,
    rubrics: Sequence[str],
    tags: Sequence[str] | None = None,
    prompt_ids: Sequence[int] | None = None,
    model_names: Sequence[str] | None = None,
) -> list[Aggregate]:
    """One SQL GROUP BY producing per-(tag, model_name) aggregates.

    All filter args are AND-ed together. Pass `None` to skip a filter; pass
    `[]` to match nothing.

    NOTE: PromptTag has multiple rows per (prompt_id, tag) — one per classifier
    that assigned the tag (e.g. heuristic + ml). Naively joining inflates the
    counts. We pre-select DISTINCT (prompt_id, tag) pairs in a subquery so each
    prompt contributes once per tag regardless of classifier provenance.
    """
    if not rubrics:
        return []
    if tags is not None and not tags:
        return []
    if model_names is not None and not model_names:
        return []
    if prompt_ids is not None and not prompt_ids:
        return []

    cache = aggregations_cache()
    key = aggregations_key(
        fn="aggregate_by_tag_model",
        rubrics=rubrics,
        tags=tags,
        model_names=model_names,
        prompt_ids=prompt_ids,
    )
    cached = cache.get(key)
    if cached is not _MISS:
        return cached

    passed_known = case((Evaluation.passed.is_not(None), 1), else_=0)
    passed_true = case((Evaluation.passed == True, 1), else_=0)  # noqa: E712

    with get_session() as s:
        distinct_tags_q = select(
            PromptTag.prompt_id.label("prompt_id"),
            PromptTag.tag.label("tag"),
        ).distinct()
        if tags is not None:
            distinct_tags_q = distinct_tags_q.where(PromptTag.tag.in_(list(tags)))
        if prompt_ids is not None:
            distinct_tags_q = distinct_tags_q.where(
                PromptTag.prompt_id.in_(list(prompt_ids))
            )
        prompt_tag = distinct_tags_q.subquery()

        q = (
            select(
                prompt_tag.c.tag,
                Model.name,
                Model.id,
                Model.size_params_b,
                Model.context_window,
                func.count(Evaluation.id),
                func.sum(passed_true),
                func.sum(passed_known),
                func.avg(Evaluation.score),
                func.avg(Run.cost_usd),
                func.avg(Run.latency_ms),
            )
            .select_from(prompt_tag)
            .join(Run, Run.prompt_id == prompt_tag.c.prompt_id)
            .join(Model, Model.id == Run.model_id)
            .join(Evaluation, Evaluation.run_id == Run.id)
            .where(Evaluation.rubric.in_(list(rubrics)))
            .group_by(
                prompt_tag.c.tag,
                Model.name,
                Model.id,
                Model.size_params_b,
                Model.context_window,
            )
        )
        if model_names is not None:
            q = q.where(Model.name.in_(list(model_names)))
        rows = s.execute(q).all()

    result = [
        Aggregate(
            key=(tag, name),
            model_id=mid,
            model_size_b=size_b,
            model_context_window=ctx,
            n_evals=int(n_evals or 0),
            n_passed=int(n_passed or 0),
            n_passed_known=int(n_passed_known or 0),
            avg_score=float(avg_score) if avg_score is not None else 0.0,
            avg_cost_usd=float(avg_cost) if avg_cost is not None else None,
            avg_latency_ms=float(avg_lat) if avg_lat is not None else None,
        )
        for (
            tag, name, mid, size_b, ctx,
            n_evals, n_passed, n_passed_known,
            avg_score, avg_cost, avg_lat,
        ) in rows
    ]
    cache.put(key, result)
    return result


# ---------- Entry point 2: Prompt.source_metadata['intended_tag'] × Model.name


def aggregate_by_intended_tag_model(
    *,
    rubrics: Sequence[str],
    prompt_ids: Sequence[int],
) -> list[Aggregate]:
    """For bench: bucket by `Prompt.source_metadata['intended_tag']`.

    SQLite's JSON1 GROUP BY is messy and not portable to other backends,
    so we fetch `(prompt_id → intended_tag)` once and bucket the per-eval
    rows in Python. Same `Aggregate` shape as the tag-model path.
    """
    if not rubrics or not prompt_ids:
        return []

    with get_session() as s:
        prompt_rows = s.execute(
            select(Prompt.id, Prompt.source_metadata).where(
                Prompt.id.in_(list(prompt_ids))
            )
        ).all()
        category_by_prompt: dict[int, str] = {
            pid: (meta or {}).get("intended_tag", "uncategorized")
            for pid, meta in prompt_rows
        }

        rows = s.execute(
            select(
                Run.prompt_id,
                Model.id,
                Model.name,
                Model.size_params_b,
                Model.context_window,
                Evaluation.score,
                Evaluation.passed,
                Run.cost_usd,
                Run.latency_ms,
            )
            .join(Model, Model.id == Run.model_id)
            .join(Evaluation, Evaluation.run_id == Run.id)
            .where(
                Run.prompt_id.in_(list(prompt_ids)),
                Evaluation.rubric.in_(list(rubrics)),
            )
        ).all()

    if not rows:
        return []

    buckets: dict[
        tuple[str, str],
        list[tuple[float, bool | None, float | None, int | None]],
    ] = defaultdict(list)
    model_meta: dict[str, tuple[int, float | None, int | None]] = {}
    for prompt_id, mid, name, size_b, ctx, score, passed, cost, lat in rows:
        cat = category_by_prompt.get(prompt_id, "uncategorized")
        buckets[(cat, name)].append((float(score), passed, cost, lat))
        model_meta[name] = (mid, size_b, ctx)

    out: list[Aggregate] = []
    for (cat, name), bucket in buckets.items():
        scores = [s for s, _, _, _ in bucket]
        passes_known = [p for _, p, _, _ in bucket if p is not None]
        costs = [c for _, _, c, _ in bucket if c is not None]
        lats = [l for _, _, _, l in bucket if l is not None]
        mid, size_b, ctx = model_meta[name]
        out.append(
            Aggregate(
                key=(cat, name),
                model_id=mid,
                model_size_b=size_b,
                model_context_window=ctx,
                n_evals=len(bucket),
                n_passed=sum(1 for p in passes_known if p),
                n_passed_known=len(passes_known),
                avg_score=mean(scores) if scores else 0.0,
                avg_cost_usd=mean(costs) if costs else None,
                avg_latency_ms=mean(lats) if lats else None,
            )
        )
    return out


# ---------- Cross-cutting helpers --------------------------------------------


def group_by_first_key(aggs: list[Aggregate]) -> dict[str, list[Aggregate]]:
    """Group results by the first key dimension (tag / category)."""
    out: dict[str, list[Aggregate]] = defaultdict(list)
    for a in aggs:
        out[a.tag].append(a)
    return out

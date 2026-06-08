"""One-shot executive summary — `lemon report`.

Three sections, computed read-only off the DB:

  1. Headline — prompt/model/run/eval counts, per-source/per-rubric breakdowns
  2. Per-tag scorecard — for each tag that has any evaluation data, show the
     quality pick (best pass_rate) and the cost pick (cheapest qualifying)
  3. Coverage gaps — tags with prompts but no qualifying model, or no
     evaluation data at all

The "qualifying" filter is the same one the router uses (`pass_rate >=
threshold` with `n >= min_samples`). Defaults are routing defaults so the
report and the router agree on what's "good enough."
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html import escape as _h
from typing import Any

from sqlalchemy import func, select

from lemon_squeeze.db import (
    Evaluation,
    Model,
    Prompt,
    PromptTag,
    Run,
    TagTaxonomy,
    get_session,
)
from lemon_squeeze.router import (
    BALANCED,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_THRESHOLD,
    _score_candidates,
    stats_by_tag,
)


@dataclass
class TagScorecard:
    tag: str
    n_prompts: int
    n_runs: int
    n_evals: int
    quality_pick: str | None       # model name with highest pass rate among qualifying
    quality_pass_rate: float | None
    quality_n: int | None
    cost_pick: str | None          # model name that's cheapest among qualifying
    cost_pick_avg_cost: float | None
    balanced_pick: str | None      # model name under BALANCED weights
    has_qualifying: bool
    qualifying_models: int


@dataclass
class CoverageGap:
    tag: str
    n_prompts: int
    reason: str  # "no_runs" | "no_evals" | "no_qualifying"


@dataclass
class RubricFreshness:
    rubric: str
    n_evals: int
    last_scored_at: datetime | None
    age_days: float | None
    stale: bool  # True if older than staleness threshold (default 30d)
    scored_by_breakdown: list[tuple[str, int]] = field(default_factory=list)


REPORT_SCHEMA_VERSION = 1


@dataclass
class Report:
    # Headline
    n_prompts: int = 0
    n_models: int = 0
    n_runs: int = 0
    n_evals: int = 0
    n_runs_with_error: int = 0
    total_cost_usd: float = 0.0
    prompts_by_source: list[tuple[str, int]] = field(default_factory=list)
    evals_by_rubric: list[tuple[str, int]] = field(default_factory=list)
    # Per-tag
    scorecards: list[TagScorecard] = field(default_factory=list)
    gaps: list[CoverageGap] = field(default_factory=list)
    # Per-rubric
    rubric_freshness: list[RubricFreshness] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a stable JSON-friendly dict.

        Schema is stable across versions — additive only. Datetimes become ISO
        strings; nested dataclasses become dicts; tuples become lists.
        """
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "headline": {
                "n_prompts": self.n_prompts,
                "n_models": self.n_models,
                "n_runs": self.n_runs,
                "n_evals": self.n_evals,
                "n_runs_with_error": self.n_runs_with_error,
                "total_cost_usd": self.total_cost_usd,
                "prompts_by_source": [
                    {"source": s, "count": c} for s, c in self.prompts_by_source
                ],
                "evals_by_rubric": [
                    {"rubric": r, "count": c} for r, c in self.evals_by_rubric
                ],
            },
            "scorecards": [asdict(sc) for sc in self.scorecards],
            "gaps": [asdict(g) for g in self.gaps],
            "rubric_freshness": [
                _freshness_to_dict(rf) for rf in self.rubric_freshness
            ],
        }


def _freshness_to_dict(rf: RubricFreshness) -> dict[str, Any]:
    return {
        "rubric": rf.rubric,
        "n_evals": rf.n_evals,
        "last_scored_at": rf.last_scored_at.isoformat() if rf.last_scored_at else None,
        "age_days": rf.age_days,
        "stale": rf.stale,
        "scored_by_breakdown": [
            {"scored_by": name, "count": n} for name, n in rf.scored_by_breakdown
        ],
    }


def build_report(
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    authoritative_rubrics: tuple[str, ...] = ("human_pass",),
    staleness_days: float = 30.0,
) -> Report:
    rep = Report()
    _fill_headline(rep)
    _fill_per_tag(rep, threshold, min_samples, authoritative_rubrics)
    _fill_rubric_freshness(rep, staleness_days)
    return rep


def headline_stats() -> Report:
    """Just the headline counts + per-source/per-rubric breakdowns.

    Cheaper than `build_report()` when the caller only needs the top metrics
    (e.g. dashboard overview tab). One session, six aggregate queries.
    """
    rep = Report()
    _fill_headline(rep)
    return rep


def _fill_headline(rep: Report) -> None:
    with get_session() as s:
        rep.n_prompts = s.scalar(select(func.count()).select_from(Prompt)) or 0
        rep.n_models = s.scalar(select(func.count()).select_from(Model)) or 0
        rep.n_runs = s.scalar(select(func.count()).select_from(Run)) or 0
        rep.n_evals = s.scalar(select(func.count()).select_from(Evaluation)) or 0
        rep.n_runs_with_error = (
            s.scalar(select(func.count()).select_from(Run).where(Run.error.is_not(None))) or 0
        )
        rep.total_cost_usd = (
            s.scalar(select(func.coalesce(func.sum(Run.cost_usd), 0.0))) or 0.0
        )
        rep.prompts_by_source = [
            (src, cnt)
            for src, cnt in s.execute(
                select(Prompt.source, func.count())
                .group_by(Prompt.source)
                .order_by(func.count().desc())
            ).all()
        ]
        rep.evals_by_rubric = [
            (rubric, cnt)
            for rubric, cnt in s.execute(
                select(Evaluation.rubric, func.count())
                .group_by(Evaluation.rubric)
                .order_by(func.count().desc())
            ).all()
        ]


def _fill_per_tag(
    rep: Report,
    threshold: float,
    min_samples: int,
    authoritative_rubrics: tuple[str, ...],
) -> None:
    with get_session() as s:
        # Taxonomy tags first, then any tags that exist on PromptTag but aren't
        # in the taxonomy (e.g. user-added tags from data import).
        taxonomy_tags = sorted(
            row.tag for row in s.scalars(select(TagTaxonomy)).all()
        )
        extra_tags = sorted(
            {
                t
                for (t,) in s.execute(select(PromptTag.tag).distinct()).all()
            }
            - set(taxonomy_tags)
        )
        all_tags = taxonomy_tags + extra_tags

        # Per-tag prompt counts.
        prompt_counts_by_tag: dict[str, int] = {
            t: cnt
            for (t, cnt) in s.execute(
                select(PromptTag.tag, func.count(func.distinct(PromptTag.prompt_id)))
                .group_by(PromptTag.tag)
            ).all()
        }
        # Per-tag run counts (any model).
        run_counts_by_tag: dict[str, int] = {
            t: cnt
            for (t, cnt) in s.execute(
                select(PromptTag.tag, func.count(Run.id))
                .join(Run, Run.prompt_id == PromptTag.prompt_id)
                .group_by(PromptTag.tag)
            ).all()
        }
        # Per-tag eval counts (any rubric).
        eval_counts_by_tag: dict[str, int] = {
            t: cnt
            for (t, cnt) in s.execute(
                select(PromptTag.tag, func.count(Evaluation.id))
                .join(Run, Run.prompt_id == PromptTag.prompt_id)
                .join(Evaluation, Evaluation.run_id == Run.id)
                .group_by(PromptTag.tag)
            ).all()
        }

    for tag in all_tags:
        n_prompts = prompt_counts_by_tag.get(tag, 0)
        if n_prompts == 0:
            continue  # tag exists in taxonomy but no prompts use it; skip
        n_runs = run_counts_by_tag.get(tag, 0)
        n_evals = eval_counts_by_tag.get(tag, 0)

        candidates = stats_by_tag([tag], authoritative_rubrics=authoritative_rubrics)
        qualifying = [
            c for c in candidates
            if c.sample_count >= min_samples and c.pass_rate >= threshold
        ]

        if not candidates:
            reason = "no_runs" if n_runs == 0 else "no_evals"
            rep.gaps.append(CoverageGap(tag=tag, n_prompts=n_prompts, reason=reason))
            continue

        # If no model qualifies under (threshold, min_samples), still report the
        # next-best candidate so the user has something to act on ("X is your
        # best bet at this tag but needs more samples").
        if not qualifying:
            rep.gaps.append(CoverageGap(tag=tag, n_prompts=n_prompts, reason="no_qualifying"))
            best = max(candidates, key=lambda c: (c.pass_rate, c.sample_count))
            cost_pick = None
            balanced_pick = None
        else:
            best = max(qualifying, key=lambda c: (c.pass_rate, -_cost_or_inf(c)))
            cost_pick = min(qualifying, key=_cost_or_inf)
            balanced_scores = _score_candidates(qualifying, BALANCED.normalize())
            for c, sc in zip(qualifying, balanced_scores, strict=True):
                c.composite_score = sc
            balanced_pick = max(qualifying, key=lambda c: c.composite_score or 0.0)

        rep.scorecards.append(
            TagScorecard(
                tag=tag,
                n_prompts=n_prompts,
                n_runs=n_runs,
                n_evals=n_evals,
                quality_pick=best.model_name,
                quality_pass_rate=best.pass_rate,
                quality_n=best.sample_count,
                cost_pick=cost_pick.model_name if cost_pick else None,
                cost_pick_avg_cost=cost_pick.avg_cost_usd if cost_pick else None,
                balanced_pick=balanced_pick.model_name if balanced_pick else None,
                has_qualifying=bool(qualifying),
                qualifying_models=len(qualifying),
            )
        )


def _cost_or_inf(c) -> float:
    return c.avg_cost_usd if c.avg_cost_usd is not None else float("inf")


# ---------- HTML serialization ----------------------------------------------


def report_to_html(rep: Report, *, title: str = "Lemon Squeeze report") -> str:
    """Render a self-contained HTML snapshot of the report.

    No external CSS / JS; safe to email or commit to a repo. Tables use a
    minimal monospace look that prints reasonably.
    """
    css = """
    body { font-family: -apple-system, system-ui, sans-serif; max-width: 1100px;
           margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
    h1 { font-size: 1.6rem; margin-bottom: 0.3rem; }
    h2 { font-size: 1.2rem; margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
    .meta { color: #666; font-size: 0.85rem; margin-bottom: 1.5rem; }
    .metrics { display: flex; gap: 1.5rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .metric { background: #f5f5f5; padding: 0.7rem 1rem; border-radius: 6px; min-width: 110px; }
    .metric-label { font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.4rem; font-weight: 600; margin-top: 0.2rem; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-bottom: 1.5rem; }
    th { text-align: left; background: #f0f0f0; padding: 0.5rem 0.7rem; border-bottom: 2px solid #ccc; }
    td { padding: 0.4rem 0.7rem; border-bottom: 1px solid #eee; }
    tr:hover td { background: #fafafa; }
    .pass { color: #2a7f3a; font-weight: 600; }
    .fail { color: #c0392b; font-weight: 600; }
    .warn { color: #c98a13; font-weight: 600; }
    .stale { background: #fff3cd; }
    .gap-no_runs { color: #c0392b; }
    .gap-no_evals { color: #c98a13; }
    .gap-no_qualifying { color: #d97706; }
    """

    sections = [
        f"<h1>{_h(title)}</h1>",
        f'<div class="meta">generated {datetime.now(timezone.utc).isoformat(timespec="seconds")}</div>',
        _section_headline_html(rep),
        _section_scorecards_html(rep),
        _section_gaps_html(rep),
        _section_freshness_html(rep),
    ]

    return (
        "<!doctype html>\n<html lang=\"en\"><head>"
        f"<meta charset=\"utf-8\"><title>{_h(title)}</title>"
        f"<style>{css}</style>"
        "</head><body>"
        + "\n".join(sections)
        + "</body></html>"
    )


def _section_headline_html(rep: Report) -> str:
    metrics = "\n".join(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>'
        for label, value in [
            ("Prompts", rep.n_prompts),
            ("Models", rep.n_models),
            ("Runs", rep.n_runs),
            ("Evaluations", rep.n_evals),
            ("Cost so far", f"${rep.total_cost_usd:.2f}"),
            ("Run errors", rep.n_runs_with_error),
        ]
    )
    sources = _kv_table_html("Prompts by source", rep.prompts_by_source, ("source", "count"))
    rubrics = _kv_table_html("Evaluations by rubric", rep.evals_by_rubric, ("rubric", "count"))
    return f"<h2>Headline</h2><div class=\"metrics\">{metrics}</div>{sources}{rubrics}"


def _section_scorecards_html(rep: Report) -> str:
    if not rep.scorecards:
        return ""
    rows = []
    for sc in rep.scorecards:
        cls = "" if sc.has_qualifying else " class=\"warn\""
        rows.append(
            "<tr>"
            f"<td>{_h(sc.tag)}</td>"
            f"<td>{sc.n_prompts}</td><td>{sc.n_runs}</td><td>{sc.n_evals}</td>"
            f"<td{cls}>{_h(sc.quality_pick or '—')}</td>"
            f"<td>{_pct(sc.quality_pass_rate)}</td>"
            f"<td>{sc.quality_n if sc.quality_n is not None else '—'}</td>"
            f"<td>{_h(sc.cost_pick or '—')}</td>"
            f"<td>{_money(sc.cost_pick_avg_cost)}</td>"
            f"<td>{_h(sc.balanced_pick or '—')}</td>"
            "</tr>"
        )
    return (
        "<h2>Per-tag scorecard</h2><table>"
        "<thead><tr>"
        "<th>tag</th><th>prompts</th><th>runs</th><th>evals</th>"
        "<th>quality pick</th><th>pass</th><th>n</th>"
        "<th>cost pick</th><th>$/run</th><th>balanced</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def _section_gaps_html(rep: Report) -> str:
    if not rep.gaps:
        return ""
    blurbs = {
        "no_runs": "no runs — register a model and run prompts",
        "no_evals": "runs exist but no evals — apply a rubric",
        "no_qualifying": "no model meets pass/sample threshold yet",
    }
    rows = "\n".join(
        f'<tr><td>{_h(g.tag)}</td><td>{g.n_prompts}</td>'
        f'<td class="gap-{g.reason}">{blurbs.get(g.reason, g.reason)}</td></tr>'
        for g in rep.gaps
    )
    return (
        "<h2>Coverage gaps</h2><table>"
        "<thead><tr><th>tag</th><th>prompts</th><th>next step</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _section_freshness_html(rep: Report) -> str:
    if not rep.rubric_freshness:
        return ""
    rows = []
    for rf in rep.rubric_freshness:
        cls = ' class="stale"' if rf.stale else ""
        scored = ", ".join(f"{name}:{n}" for name, n in rf.scored_by_breakdown)
        rows.append(
            f"<tr{cls}>"
            f"<td>{_h(rf.rubric)}</td>"
            f"<td>{rf.n_evals}</td>"
            f"<td>{_h(scored)}</td>"
            f"<td>{rf.last_scored_at.strftime('%Y-%m-%d') if rf.last_scored_at else '—'}</td>"
            f"<td>{'%.0fd' % rf.age_days if rf.age_days is not None else '—'}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    return (
        "<h2>Rubric freshness</h2><table>"
        "<thead><tr><th>rubric</th><th>evals</th><th>scored by</th>"
        "<th>last scored</th><th>age</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _kv_table_html(
    title: str, rows: list[tuple[str, int]], headers: tuple[str, str]
) -> str:
    if not rows:
        return ""
    body = "\n".join(f"<tr><td>{_h(k)}</td><td>{v}</td></tr>" for k, v in rows)
    return (
        f"<h3>{_h(title)}</h3><table>"
        f"<thead><tr><th>{headers[0]}</th><th>{headers[1]}</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _pct(v: float | None) -> str:
    return f"{v:.0%}" if v is not None else "—"


def _money(v: float | None) -> str:
    return f"${v:.4f}" if v is not None else "—"


def _fill_rubric_freshness(rep: Report, staleness_days: float) -> None:
    """For each distinct rubric, capture n_evals, last_scored_at, and a
    `stale` flag if the most recent eval is older than `staleness_days`.
    """
    now = datetime.now(timezone.utc)
    with get_session() as s:
        rows = s.execute(
            select(
                Evaluation.rubric,
                func.count(Evaluation.id),
                func.max(Evaluation.created_at),
            )
            .group_by(Evaluation.rubric)
            .order_by(func.count(Evaluation.id).desc())
        ).all()

        scored_by_rows = s.execute(
            select(Evaluation.rubric, Evaluation.scored_by, func.count(Evaluation.id))
            .group_by(Evaluation.rubric, Evaluation.scored_by)
        ).all()

    scored_by_by_rubric: dict[str, list[tuple[str, int]]] = {}
    for rubric, scored_by, cnt in scored_by_rows:
        scored_by_by_rubric.setdefault(rubric, []).append((scored_by, cnt))

    for rubric, n_evals, last_at in rows:
        # Timestamps come back tz-aware thanks to db/types.py:UTCDateTime.
        age_days = None
        stale = False
        if last_at is not None:
            age_days = (now - last_at).total_seconds() / 86400
            stale = age_days > staleness_days
        rep.rubric_freshness.append(
            RubricFreshness(
                rubric=rubric,
                n_evals=n_evals,
                last_scored_at=last_at,
                age_days=age_days,
                stale=stale,
                scored_by_breakdown=sorted(
                    scored_by_by_rubric.get(rubric, []),
                    key=lambda kv: -kv[1],
                ),
            )
        )

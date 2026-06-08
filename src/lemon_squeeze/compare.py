"""Head-to-head model comparison.

`compare(model_a, model_b, rubric=...)` aggregates per-tag pass rates for two
models over the same authoritative rubric and returns a table with deltas.
The "winner" column highlights where each model dominates — that's the
nugget the project exists to surface.

Per-tag samples are independent; we don't pair runs (different prompts may
have hit each model). That's intentional — pairing would silently shrink the
dataset whenever one model has a run the other doesn't, and the average
behavior across a tag is what informs routing.

Aggregation goes through `aggregations.aggregate_by_tag_model` so the
per-(tag, model) numbers stay in lock-step with router and report.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select

from lemon_squeeze.aggregations import aggregate_by_tag_model, group_by_first_key
from lemon_squeeze.db import Model, get_session
from lemon_squeeze.stats import intervals_disjoint, wilson_interval

DEFAULT_RUBRIC = "human_pass"


@dataclass
class TagComparison:
    tag: str
    a_n: int
    a_pass_rate: float
    a_pass_ci: tuple[float, float]
    a_avg_score: float
    a_avg_cost: float | None
    a_avg_latency: float | None
    b_n: int
    b_pass_rate: float
    b_pass_ci: tuple[float, float]
    b_avg_score: float
    b_avg_cost: float | None
    b_avg_latency: float | None
    delta_pass_rate: float          # a - b (point estimate)
    significant: bool               # CIs don't overlap
    winner: str                     # 'A', 'B', or 'tie'


@dataclass
class ComparisonReport:
    model_a: str
    model_b: str
    rubric: str
    per_tag: list[TagComparison] = field(default_factory=list)
    a_wins: int = 0
    b_wins: int = 0
    ties: int = 0
    overall_winner: str = "tie"  # 'A', 'B', or 'tie'


def compare(
    model_a: str,
    model_b: str,
    *,
    rubric: str = DEFAULT_RUBRIC,
    min_samples: int = 1,
    tie_threshold: float = 0.05,
    require_significance: bool = True,
) -> ComparisonReport:
    """Build per-tag pass-rate comparison between two models.

    `tie_threshold` — pass-rate deltas smaller than this count as ties (so a
    51% vs 50% squeak doesn't show up as a "win" with sparse data).

    `require_significance` (default True) — a model is only declared the winner
    when its 95% Wilson CI doesn't overlap the other's. With few samples the
    CIs are wide; even a "100% vs 50%" delta can be a tie if you have 3 runs.
    Set False to fall back to the point-estimate + tie_threshold heuristic.
    """
    # Validate model existence up front for a clear error message.
    with get_session() as s:
        known = {
            name
            for (name,) in s.execute(
                select(Model.name).where(Model.name.in_([model_a, model_b]))
            ).all()
        }
    if model_a not in known or model_b not in known:
        missing = [n for n in (model_a, model_b) if n not in known]
        raise ValueError(f"unknown model(s): {missing}")

    aggs = aggregate_by_tag_model(
        rubrics=[rubric], model_names=[model_a, model_b]
    )
    by_tag = group_by_first_key(aggs)
    report = ComparisonReport(model_a=model_a, model_b=model_b, rubric=rubric)

    for tag in sorted(by_tag):
        per_model = {a.model_name: a for a in by_tag[tag]}
        a = per_model.get(model_a)
        b = per_model.get(model_b)
        if a is None or b is None:
            continue
        if a.n_evals < min_samples or b.n_evals < min_samples:
            continue

        a_ci = wilson_interval(a.n_passed, a.n_passed_known)
        b_ci = wilson_interval(b.n_passed, b.n_passed_known)
        significant = intervals_disjoint(a_ci, b_ci)
        delta = a.pass_rate - b.pass_rate

        if require_significance and not significant:
            winner = "tie"
        elif abs(delta) <= tie_threshold:
            winner = "tie"
        elif delta > 0:
            winner = "A"
        else:
            winner = "B"

        if winner == "tie":
            report.ties += 1
        elif winner == "A":
            report.a_wins += 1
        else:
            report.b_wins += 1

        report.per_tag.append(
            TagComparison(
                tag=tag,
                a_n=a.n_evals,
                a_pass_rate=a.pass_rate,
                a_pass_ci=a_ci,
                a_avg_score=a.avg_score,
                a_avg_cost=a.avg_cost_usd,
                a_avg_latency=a.avg_latency_ms,
                b_n=b.n_evals,
                b_pass_rate=b.pass_rate,
                b_pass_ci=b_ci,
                b_avg_score=b.avg_score,
                b_avg_cost=b.avg_cost_usd,
                b_avg_latency=b.avg_latency_ms,
                delta_pass_rate=delta,
                significant=significant,
                winner=winner,
            )
        )

    if report.a_wins > report.b_wins:
        report.overall_winner = "A"
    elif report.b_wins > report.a_wins:
        report.overall_winner = "B"
    else:
        report.overall_winner = "tie"
    return report

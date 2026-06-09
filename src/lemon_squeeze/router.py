"""Model router — pick the best model that reliably wins for a given prompt.

Algorithm:
  1. Classify the incoming prompt (heuristic by default; whatever the
     ensemble returns works too).
  2. For each tag the prompt has, look up historical pass rates per model in
     `evaluations` — restricted to the rubrics treated as authoritative (by
     default: `human_pass`).
  3. Filter to models with at least `min_samples` runs against that tag.
  4. Of those, keep the ones whose pass rate ≥ `threshold`.
  5. Score each survivor by a weighted combination of size, cost, and latency
     (all lower-is-better, min-max-normalized across candidates so the weights
     are comparable). Higher score wins.

Default weights bias toward `size` so the router behaves like "smallest
qualifying" by default. Set `weights = RouterWeights(size=0.0, cost=1.0)` to
pick the cheapest, etc.

`recommend()` returns a `Recommendation` with the picked model plus the
support data, so callers can show "we chose X because Y had 80% pass rate over
12 runs at $0.001/run." If no model qualifies, the recommendation includes
`fallback=True` and the highest-pass-rate candidate regardless of constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lemon_squeeze.aggregations import aggregate_by_tag_model
from lemon_squeeze.classification.ensemble import build_default_classifier

DEFAULT_AUTH_RUBRICS = ("human_pass",)
DEFAULT_THRESHOLD = 0.7
DEFAULT_MIN_SAMPLES = 3


@dataclass(frozen=True)
class RouterWeights:
    """Weights for the multi-criteria scoring step. Higher weight = matters more.

    All three axes are lower-is-better (smaller, cheaper, faster). After
    min-max normalization the per-axis score is in [0, 1] where 1 means
    "this is the best on this axis among candidates." The composite score is a
    weighted sum; ties are broken alphabetically.

    `size` measures `size_params_b` (treats unknown as worst case).
    `cost` measures `avg_cost_usd` over the historical runs.
    `latency` measures `avg_latency_ms` over the historical runs.
    """

    size: float = 1.0
    cost: float = 0.0
    latency: float = 0.0

    def normalize(self) -> "RouterWeights":
        total = self.size + self.cost + self.latency
        if total == 0:
            return RouterWeights(size=1.0)
        return RouterWeights(
            size=self.size / total,
            cost=self.cost / total,
            latency=self.latency / total,
        )

    @classmethod
    def from_preset_and_overrides(
        cls,
        preset: str | None = None,
        *,
        size: float | None = None,
        cost: float | None = None,
        latency: float | None = None,
    ) -> "RouterWeights":
        """Build weights by overlaying explicit overrides on a preset.

        `preset` defaults to "size" if None. Validates preset name with a
        ValueError so the same validation lives in one place — both CLI and
        HTTP server delegate here.
        """
        preset = preset or "size"
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
        base = PRESETS[preset]
        return cls(
            size=size if size is not None else base.size,
            cost=cost if cost is not None else base.cost,
            latency=latency if latency is not None else base.latency,
        )


# Backward-compatible default — pure size minimization.
SIZE_ONLY = RouterWeights(size=1.0, cost=0.0, latency=0.0)
BALANCED = RouterWeights(size=0.34, cost=0.33, latency=0.33)
CHEAP = RouterWeights(size=0.0, cost=1.0, latency=0.0)
FAST = RouterWeights(size=0.0, cost=0.0, latency=1.0)

PRESETS: dict[str, RouterWeights] = {
    "size": SIZE_ONLY,
    "balanced": BALANCED,
    "cheap": CHEAP,
    "fast": FAST,
}


@dataclass
class ModelStats:
    model_name: str
    size_params_b: float | None
    context_window: int | None
    sample_count: int
    pass_rate: float
    avg_score: float
    avg_cost_usd: float | None
    avg_latency_ms: float | None
    composite_score: float | None = None  # populated after scoring


@dataclass
class Recommendation:
    prompt: str
    tags: list[str]
    picked: ModelStats | None
    fallback: bool = False
    candidates: list[ModelStats] = field(default_factory=list)
    reason: str = ""
    weights: RouterWeights = field(default_factory=lambda: SIZE_ONLY)


def recommend(
    prompt: str,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    authoritative_rubrics: tuple[str, ...] = DEFAULT_AUTH_RUBRICS,
    weights: RouterWeights | str = SIZE_ONLY,
) -> Recommendation:
    """Classify `prompt` and recommend the best-scoring qualifying model.

    `weights` is either a RouterWeights instance or a preset name
    ("size" / "balanced" / "cheap" / "fast").
    """
    if isinstance(weights, str):
        if weights not in PRESETS:
            raise ValueError(f"unknown preset {weights!r}; known: {sorted(PRESETS)}")
        weights = PRESETS[weights]
    weights_n = weights.normalize()

    # Classify with the full ensemble (heuristic + trained ML + optional LLM),
    # not just the heuristic. This closes the project's feedback loop:
    # accumulate labels -> train ML -> the router actually uses what it
    # learned. Until now recommend() called HeuristicClassifier() directly,
    # so a trained ML model that could tag e.g. "reasoning" (a category the
    # heuristic has no signal for) was consulted by `lemon classify ask` but
    # ignored by `lemon route pick`. Dedupe by tag, highest confidence first.
    preds = sorted(
        build_default_classifier().predict(prompt), key=lambda p: -p.confidence
    )
    tags: list[str] = []
    for p in preds:
        if p.tag != "unknown" and p.tag not in tags:
            tags.append(p.tag)
    stats = stats_by_tag(tags, authoritative_rubrics=authoritative_rubrics)

    if not stats:
        return Recommendation(
            prompt=prompt,
            tags=tags,
            picked=None,
            reason="no historical evaluations for these tags",
            weights=weights_n,
        )

    qualifying = [s for s in stats if s.sample_count >= min_samples and s.pass_rate >= threshold]
    if qualifying:
        scores = _score_candidates(qualifying, weights_n)
        for s, score in zip(qualifying, scores, strict=True):
            s.composite_score = score
        # Highest composite wins; alphabetical tiebreak.
        picked = max(qualifying, key=lambda s: (s.composite_score or 0.0, -ord(s.model_name[0])))
        return Recommendation(
            prompt=prompt,
            tags=tags,
            picked=picked,
            candidates=stats,
            weights=weights_n,
            reason=(
                f"best composite score under weights "
                f"(size={weights_n.size:.2f}, cost={weights_n.cost:.2f}, "
                f"latency={weights_n.latency:.2f}) among models with pass_rate "
                f">= {threshold:.0%} over >= {min_samples} runs on tags {tags}"
            ),
        )

    # No model qualifies — fall back to the one with the highest pass rate.
    stats_sorted = sorted(stats, key=lambda s: (-s.pass_rate, -s.sample_count, s.model_name))
    return Recommendation(
        prompt=prompt,
        tags=tags,
        picked=stats_sorted[0],
        fallback=True,
        candidates=stats,
        weights=weights_n,
        reason=(
            f"no model met threshold {threshold:.0%} with >= {min_samples} samples; "
            "falling back to best available"
        ),
    )


def stats_by_tag(
    tags: list[str],
    *,
    authoritative_rubrics: tuple[str, ...] = DEFAULT_AUTH_RUBRICS,
) -> list[ModelStats]:
    """Aggregate per-model stats across runs whose prompts carry any of `tags`.

    Multiple `tags` are OR-ed: a prompt counts toward this query if it carries
    any of them. Per-model rows are then merged across tags (so the same model
    that has runs on tags A and B yields one row whose `sample_count` is the
    combined total).
    """
    if not tags:
        return []

    aggs = aggregate_by_tag_model(
        rubrics=authoritative_rubrics, tags=tags,
    )
    if not aggs:
        return []

    # The aggregation is per-(tag, model); we want per-model. Merge across tags.
    # We can't merge avg_cost/avg_latency directly without per-row weights, so
    # we compute a sample-count-weighted mean.
    merged: dict[str, dict] = {}
    for a in aggs:
        m = merged.setdefault(
            a.model_name,
            {
                "size_b": a.model_size_b,
                "ctx": a.model_context_window,
                "n_evals": 0,
                "n_passed": 0,
                "n_passed_known": 0,
                "score_sum": 0.0,
                "cost_weighted": 0.0,
                "cost_n": 0,
                "lat_weighted": 0.0,
                "lat_n": 0,
            },
        )
        m["n_evals"] += a.n_evals
        m["n_passed"] += a.n_passed
        m["n_passed_known"] += a.n_passed_known
        m["score_sum"] += a.avg_score * a.n_evals
        if a.avg_cost_usd is not None:
            m["cost_weighted"] += a.avg_cost_usd * a.n_evals
            m["cost_n"] += a.n_evals
        if a.avg_latency_ms is not None:
            m["lat_weighted"] += a.avg_latency_ms * a.n_evals
            m["lat_n"] += a.n_evals

    return [
        ModelStats(
            model_name=name,
            size_params_b=m["size_b"],
            context_window=m["ctx"],
            sample_count=m["n_evals"],
            pass_rate=(
                m["n_passed"] / m["n_passed_known"] if m["n_passed_known"] else 0.0
            ),
            avg_score=m["score_sum"] / m["n_evals"] if m["n_evals"] else 0.0,
            avg_cost_usd=(
                m["cost_weighted"] / m["cost_n"] if m["cost_n"] else None
            ),
            avg_latency_ms=(
                m["lat_weighted"] / m["lat_n"] if m["lat_n"] else None
            ),
        )
        for name, m in merged.items()
    ]


def _score_candidates(
    candidates: list[ModelStats], weights: RouterWeights
) -> list[float]:
    """Min-max normalize each axis to [0, 1] (1 = best), then weighted sum.

    Returns a list of composite scores parallel to `candidates`. (We use a
    parallel list rather than a {candidate: score} dict because ModelStats
    isn't hashable — mutable dataclasses opt out of __hash__.)
    """
    known_sizes = [c.size_params_b for c in candidates if c.size_params_b is not None]
    unknown_size_fallback = max(known_sizes) if known_sizes else 1e9
    sizes = [
        s.size_params_b if s.size_params_b is not None else unknown_size_fallback
        for s in candidates
    ]
    costs = [s.avg_cost_usd or 0.0 for s in candidates]
    lats = [s.avg_latency_ms or 0.0 for s in candidates]

    size_axis = _normalize_lower_is_better(sizes)
    cost_axis = _normalize_lower_is_better(costs)
    lat_axis = _normalize_lower_is_better(lats)

    return [
        weights.size * size_axis[i]
        + weights.cost * cost_axis[i]
        + weights.latency * lat_axis[i]
        for i in range(len(candidates))
    ]


def _normalize_lower_is_better(values: list[float]) -> list[float]:
    """Map values such that the smallest → 1.0, the largest → 0.0.

    If all values are equal, everyone gets 1.0 (this axis doesn't differentiate).
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(hi - v) / (hi - lo) for v in values]



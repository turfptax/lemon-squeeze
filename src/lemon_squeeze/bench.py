"""Benchmark runner — ties prompts + rubrics + models into one operation.

A benchmark is a directory:

    benchmarks/<name>/
      prompts/<category>.jsonl     # each line: {prompt, intended_tag, expected_contains?}
      rubrics/*.yaml               # optional; standard Rubric YAML files

`bench load(dir)` ingests every .jsonl in prompts/ (using SeedFileIngester,
which preserves all metadata fields including `expected_contains`).

`bench run(dir, model_names)`:
    1. Loads the bench (deduped against existing prompts)
    2. Runs each prompt against each model (parallel fanout)
    3. Per-prompt deterministic scoring: if a prompt has `expected_contains`
       in its source_metadata, write an Evaluation with rubric
       `bench:expected_contains` — score = fraction matched, passed = all matched
    4. Applies any rubrics in rubrics/ on top
    5. Returns a BenchReport with per-category pass rates

The per-prompt `expected_contains` scoring is the value-add over the generic
Rubric framework: rubrics treat all matching runs uniformly, but bench prompts
each have their own ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select

from lemon_squeeze.aggregations import aggregate_by_intended_tag_model
from lemon_squeeze.db import Prompt, PromptTag, Run, get_session
from lemon_squeeze.eval.rubric import Rubric, evaluate_runs
from lemon_squeeze.eval.runner import fanout
from lemon_squeeze.ingestion.self_generated import SeedFileIngester

BENCH_EXPECTED_RUBRIC = "bench:expected_contains"
BENCH_TAG_CLASSIFIER = "bench"

# The standard rubric used to score bench prompts against their per-prompt
# `expected_contains` ground truth. Lives here rather than in rubrics/ so it
# stays in lockstep with the bench code that depends on its name.
_BENCH_RUBRIC = Rubric(
    name=BENCH_EXPECTED_RUBRIC,
    description="Response contains all `expected_contains` substrings from the prompt's metadata.",
    judge_kind="expected_contains",
    judge_config={"metadata_key": "expected_contains", "on_missing": "skip"},
)


@dataclass
class CategoryStat:
    category: str
    model_name: str
    n_runs: int
    pass_count: int
    avg_score: float
    pass_rate: float
    avg_cost_usd: float | None = None      # mean cost per run
    avg_latency_ms: float | None = None    # mean wall time per run

    @property
    def cost_per_pass(self) -> float | None:
        """Expected cost to get one successful run on this task.

        Derived from `avg_cost_usd / pass_rate`; None when either cost is
        unknown or no runs passed (division by zero).
        """
        if self.avg_cost_usd is None or self.pass_rate <= 0:
            return None
        return self.avg_cost_usd / self.pass_rate


@dataclass
class BenchReport:
    bench_name: str
    prompts_loaded: int = 0
    prompts_deduped: int = 0
    runs_attempted: int = 0
    runs_succeeded: int = 0
    runs_failed: int = 0
    expected_evals_written: int = 0
    rubric_evals_written: int = 0
    per_category: list[CategoryStat] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


_BENCH_DIR_META_KEY = "lemon_bench_dir"


def _bench_dir_marker(bench_dir: Path) -> str:
    """Canonical string for marking which bench a prompt came from. We
    resolve+as_posix so the same bench dir compares equal regardless of how
    the user spelled the path (relative vs absolute, Windows backslashes)."""
    return Path(bench_dir).resolve().as_posix()


def load(bench_dir: Path) -> tuple[int, int]:
    """Ingest every prompts/*.jsonl in the bench directory.

    Returns (inserted, duplicates) across all files. Each prompt's
    source_metadata gets a `lemon_bench_dir` field marking which bench it
    came from, so two benches that happen to share filenames (e.g. both
    have `coding.jsonl`) don't bleed into each other.
    """
    bench_dir = Path(bench_dir)
    prompts_dir = bench_dir / "prompts"
    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"no prompts/ subdirectory in {bench_dir}")

    overlay = {_BENCH_DIR_META_KEY: _bench_dir_marker(bench_dir)}
    inserted = 0
    duplicates = 0
    for jsonl in sorted(prompts_dir.glob("*.jsonl")):
        result = SeedFileIngester(jsonl, metadata_overlay=overlay).run()
        inserted += result.inserted
        duplicates += result.duplicates
    _tag_intended(bench_dir)
    return inserted, duplicates


def _tag_intended(bench_dir: Path) -> int:
    """Write ground-truth PromptTag rows from each bench prompt's
    `intended_tag` metadata (classifier="bench", confidence=1.0).

    Bench JSONL files declare their category outright, but until this step
    the only PromptTags came from the heuristic/ML classifiers guessing it
    back from the prompt text -- and guessing imperfectly. In a real run of
    benchmarks/starter, the 4 reasoning prompts all landed under "unknown"
    and 3 of 5 math prompts were missed, so every per-tag surface (report
    scorecard, route pick, dashboard heatmap) was aggregating over wrong
    tags while bench's own per-category table used the truth. Idempotent:
    skips (prompt, tag) pairs that already have a bench-classifier row.
    """
    marker = _bench_dir_marker(bench_dir)
    written = 0
    with get_session() as s:
        existing = {
            (pt.prompt_id, pt.tag)
            for pt in s.query(PromptTag).filter(
                PromptTag.classifier == BENCH_TAG_CLASSIFIER
            )
        }
        rows = s.execute(
            select(Prompt.id, Prompt.source_metadata).where(
                Prompt.source == "self_generated:seed"
            )
        ).all()
        for pid, meta in rows:
            if not meta or meta.get(_BENCH_DIR_META_KEY) != marker:
                continue
            tag = meta.get("intended_tag")
            if not isinstance(tag, str) or not tag:
                continue
            if (pid, tag) in existing:
                continue
            s.add(
                PromptTag(
                    prompt_id=pid,
                    tag=tag,
                    classifier=BENCH_TAG_CLASSIFIER,
                    confidence=1.0,
                )
            )
            written += 1
    return written


def run(
    bench_dir: Path,
    model_names: list[str] | None = None,
    *,
    max_workers: int = 4,
    skip_existing: bool = True,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> BenchReport:
    """Load → fanout → per-prompt scoring → apply rubrics → report."""
    bench_dir = Path(bench_dir)
    report = BenchReport(bench_name=bench_dir.name)

    inserted, deduped = load(bench_dir)
    report.prompts_loaded = inserted
    report.prompts_deduped = deduped

    # Collect the prompt IDs for prompts whose source is one of our JSONL files.
    bench_prompt_ids = _bench_prompt_ids(bench_dir)
    if not bench_prompt_ids:
        report.errors.append("no prompts found in bench after load")
        return report

    fan = fanout(
        prompt_ids=bench_prompt_ids,
        model_names=model_names,
        temperature=temperature,
        max_tokens=max_tokens,
        skip_existing=skip_existing,
        max_workers=max_workers,
    )
    report.runs_attempted = fan.attempted
    report.runs_succeeded = fan.succeeded
    report.runs_failed = fan.failed
    report.errors.extend(fan.errors[:10])

    # Score per-prompt `expected_contains` ground truth via the standard
    # rubric framework — no special-case scorer required.
    with get_session() as s:
        run_ids = [
            row[0]
            for row in s.execute(
                select(Run.id).where(Run.prompt_id.in_(bench_prompt_ids))
            ).all()
        ]
    if run_ids:
        er = evaluate_runs(
            _BENCH_RUBRIC,
            run_ids=run_ids,
            skip_existing=skip_existing,
        )
        report.expected_evals_written = er.evaluations_written

    # Apply category-level rubrics if any.
    rubrics_dir = bench_dir / "rubrics"
    if rubrics_dir.is_dir():
        for ry in sorted(rubrics_dir.glob("*.yaml")):
            rubric = Rubric.from_file(ry)
            er = evaluate_runs(rubric, skip_existing=skip_existing)
            report.rubric_evals_written += er.evaluations_written

    report.per_category = _per_category_breakdown(bench_prompt_ids)
    return report


def _bench_prompt_ids(bench_dir: Path) -> list[int]:
    """Collect Prompt IDs that came from this bench's JSONL files.

    Filters by `source_metadata[_BENCH_DIR_META_KEY]` matching the resolved
    bench dir. Previously this filtered by `source_ref.startswith(filename)`,
    which collided across benches sharing filename conventions (e.g. both
    `benchmarks/starter` and a user's custom bench having `coding.jsonl`):
    each bench would see the union of all matching prompts and downstream
    per-category breakdowns would over-count.

    Legacy prompts ingested before this marker existed have no
    `lemon_bench_dir` key and won't be matched here. They'd need to be
    re-loaded or migrated by hand if the user still wants them counted.
    """
    marker = _bench_dir_marker(bench_dir)
    with get_session() as session:
        # JSON-path filtering varies across DBs; load and filter in Python.
        # This is called once per bench operation, not per request, so the
        # O(n_prompts) overhead is fine.
        rows = session.execute(
            select(Prompt.id, Prompt.source_metadata).where(
                Prompt.source == "self_generated:seed"
            )
        ).all()
        return [
            pid for pid, meta in rows
            if meta and meta.get(_BENCH_DIR_META_KEY) == marker
        ]


def _per_category_breakdown(prompt_ids: list[int]) -> list[CategoryStat]:
    """Group bench results by (intended_tag, model) and compute pass rates."""
    if not prompt_ids:
        return []

    aggs = aggregate_by_intended_tag_model(
        rubrics=[BENCH_EXPECTED_RUBRIC],
        prompt_ids=prompt_ids,
    )
    out = [
        CategoryStat(
            category=a.tag,
            model_name=a.model_name,
            n_runs=a.n_evals,
            pass_count=a.n_passed,
            avg_score=a.avg_score,
            pass_rate=a.pass_rate,
            avg_cost_usd=a.avg_cost_usd,
            avg_latency_ms=a.avg_latency_ms,
        )
        for a in aggs
    ]
    out.sort(key=lambda s: (s.category, -s.pass_rate, s.model_name))
    return out

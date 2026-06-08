"""Setup health check — `lemon doctor`.

Walks every prerequisite a fresh install needs, reporting OK / WARN / FAIL with
a remediation hint. Each check is a small dataclass so we can grow the list
without touching the formatter.

DB-touching checks share a single session and pre-computed counts, so the whole
diagnostic is one connection round-trip instead of N.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from lemon_squeeze.classification.ml import MODEL_PATH as ML_MODEL_PATH
from lemon_squeeze.config import settings
from lemon_squeeze.db import (
    Evaluation,
    Model,
    Prompt,
    PromptTag,
    TagTaxonomy,
    get_session,
)

Status = Literal["ok", "warn", "fail"]


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str
    hint: str | None = None


@dataclass
class _DbCounts:
    """Pre-computed counts gathered in one session. Populated by `_gather_counts`."""

    schema_ok: bool
    schema_error: str | None
    taxonomy: int
    prompts: int
    tagged_prompts: int
    models: int
    evaluations: int


def run_all_checks() -> list[CheckResult]:
    counts = _gather_counts()
    results: list[CheckResult] = [
        _check_env_file(),
        _check_db_path_writable(),
        _check_schema(counts),
        _check_taxonomy(counts),
        _check_prompts(counts),
        _check_models(counts),
        _check_classification_coverage(counts),
        _check_ml_classifier_present(),
        _check_evals(counts),
        _check_openrouter_or_lmstudio(),
    ]
    return results


def _gather_counts() -> _DbCounts:
    """One session, all counts. If the schema is missing, return zeros + the error."""
    try:
        with get_session() as s:
            return _DbCounts(
                schema_ok=True,
                schema_error=None,
                taxonomy=s.scalar(select(func.count()).select_from(TagTaxonomy)) or 0,
                prompts=s.scalar(select(func.count()).select_from(Prompt)) or 0,
                tagged_prompts=(
                    s.scalar(select(func.count(func.distinct(PromptTag.prompt_id)))) or 0
                ),
                models=s.scalar(select(func.count()).select_from(Model)) or 0,
                evaluations=s.scalar(select(func.count()).select_from(Evaluation)) or 0,
            )
    except OperationalError as e:
        return _DbCounts(
            schema_ok=False,
            schema_error=str(e).split("\n")[0],
            taxonomy=0, prompts=0, tagged_prompts=0, models=0, evaluations=0,
        )


# ---------- non-DB checks ----------------------------------------------------


def _check_env_file() -> CheckResult:
    env = settings.model_config.get("env_file") if hasattr(settings, "model_config") else None
    if env is None:
        return CheckResult("env file", "warn", "no env_file configured")
    if not Path(env).exists():
        return CheckResult(
            "env file", "warn", f"{env} not found",
            "Copy .env.example to .env and fill in any keys you'll use.",
        )
    return CheckResult("env file", "ok", f"loaded from {env}")


def _check_db_path_writable() -> CheckResult:
    parent = settings.db_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return CheckResult(
            "db parent dir", "fail", f"can't create {parent}: {e}",
            "Ensure the path in LEMON_DB_PATH is writable.",
        )
    return CheckResult("db parent dir", "ok", f"{parent} writable")


def _check_openrouter_or_lmstudio() -> CheckResult:
    """We don't ping the endpoints — just sanity-check that *something* is configured."""
    or_key = settings.openrouter_api_key
    if or_key and or_key.strip() and or_key.strip() != "your_key_here":
        return CheckResult("model provider", "ok", "OpenRouter key configured")
    return CheckResult(
        "model provider", "warn",
        "OpenRouter key absent (LM Studio default URL only)",
        "Set OPENROUTER_API_KEY in .env if you want remote models.",
    )


def _check_ml_classifier_present() -> CheckResult:
    if ML_MODEL_PATH.exists():
        return CheckResult("ml classifier", "ok", f"trained model at {ML_MODEL_PATH.name}")
    return CheckResult(
        "ml classifier", "warn", "not trained yet",
        "Once you have ≥3 labels per category, run `lemon classify train-ml`.",
    )


# ---------- DB-derived checks (consume `_DbCounts`) --------------------------


def _check_schema(c: _DbCounts) -> CheckResult:
    if not c.schema_ok:
        return CheckResult("schema", "fail", c.schema_error or "unknown error", "Run `lemon db init`.")
    return CheckResult("schema", "ok", "tables present")


def _check_taxonomy(c: _DbCounts) -> CheckResult:
    if not c.schema_ok:
        return CheckResult("taxonomy", "fail", "schema missing", "Run `lemon db init`.")
    if c.taxonomy == 0:
        return CheckResult(
            "taxonomy", "warn", "no taxonomy rows",
            "Re-run `lemon db init` to seed default tags.",
        )
    return CheckResult("taxonomy", "ok", f"{c.taxonomy} tags seeded")


def _check_prompts(c: _DbCounts) -> CheckResult:
    if not c.schema_ok:
        return CheckResult("prompts", "fail", "schema missing", "Run `lemon db init`.")
    if c.prompts == 0:
        return CheckResult(
            "prompts", "warn", "no prompts ingested",
            "Try `lemon bench load benchmarks/starter` or `lemon ingest ai-harness`.",
        )
    return CheckResult("prompts", "ok", f"{c.prompts} prompts in DB")


def _check_models(c: _DbCounts) -> CheckResult:
    if not c.schema_ok:
        return CheckResult("models", "fail", "schema missing", "Run `lemon db init`.")
    if c.models == 0:
        return CheckResult(
            "models", "warn", "no models registered",
            "Register with `lemon models register <provider>/<name> --size-b N`.",
        )
    return CheckResult("models", "ok", f"{c.models} models registered")


def _check_classification_coverage(c: _DbCounts) -> CheckResult:
    if not c.schema_ok:
        return CheckResult("classification", "fail", "schema missing")
    if c.prompts == 0:
        return CheckResult(
            "classification", "warn", "no prompts to classify",
            "Ingest prompts first.",
        )
    if c.tagged_prompts == 0:
        return CheckResult(
            "classification", "warn", "no prompts have tags yet",
            "Run `lemon classify run`.",
        )
    pct = c.tagged_prompts / c.prompts
    if pct < 0.5:
        return CheckResult(
            "classification", "warn",
            f"only {c.tagged_prompts}/{c.prompts} prompts tagged ({pct:.0%})",
            "Run `lemon classify run` to back-fill.",
        )
    return CheckResult(
        "classification", "ok",
        f"{c.tagged_prompts}/{c.prompts} prompts tagged ({pct:.0%})",
    )


def _check_evals(c: _DbCounts) -> CheckResult:
    if not c.schema_ok:
        return CheckResult("evaluations", "fail", "schema missing")
    if c.evaluations == 0:
        return CheckResult(
            "evaluations", "warn", "no evaluations yet",
            "Apply a rubric with `lemon eval score rubrics/<name>.yaml`.",
        )
    return CheckResult("evaluations", "ok", f"{c.evaluations} evaluations recorded")


# ---------- summary ---------------------------------------------------------


def summarize(results: list[CheckResult]) -> tuple[int, int, int]:
    """(ok, warn, fail) counts."""
    ok = sum(1 for r in results if r.status == "ok")
    warn = sum(1 for r in results if r.status == "warn")
    fail = sum(1 for r in results if r.status == "fail")
    return ok, warn, fail

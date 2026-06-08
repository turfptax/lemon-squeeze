"""HTTP server — `lemon serve` exposes the router + report as a REST API.

Production apps integrate Lemon Squeeze via HTTP rather than importing Python.
The endpoints mirror the CLI surface:

  GET  /healthz                  — liveness probe (returns {"ok": true})
  GET  /models                   — list registered models
  POST /route                    — recommendation for a prompt
  POST /classify                 — predict tags for a prompt
  GET  /report                   — JSON executive summary
  POST /compare                  — head-to-head between two models

FastAPI is an optional extra (`pip install -e ".[server]"`); the bare project
doesn't import this module so the dep stays optional. If you `lemon serve`
without the extra installed, the CLI exits with an install hint.
"""
from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover - tested elsewhere
    raise ImportError(
        "lemon_squeeze.server requires fastapi + uvicorn. "
        "Install with: pip install -e \".[server]\""
    ) from e

import threading
from collections import Counter
from dataclasses import asdict

from sqlalchemy import select

from lemon_squeeze.cache import cache_stats
from lemon_squeeze.classification import build_default_classifier
from lemon_squeeze.compare import compare as compare_models
from lemon_squeeze.db import Model, get_session
from lemon_squeeze.report import build_report, headline_stats
from lemon_squeeze.router import PRESETS, RouterWeights, recommend


# ---------- Request / response models ---------------------------------------


class RouteRequest(BaseModel):
    prompt: str
    threshold: float = 0.7
    min_samples: int = 3
    rubrics: list[str] = Field(default_factory=lambda: ["human_pass"])
    preset: str | None = None  # 'size' / 'balanced' / 'cheap' / 'fast'
    weights: dict[str, float] | None = None  # explicit override; keys: size/cost/latency


class ClassifyRequest(BaseModel):
    prompt: str


class CompareRequest(BaseModel):
    model_a: str
    model_b: str
    rubric: str = "human_pass"
    min_samples: int = 1
    tie_threshold: float = 0.05
    require_significance: bool = True


# ---------- App ----------------------------------------------------------------


def create_app() -> FastAPI:
    """Factory so tests can grab a fresh instance per case."""
    from lemon_squeeze import __version__
    app = FastAPI(title="Lemon Squeeze", version=__version__)

    # Per-app request counter (thread-safe). Each `create_app()` gets its own,
    # which makes the metrics endpoint easy to test in isolation.
    request_counts: Counter[str] = Counter()
    counts_lock = threading.Lock()

    @app.middleware("http")
    async def _count_requests(request, call_next):
        path = request.url.path
        with counts_lock:
            request_counts[path] += 1
        return await call_next(request)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        h = headline_stats()
        with counts_lock:
            requests_snapshot = dict(request_counts)
        return {
            "db": {
                "prompts": h.n_prompts,
                "models": h.n_models,
                "runs": h.n_runs,
                "evaluations": h.n_evals,
                "runs_with_error": h.n_runs_with_error,
                "total_cost_usd": h.total_cost_usd,
            },
            "requests_by_path": requests_snapshot,
            "caches": {name: asdict(s) for name, s in cache_stats().items()},
        }

    @app.get("/models")
    def list_models() -> dict[str, Any]:
        with get_session() as s:
            models = list(s.scalars(select(Model)).all())
        return {
            "models": [
                {
                    "name": m.name,
                    "provider": m.provider,
                    "family": m.family,
                    "size_params_b": m.size_params_b,
                    "context_window": m.context_window,
                    "local": m.local,
                    "cost_in_per_mtok": m.cost_in_per_mtok,
                    "cost_out_per_mtok": m.cost_out_per_mtok,
                }
                for m in models
            ]
        }

    @app.post("/route")
    def route(req: RouteRequest) -> dict[str, Any]:
        try:
            weights = _resolve_weights(req.preset, req.weights)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        rec = recommend(
            req.prompt,
            threshold=req.threshold,
            min_samples=req.min_samples,
            authoritative_rubrics=tuple(req.rubrics),
            weights=weights,
        )
        return _recommendation_to_dict(rec)

    @app.post("/classify")
    def classify(req: ClassifyRequest) -> dict[str, Any]:
        classifier = build_default_classifier()
        preds = classifier.predict(req.prompt)
        return {
            "predictions": [
                {"tag": p.tag, "confidence": p.confidence, "classifier": p.classifier}
                for p in preds
            ]
        }

    @app.get("/report")
    def report() -> dict[str, Any]:
        return build_report().to_dict()

    @app.post("/compare")
    def compare(req: CompareRequest) -> dict[str, Any]:
        try:
            report = compare_models(
                req.model_a,
                req.model_b,
                rubric=req.rubric,
                min_samples=req.min_samples,
                tie_threshold=req.tie_threshold,
                require_significance=req.require_significance,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _comparison_to_dict(report)

    return app


def _resolve_weights(
    preset: str | None,
    weights: dict[str, float] | None,
) -> RouterWeights:
    """Combine a preset name (or `None`) with optional explicit overrides.

    Both CLI and server route preset validation through
    `RouterWeights.from_preset_and_overrides`, so the "unknown preset" error
    message stays in lockstep.
    """
    w = weights or {}
    return RouterWeights.from_preset_and_overrides(
        preset,
        size=w.get("size"),
        cost=w.get("cost"),
        latency=w.get("latency"),
    )


def _recommendation_to_dict(rec) -> dict[str, Any]:
    return {
        "prompt": rec.prompt,
        "tags": rec.tags,
        "picked": _model_stats_to_dict(rec.picked) if rec.picked else None,
        "fallback": rec.fallback,
        "reason": rec.reason,
        "weights": {
            "size": rec.weights.size,
            "cost": rec.weights.cost,
            "latency": rec.weights.latency,
        },
        "candidates": [_model_stats_to_dict(c) for c in rec.candidates],
    }


def _model_stats_to_dict(stats) -> dict[str, Any]:
    return {
        "model_name": stats.model_name,
        "size_params_b": stats.size_params_b,
        "context_window": stats.context_window,
        "sample_count": stats.sample_count,
        "pass_rate": stats.pass_rate,
        "avg_score": stats.avg_score,
        "avg_cost_usd": stats.avg_cost_usd,
        "avg_latency_ms": stats.avg_latency_ms,
        "composite_score": stats.composite_score,
    }


def _comparison_to_dict(report) -> dict[str, Any]:
    return {
        "model_a": report.model_a,
        "model_b": report.model_b,
        "rubric": report.rubric,
        "a_wins": report.a_wins,
        "b_wins": report.b_wins,
        "ties": report.ties,
        "overall_winner": report.overall_winner,
        "per_tag": [
            {
                "tag": tc.tag,
                "a_n": tc.a_n,
                "a_pass_rate": tc.a_pass_rate,
                "a_pass_ci": list(tc.a_pass_ci),
                "a_avg_cost": tc.a_avg_cost,
                "a_avg_latency": tc.a_avg_latency,
                "b_n": tc.b_n,
                "b_pass_rate": tc.b_pass_rate,
                "b_pass_ci": list(tc.b_pass_ci),
                "b_avg_cost": tc.b_avg_cost,
                "b_avg_latency": tc.b_avg_latency,
                "delta_pass_rate": tc.delta_pass_rate,
                "significant": tc.significant,
                "winner": tc.winner,
            }
            for tc in report.per_tag
        ],
    }


# Module-level singleton — uvicorn imports lemon_squeeze.server:app.
app = create_app()

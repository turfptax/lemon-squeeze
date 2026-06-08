"""Run executor — fan a prompt across registered models, persist Run rows.

Key responsibilities:
  * Look up Prompt + Model records.
  * Call the right provider client (LM Studio or OpenRouter, inferred from
    the `Model.provider` field).
  * Record everything in a `Run`, including errors (so we can compare which
    models choke on which prompts).

Errors are captured, not raised: a model timing out is a data point, not a
crash. The executor returns a `RunReport` summarizing the batch.
"""
from __future__ import annotations

import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import httpx
from sqlalchemy import select

from lemon_squeeze.db import Model, Prompt, Run, get_session
from lemon_squeeze.eval.clients import ChatClient

DEFAULT_MAX_WORKERS = 4


@dataclass
class RunReport:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    run_ids: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _provider_for_model(model: Model) -> str:
    """Map a Model row to a ChatClient provider id."""
    if model.local:
        return "lm_studio"
    if model.provider in ("lm_studio", "openrouter"):
        return model.provider
    # anthropic/openai/google/meta-llama and friends route through OpenRouter.
    return "openrouter"


def execute_run(
    prompt: Prompt,
    model: Model,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Run:
    """Execute one (prompt, model) pair, persist a Run row, return the row."""
    provider = _provider_for_model(model)
    client = ChatClient(provider)  # type: ignore[arg-type]

    run = Run(
        prompt_id=prompt.id,
        model_id=model.id,
        temperature=temperature,
        run_metadata={"system": system, **(extra_metadata or {})} or None,
    )

    try:
        result = client.chat(
            model.name,
            prompt.content,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            cost_in_per_mtok=model.cost_in_per_mtok,
            cost_out_per_mtok=model.cost_out_per_mtok,
        )
        run.response = result.text
        run.tokens_in = result.tokens_in
        run.tokens_out = result.tokens_out
        run.latency_ms = result.latency_ms
        run.cost_usd = result.cost_usd
    except httpx.HTTPError as e:
        run.error = f"http_error: {e}"
    except (KeyError, ValueError, TypeError) as e:
        run.error = f"parse_error: {e!r}"

    with get_session() as session:
        session.add(run)
        session.flush()
        run_id = run.id
    # Re-fetch detached row inside a fresh session so callers get a live object.
    with get_session() as session:
        return session.get(Run, run_id)  # type: ignore[return-value]


def fanout(
    prompt_ids: Iterable[int] | None = None,
    model_names: Iterable[str] | None = None,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    skip_existing: bool = True,
    max_workers: int = DEFAULT_MAX_WORKERS,
    progress: bool = False,
) -> RunReport:
    """Execute every (prompt, model) combination concurrently.

    `skip_existing` — if there's already a run for this (prompt, model) pair, skip
    it. Set False to force re-runs (e.g. after changing temperature).
    `max_workers` — concurrent HTTP calls. 1 falls back to a sequential loop
    (useful for debugging). Defaults to 4. Higher is fine for OpenRouter; LM
    Studio usually wants 1–2 since a single local model serves serially.
    """
    report = RunReport()
    lock = threading.Lock()  # guards report mutation

    work_items = _build_work_items(prompt_ids, model_names, skip_existing)
    if not work_items:
        return report

    def _do_one(item: tuple[Prompt, Model]) -> None:
        prompt, model = item
        try:
            run = execute_run(
                prompt,
                model,
                system=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:  # last-resort safety net
            with lock:
                report.failed += 1
                report.errors.append(
                    f"{model.name} on prompt {prompt.id}: unhandled {e!r}"
                )
            return
        with lock:
            if run.error:
                report.failed += 1
                report.errors.append(
                    f"{model.name} on prompt {prompt.id}: {run.error}"
                )
            else:
                report.succeeded += 1
            report.run_ids.append(run.id)

    # Caller asks for attempted up-front so they can see scale even on slow runs.
    with lock:
        report.attempted = len(work_items)

    if max_workers <= 1:
        for item in work_items:
            _do_one(item)
        return report

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_do_one, item) for item in work_items]
        if progress:
            # Drain as_completed for the side-effect of letting callers tail
            # progress; the count is already on report.attempted.
            for _ in as_completed(futures):
                pass
        else:
            for f in as_completed(futures):
                f.result()  # surface any unhandled exception (shouldn't happen)
    return report


def _build_work_items(
    prompt_ids: Iterable[int] | None,
    model_names: Iterable[str] | None,
    skip_existing: bool,
) -> list[tuple[Prompt, Model]]:
    """Build the cartesian (prompt, model) work list, applying skip_existing."""
    with get_session() as session:
        prompt_q = select(Prompt)
        if prompt_ids is not None:
            prompt_q = prompt_q.where(Prompt.id.in_(list(prompt_ids)))
        prompts = list(session.scalars(prompt_q).all())

        model_q = select(Model)
        if model_names is not None:
            model_q = model_q.where(Model.name.in_(list(model_names)))
        models = list(session.scalars(model_q).all())

        existing: set[tuple[int, int]] = set()
        if skip_existing:
            for row in session.execute(select(Run.prompt_id, Run.model_id)).all():
                existing.add((row[0], row[1]))

        # Scalar columns are already loaded by `scalars().all()`; with
        # `expire_on_commit=False` (set in db/session.py) they stay accessible
        # after the session closes. Worker threads only read scalar columns —
        # no lazy-loaded relationships — so detachment is safe.

    return [
        (p, m)
        for p in prompts
        for m in models
        if not skip_existing or (p.id, m.id) not in existing
    ]

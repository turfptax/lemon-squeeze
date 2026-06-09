"""Lemon Squeeze CLI — `lemon ...`"""
from __future__ import annotations

import re
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import func, select

import lemon_squeeze
from lemon_squeeze.classification.ensemble import build_default_classifier, classify_unlabeled
from lemon_squeeze.classification.ml import MLClassifier
from lemon_squeeze.db import Model, Prompt, PromptTag, Run, get_session, init_db
from lemon_squeeze.ingestion.ai_harness import AIHarnessImporter
from lemon_squeeze.ingestion.claude_export import ClaudeExportIngester
from lemon_squeeze.ingestion.grok_export import GrokExportIngester
from lemon_squeeze.ingestion.lm_studio import LMStudioIngester
from lemon_squeeze.ingestion.openrouter import OpenRouterIngester
from lemon_squeeze.ingestion.self_generated import SeedFileIngester
from lemon_squeeze.eval.rubric import Rubric, evaluate_runs
from lemon_squeeze.eval.runner import fanout
from lemon_squeeze.router import PRESETS, RouterWeights, recommend
from lemon_squeeze import bench as bench_mod
from lemon_squeeze.compare import compare as compare_models
from lemon_squeeze.report import build_report, report_to_html
from lemon_squeeze.doctor import run_all_checks, summarize
from lemon_squeeze.providers import (
    DiscoveredModel,
    list_lm_studio_models,
    list_openrouter_models,
)
from lemon_squeeze.portable import export_to_dir, import_from_dir

app = typer.Typer(help="Lemon Squeeze — LLM performance harness.", no_args_is_help=True)
db_app = typer.Typer(help="Database operations.", no_args_is_help=True)
ingest_app = typer.Typer(help="Ingest prompts from various sources.", no_args_is_help=True)
classify_app = typer.Typer(help="Classify prompts.", no_args_is_help=True)
models_app = typer.Typer(help="Manage registered models.", no_args_is_help=True)
eval_app = typer.Typer(help="Run executor + rubric evaluator.", no_args_is_help=True)
route_app = typer.Typer(help="Router — pick the smallest model that wins.", no_args_is_help=True)
bench_app = typer.Typer(help="Run packaged benchmarks against registered models.", no_args_is_help=True)
providers_app = typer.Typer(help="Discover available models on LM Studio + OpenRouter.", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(ingest_app, name="ingest")
app.add_typer(classify_app, name="classify")
app.add_typer(models_app, name="models")
app.add_typer(eval_app, name="eval")
app.add_typer(route_app, name="route")
app.add_typer(bench_app, name="bench")
app.add_typer(providers_app, name="providers")


@app.command("report")
def cmd_report(
    threshold: float = typer.Option(0.7, "--threshold"),
    min_samples: int = typer.Option(3, "--min-samples"),
    rubric: list[str] = typer.Option(["human_pass"], "--rubric", help="Authoritative rubrics."),
    json_out: Path = typer.Option(None, "--json", help="Write report as JSON to this path."),
    html_out: Path = typer.Option(None, "--html", help="Write report as a self-contained HTML file."),
    title: str = typer.Option("Lemon Squeeze report", "--title", help="Title for HTML output."),
) -> None:
    """One-shot executive summary: stats + per-tag scorecard + coverage gaps.

    Use `--json` for machine-readable output, `--html` for a shareable snapshot.
    If neither is set, prints to the terminal as before.
    """
    import json as _json

    rep = build_report(
        threshold=threshold,
        min_samples=min_samples,
        authoritative_rubrics=tuple(rubric),
    )

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(_json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
        console.print(f"[green]Wrote JSON[/green] to [bold]{json_out}[/bold]")

    if html_out is not None:
        html_out.parent.mkdir(parents=True, exist_ok=True)
        html_out.write_text(report_to_html(rep, title=title), encoding="utf-8")
        console.print(f"[green]Wrote HTML[/green] to [bold]{html_out}[/bold]")

    if json_out is not None or html_out is not None:
        return

    # --- Headline ---
    console.print(
        f"[bold]Prompts:[/bold] {rep.n_prompts}    "
        f"[bold]Models:[/bold] {rep.n_models}    "
        f"[bold]Runs:[/bold] {rep.n_runs}    "
        f"[bold]Evals:[/bold] {rep.n_evals}"
    )
    if rep.n_runs_with_error:
        console.print(
            f"  [yellow]runs with errors:[/yellow] {rep.n_runs_with_error}  "
            f"  total run cost: ${rep.total_cost_usd:.2f}"
        )
    else:
        console.print(f"  total run cost: ${rep.total_cost_usd:.2f}")

    left, right = ("Prompts by source", "Evals by rubric")
    if rep.prompts_by_source:
        t = Table(title=left)
        t.add_column("source"); t.add_column("count", justify="right")
        for src, cnt in rep.prompts_by_source:
            t.add_row(src, str(cnt))
        console.print(t)
    if rep.evals_by_rubric:
        t = Table(title=right)
        t.add_column("rubric"); t.add_column("count", justify="right")
        for rub, cnt in rep.evals_by_rubric:
            t.add_row(rub, str(cnt))
        console.print(t)

    # --- Per-tag scorecard ---
    if rep.scorecards:
        t = Table(title=f"Per-tag scorecard  (pass_rate ≥ {threshold:.0%}, n ≥ {min_samples})")
        t.add_column("tag")
        t.add_column("prompts", justify="right")
        t.add_column("runs", justify="right")
        t.add_column("evals", justify="right")
        t.add_column("quality pick")
        t.add_column("pass", justify="right")
        t.add_column("cost pick")
        t.add_column("$ /run", justify="right")
        t.add_column("balanced")
        for sc in rep.scorecards:
            quality = (
                f"{sc.quality_pick} ({sc.quality_n})" if sc.quality_pick else "—"
            )
            cost = sc.cost_pick or "—"
            cost_amt = f"{sc.cost_pick_avg_cost:.4f}" if sc.cost_pick_avg_cost else "—"
            balanced = sc.balanced_pick or "—"
            if not sc.has_qualifying:
                quality = f"[yellow]{quality}[/yellow]"  # below threshold
            t.add_row(
                sc.tag,
                str(sc.n_prompts),
                str(sc.n_runs),
                str(sc.n_evals),
                quality,
                f"{sc.quality_pass_rate:.0%}" if sc.quality_pass_rate is not None else "—",
                cost,
                cost_amt,
                balanced,
            )
        console.print(t)

    # --- Coverage gaps ---
    if rep.gaps:
        t = Table(title="Coverage gaps")
        t.add_column("tag"); t.add_column("prompts", justify="right"); t.add_column("reason")
        for g in rep.gaps:
            reason_blurb = {
                "no_runs": "[red]no runs[/red] — register a model and `lemon eval run`",
                "no_evals": "[yellow]runs exist but no evals[/yellow] — apply a rubric",
                "no_qualifying": (
                    f"[yellow]no model meets threshold {threshold:.0%}/min {min_samples}[/yellow]"
                ),
            }.get(g.reason, g.reason)
            t.add_row(g.tag, str(g.n_prompts), reason_blurb)
        console.print(t)

    # --- Rubric freshness ---
    if rep.rubric_freshness:
        t = Table(title="Rubric freshness")
        t.add_column("rubric")
        t.add_column("evals", justify="right")
        t.add_column("scored by")
        t.add_column("last scored")
        t.add_column("age", justify="right")
        for rf in rep.rubric_freshness:
            scored_by = ", ".join(f"{name}:{n}" for name, n in rf.scored_by_breakdown)
            last = rf.last_scored_at.strftime("%Y-%m-%d") if rf.last_scored_at else "—"
            age = "—"
            if rf.age_days is not None:
                age = f"{rf.age_days:.0f}d" if rf.age_days >= 1 else "<1d"
                if rf.stale:
                    age = f"[yellow]{age}[/yellow]"
            t.add_row(rf.rubric, str(rf.n_evals), scored_by, last, age)
        console.print(t)


# `compare` is registered on the root app as a standalone command — it's the
# main "what's the headline?" surface and deserves a short invocation.
@app.command("compare")
def cmd_compare(
    model_a: str = typer.Argument(..., help="First model name."),
    model_b: str = typer.Argument(..., help="Second model name."),
    rubric: str = typer.Option("human_pass", "--rubric", "-r"),
    min_samples: int = typer.Option(1, "--min-samples"),
    tie_threshold: float = typer.Option(0.05, "--tie-threshold"),
    no_significance: bool = typer.Option(
        False, "--no-significance",
        help="Skip the Wilson-CI significance gate (use raw deltas only). "
             "Off by default — small-sample winners are misleading.",
    ),
) -> None:
    """Head-to-head per-tag pass-rate comparison between two models."""
    try:
        report = compare_models(
            model_a,
            model_b,
            rubric=rubric,
            min_samples=min_samples,
            tie_threshold=tie_threshold,
            require_significance=not no_significance,
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)

    if not report.per_tag:
        console.print(
            f"[yellow]No overlapping (tag, model) data[/yellow] for rubric "
            f"'{rubric}' and models [bold]{model_a}[/bold], [bold]{model_b}[/bold]."
        )
        return

    sig_note = "" if no_significance else "  · 95% Wilson CI shown"
    t = Table(title=f"{model_a} (A) vs {model_b} (B) — rubric: {rubric}{sig_note}")
    t.add_column("tag")
    t.add_column("A pass", justify="right")
    t.add_column("A CI", justify="right")
    t.add_column("A n", justify="right")
    t.add_column("B pass", justify="right")
    t.add_column("B CI", justify="right")
    t.add_column("B n", justify="right")
    t.add_column("Δ", justify="right")
    t.add_column("sig")
    t.add_column("winner")
    for tc in report.per_tag:
        delta_sign = "+" if tc.delta_pass_rate > 0 else ""
        winner_color = {"A": "green", "B": "cyan", "tie": "yellow"}[tc.winner]
        sig = "[green]✓[/green]" if tc.significant else "[dim]·[/dim]"
        t.add_row(
            tc.tag,
            f"{tc.a_pass_rate:.0%}",
            f"[{tc.a_pass_ci[0]:.0%},{tc.a_pass_ci[1]:.0%}]",
            str(tc.a_n),
            f"{tc.b_pass_rate:.0%}",
            f"[{tc.b_pass_ci[0]:.0%},{tc.b_pass_ci[1]:.0%}]",
            str(tc.b_n),
            f"{delta_sign}{tc.delta_pass_rate:.0%}",
            sig,
            f"[{winner_color}]{tc.winner}[/{winner_color}]",
        )
    console.print(t)
    overall_color = {"A": "green", "B": "cyan", "tie": "yellow"}[report.overall_winner]
    console.print(
        f"[bold]Overall:[/bold] A wins {report.a_wins}, B wins {report.b_wins}, "
        f"ties {report.ties} → [{overall_color}]{report.overall_winner}[/{overall_color}]"
    )


# --- bench ---

console = Console()


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"lemon-squeeze {lemon_squeeze.__version__}")


@app.command()
def demo() -> None:
    """Run a zero-config offline demo of the full pipeline.

    Walks through: fresh DB → seed prompts → classify → register two fake
    models → mocked fan-out → score → compare → router recommendations →
    executive report. Demonstrates that `cheap/small-3b` is the right pick
    for math (cheaper, still correct) while `premium/big-70b` wins on coding.
    No API keys or LM Studio required.

    Use this as your first taste of what the project does.
    """
    from lemon_squeeze.demo import run_demo

    run_demo()


@app.command()
def judge(
    rubric_path: Path = typer.Argument(..., exists=True, help="YAML/JSON rubric file."),
    prompt: str = typer.Option(None, "--prompt", "-p", help="The prompt text."),
    response: str = typer.Option(None, "--response", "-r", help="The response to score."),
    response_file: Path = typer.Option(
        None, "--response-file", help="Read response text from a file instead of --response."
    ),
    metadata_json: str = typer.Option(
        None, "--metadata", help="Optional JSON dict of prompt metadata (for per-prompt judges)."
    ),
) -> None:
    """Score a single (prompt, response) pair against a rubric without touching the DB.

    Useful for ad-hoc testing: "is this response acceptable under my rubric?"
    Without registering a model, running it, or persisting evaluations. Returns
    the verdict as terminal output.

    Pass `--metadata '{"expected_contains": ["foo", "bar"]}'` to test per-prompt
    rubrics like `rubrics/per_prompt_expected.yaml`.
    """
    import json as _json

    if response is None and response_file is None:
        console.print("[red]Error:[/red] either --response or --response-file required.")
        raise typer.Exit(code=2)
    if response is None:
        response = response_file.read_text(encoding="utf-8")
    if prompt is None:
        prompt = ""  # some judges don't need the prompt text

    metadata: dict | None = None
    if metadata_json:
        try:
            metadata = _json.loads(metadata_json)
        except _json.JSONDecodeError as e:
            console.print(f"[red]Invalid --metadata JSON:[/red] {e}")
            raise typer.Exit(code=2)

    from lemon_squeeze.eval.rubric import Rubric
    from lemon_squeeze.eval.judges import build_judge

    rubric = Rubric.from_file(rubric_path)
    judge_inst = build_judge(rubric.judge_kind, rubric.judge_config)
    verdict = judge_inst.evaluate(prompt, response, metadata=metadata)

    badge_color, badge = (
        ("green", "PASS") if verdict.passed is True
        else ("red", "FAIL") if verdict.passed is False
        else ("yellow", "SKIPPED")
    )
    console.print(
        f"[bold]Rubric:[/bold] {rubric.name}  judge={rubric.judge_kind}\n"
        f"[{badge_color}]{badge}[/{badge_color}]  score={verdict.score:.2f}"
    )
    if verdict.notes:
        console.print(f"  notes: {verdict.notes}")
    if verdict.judge_model:
        console.print(f"  judge_model: {verdict.judge_model}")


@app.command()
def doctor() -> None:
    """Diagnose setup: walks every prerequisite, reports OK/WARN/FAIL with hints."""
    results = run_all_checks()
    ok, warn, fail = summarize(results)

    glyphs = {
        "ok": "[green]OK  [/green]",
        "warn": "[yellow]WARN[/yellow]",
        "fail": "[red]FAIL[/red]",
    }
    for r in results:
        console.print(f"  {glyphs[r.status]}  [bold]{r.name:18s}[/bold] {r.detail}")
        if r.hint and r.status != "ok":
            console.print(f"        [dim]→ {r.hint}[/dim]")
    console.print(
        f"\n[bold]Summary:[/bold] [green]{ok} ok[/green]  "
        f"[yellow]{warn} warn[/yellow]  [red]{fail} fail[/red]"
    )
    if fail:
        raise typer.Exit(code=1)


@app.command()
def serve(
    port: int = typer.Option(8080, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host"),
    workers: int = typer.Option(1, "--workers"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on file changes (dev only)."),
) -> None:
    """Launch the HTTP API server. Requires `pip install '.[server]'`.

    Exposes /healthz /models /route /classify /report /compare. The router
    you'd otherwise call via Python now lives behind POST /route.
    """
    try:
        import uvicorn  # noqa: F401
        from lemon_squeeze.server import app as _server_app  # noqa: F401
    except ImportError:
        console.print(
            "[red]Server extras not installed.[/red] Install with: "
            "[bold]pip install -e '.[server]'[/bold]"
        )
        raise typer.Exit(code=1)

    console.print(f"[bold]Serving:[/bold] http://{host}:{port} (workers={workers})")
    uvicorn.run(
        "lemon_squeeze.server:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info",
    )


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p"),
    host: str = typer.Option("localhost", "--host"),
) -> None:
    """Launch the Streamlit dashboard. Requires `pip install '.[dashboard]'`."""
    import subprocess
    import sys

    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print(
            "[red]Streamlit not installed.[/red] Install with: "
            "[bold]pip install -e '.[dashboard]'[/bold]"
        )
        raise typer.Exit(code=1)

    dashboard_path = Path(__file__).parent / "dashboard.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
    ]
    console.print(f"[bold]Launching dashboard:[/bold] http://{host}:{port}")
    subprocess.run(cmd, check=False)


# --- db ---


@db_app.command("init")
def db_init(
    no_stamp: bool = typer.Option(
        False,
        "--no-stamp",
        help="Skip the Alembic stamp step. Use only if you want to manage "
        "migrations manually.",
    ),
) -> None:
    """Create tables, seed the tag taxonomy, and stamp at current Alembic head.

    The stamp registers the schema as current in Alembic's bookkeeping so future
    `lemon db upgrade` calls apply incremental changes correctly. Without it,
    Alembic thinks the DB is at the *zero* revision and would try to re-run the
    initial migration on the next upgrade — which would fail because the tables
    already exist.
    """
    init_db()
    if no_stamp:
        console.print("[green]OK[/green] Database initialized [dim](no stamp)[/dim].")
        return

    # Best-effort: stamp the DB at head so migrations are tracked. Graceful
    # fallback when alembic.ini is missing (e.g. wheel installs that excluded
    # the project root) — init still succeeded, the user just has to stamp
    # manually if they want migrations.
    try:
        from alembic import command

        cfg = _alembic_config()
        command.stamp(cfg, "head")
        console.print("[green]OK[/green] Database initialized and stamped at head.")
    except (ImportError, FileNotFoundError) as e:
        console.print(
            "[green]OK[/green] Database initialized "
            f"[dim](stamp skipped: {e})[/dim]."
        )


def _alembic_config():
    """Build an Alembic Config that points at our project layout.

    Imported lazily because alembic config setup can fail when its files
    aren't present (e.g. running from a wheel install that excluded them).
    """
    from pathlib import Path

    from alembic.config import Config

    project_root = Path(__file__).resolve().parents[2]
    ini = project_root / "alembic.ini"
    if not ini.exists():
        raise FileNotFoundError(
            f"alembic.ini not found at {ini}. "
            "Alembic migrations require the source tree, not a wheel install."
        )
    cfg = Config(str(ini))
    cfg.set_main_option("script_location", str(project_root / "alembic"))
    return cfg


@db_app.command("upgrade")
def db_upgrade(
    revision: str = typer.Argument("head", help="Target revision (default: head)."),
) -> None:
    """Run Alembic migrations up to a target revision.

    For new installs, prefer `lemon db init` — it uses Base.metadata.create_all
    which is faster than running migrations from scratch. Use `db upgrade` to
    apply incremental changes to an existing database.
    """
    try:
        from alembic import command

        cfg = _alembic_config()
    except (ImportError, FileNotFoundError) as e:
        console.print(f"[red]Cannot run migrations:[/red] {e}")
        raise typer.Exit(code=1)
    command.upgrade(cfg, revision)
    console.print(f"[green]OK[/green] Database upgraded to {revision}")


@db_app.command("downgrade")
def db_downgrade(
    revision: str = typer.Argument(..., help="Target revision (e.g. -1 to roll back one step)."),
) -> None:
    """Roll back Alembic migrations to a target revision."""
    try:
        from alembic import command

        cfg = _alembic_config()
    except (ImportError, FileNotFoundError) as e:
        console.print(f"[red]Cannot run migrations:[/red] {e}")
        raise typer.Exit(code=1)
    command.downgrade(cfg, revision)
    console.print(f"[green]OK[/green] Database downgraded to {revision}")


@db_app.command("current")
def db_current() -> None:
    """Show the current migration revision applied to the DB."""
    try:
        from alembic import command

        cfg = _alembic_config()
    except (ImportError, FileNotFoundError) as e:
        console.print(f"[red]Cannot read migrations:[/red] {e}")
        raise typer.Exit(code=1)
    command.current(cfg, verbose=True)


@db_app.command("stamp")
def db_stamp(
    revision: str = typer.Argument("head", help="Revision to stamp (default: head)."),
) -> None:
    """Mark the DB as being at a revision WITHOUT running migrations.

    Useful when you used `lemon db init` (which calls create_all) and want to
    register the current schema as the latest revision so future migrations
    apply incrementally.
    """
    try:
        from alembic import command

        cfg = _alembic_config()
    except (ImportError, FileNotFoundError) as e:
        console.print(f"[red]Cannot stamp:[/red] {e}")
        raise typer.Exit(code=1)
    command.stamp(cfg, revision)
    console.print(f"[green]OK[/green] Database stamped at {revision}")


@db_app.command("stats")
def db_stats() -> None:
    """Show counts and breakdown by source/tag."""
    with get_session() as s:
        prompt_count = s.scalar(select(func.count()).select_from(Prompt)) or 0
        model_count = s.scalar(select(func.count()).select_from(Model)) or 0
        run_count = s.scalar(select(func.count()).select_from(Run)) or 0

        by_source = s.execute(
            select(Prompt.source, func.count()).group_by(Prompt.source)
        ).all()
        by_tag = s.execute(
            select(PromptTag.tag, PromptTag.classifier, func.count())
            .group_by(PromptTag.tag, PromptTag.classifier)
            .order_by(func.count().desc())
        ).all()

    console.print(f"[bold]Prompts:[/bold] {prompt_count}    [bold]Models:[/bold] {model_count}    [bold]Runs:[/bold] {run_count}")

    if by_source:
        t = Table(title="Prompts by source")
        t.add_column("source")
        t.add_column("count", justify="right")
        for src, c in by_source:
            t.add_row(src, str(c))
        console.print(t)

    if by_tag:
        t = Table(title="Tags by classifier")
        t.add_column("tag")
        t.add_column("classifier")
        t.add_column("count", justify="right")
        for tag, cls, c in by_tag:
            t.add_row(tag, cls, str(c))
        console.print(t)


# --- ingest ---


def _report(label: str, result, *, dry_run: bool = False) -> None:
    prefix = "[dim](dry-run)[/dim] " if dry_run else ""
    would = "would-insert" if dry_run else "inserted"
    console.print(
        f"{prefix}[bold]{label}[/bold] — {would}: [green]{result.inserted}[/green], "
        f"duplicates: [yellow]{result.duplicates}[/yellow], "
        f"skipped: {result.skipped}, errors: {len(result.errors)}"
    )
    for err in result.errors[:5]:
        console.print(f"  [red]![/red] {err}")


DRY_RUN_OPT = typer.Option(False, "--dry-run", help="Preview without writing to the DB.")
NO_CLASSIFY_OPT = typer.Option(
    False,
    "--no-classify",
    help="Skip the heuristic classifier pass on newly-ingested prompts.",
)


def _auto_classify(skip: bool) -> None:
    """After an ingest, heuristic-tag any prompts that don't yet have a heuristic
    tag. Idempotent: prompts already tagged are skipped via `only_missing_classifier`.
    Called from every ingest CLI command unless `--no-classify` was passed.
    """
    if skip:
        return
    from lemon_squeeze.classification import HeuristicClassifier
    from lemon_squeeze.classification.ensemble import classify_unlabeled

    stats = classify_unlabeled(
        HeuristicClassifier(), only_missing_classifier="heuristic"
    )
    if stats.tags_written > 0:
        console.print(
            f"  [dim]auto-classify:[/dim] tagged {stats.tags_written} new prompts"
        )


@ingest_app.command("lm-studio")
def ingest_lm_studio(
    logs_dir: Path = typer.Option(
        None, "--logs-dir", "-d", help="Override LM_STUDIO_LOGS_DIR."
    ),
    dry_run: bool = DRY_RUN_OPT,
    no_classify: bool = NO_CLASSIFY_OPT,
) -> None:
    """Walk LM Studio's local conversation logs."""
    res = LMStudioIngester(logs_dir=logs_dir).run(dry_run=dry_run)
    _report("LM Studio", res, dry_run=dry_run)
    if not dry_run:
        _auto_classify(no_classify)


@ingest_app.command("claude")
def ingest_claude(
    export_path: Path = typer.Argument(..., exists=True, help="Path to Claude conversations.json"),
    dry_run: bool = DRY_RUN_OPT,
    no_classify: bool = NO_CLASSIFY_OPT,
) -> None:
    """Ingest a Claude.ai data export."""
    res = ClaudeExportIngester(export_path).run(dry_run=dry_run)
    _report("Claude export", res, dry_run=dry_run)
    if not dry_run:
        _auto_classify(no_classify)


@ingest_app.command("grok")
def ingest_grok(
    path: Path = typer.Argument(..., exists=True, help="Grok export file or directory"),
    dry_run: bool = DRY_RUN_OPT,
    no_classify: bool = NO_CLASSIFY_OPT,
) -> None:
    """Ingest a Grok export (file or directory of JSON)."""
    res = GrokExportIngester(path).run(dry_run=dry_run)
    _report("Grok export", res, dry_run=dry_run)
    if not dry_run:
        _auto_classify(no_classify)


@ingest_app.command("openrouter")
def ingest_openrouter(
    history_file: Path = typer.Option(
        None, "--file", "-f", help="Pre-downloaded JSON history (offline mode)."
    ),
    since: str = typer.Option(
        None, "--since", help="Filter to records newer than this (e.g. 7d, 24h)."
    ),
    dry_run: bool = DRY_RUN_OPT,
    no_classify: bool = NO_CLASSIFY_OPT,
) -> None:
    """Pull from OpenRouter generation history."""
    res = OpenRouterIngester(
        history_file=history_file,
        since=_parse_duration(since) if since else None,
    ).run(dry_run=dry_run)
    _report("OpenRouter", res, dry_run=dry_run)
    if not dry_run:
        _auto_classify(no_classify)


@ingest_app.command("seed")
def ingest_seed(
    path: Path = typer.Argument(..., exists=True, help="JSON or JSONL seed prompt file."),
    dry_run: bool = DRY_RUN_OPT,
    no_classify: bool = NO_CLASSIFY_OPT,
) -> None:
    """Ingest hand-authored test prompts."""
    res = SeedFileIngester(path).run(dry_run=dry_run)
    _report("Seed file", res, dry_run=dry_run)
    if not dry_run:
        _auto_classify(no_classify)


@ingest_app.command("ai-harness")
def ingest_ai_harness(
    db: Path = typer.Argument(
        Path(r"C:\dev\ttx\AI Harness\data\harness_logs.db"),
        exists=True,
        help="Path to harness_logs.db from the sibling AI Harness project.",
    ),
    dry_run: bool = DRY_RUN_OPT,
    no_classify: bool = NO_CLASSIFY_OPT,
) -> None:
    """Import prompts + models + runs + evaluations from an AI Harness SQLite DB."""
    res = AIHarnessImporter(db).run(dry_run=dry_run)
    prefix = "[dim](dry-run)[/dim] " if dry_run else ""
    would_label = "would-import" if dry_run else "imported"
    console.print(
        f"{prefix}[bold]AI Harness[/bold] — runs seen: {res.runs_seen}, "
        f"{would_label}: [green]{res.runs_imported}[/green], "
        f"skipped existing: {res.runs_skipped_existing}\n"
        f"  prompts inserted: [green]{res.prompts_inserted}[/green], deduped: {res.prompts_deduped}\n"
        f"  models registered: [green]{res.models_registered}[/green]\n"
        f"  evaluations inserted: [green]{res.evaluations_inserted}[/green]"
    )
    for err in res.errors[:5]:
        console.print(f"  [red]![/red] {err}")
    if not dry_run:
        _auto_classify(no_classify)


# --- classify ---


@classify_app.command("run")
def classify_run(
    limit: int = typer.Option(None, "--limit", "-n"),
    only_missing: str = typer.Option(
        None,
        "--only-missing",
        help="Only classify prompts missing a tag from this classifier "
        "(e.g. 'ml' to skip ones already covered).",
    ),
) -> None:
    """Run the ensemble classifier over prompts in the DB."""
    classifier = build_default_classifier()
    members = ", ".join(m.name for m in classifier.members)
    console.print(f"[bold]Ensemble members:[/bold] {members}")
    stats = classify_unlabeled(classifier, limit=limit, only_missing_classifier=only_missing)
    console.print(
        f"seen: {stats.prompts_seen}, classified: {stats.prompts_classified}, "
        f"tags written: [green]{stats.tags_written}[/green], "
        f"skipped existing: {stats.tags_skipped_existing}"
    )


@classify_app.command("ask")
def classify_ask(
    prompt: str = typer.Argument(..., help="The prompt to classify."),
    classifier: str = typer.Option(
        "ensemble", "--classifier", "-c",
        help="Which classifier: 'heuristic', 'ml', 'ensemble' (default).",
    ),
    top: int = typer.Option(
        0, "--top", help="Show only the top N predictions by confidence (0 = show all)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """One-shot classify a single prompt without touching the DB.

    Useful for quick "what tag is this?" queries. Mirrors `lemon judge` for
    classification: library exposes the same call, this surfaces it on the CLI.
    """
    import json as _json

    from lemon_squeeze.classification import (
        HeuristicClassifier,
        MLClassifier,
        build_default_classifier,
    )

    if classifier == "heuristic":
        clf = HeuristicClassifier()
    elif classifier == "ml":
        loaded = MLClassifier.load()
        if loaded is None:
            console.print(
                "[yellow]No trained ML classifier found.[/yellow] "
                "Run [bold]lemon classify train-ml[/bold] first."
            )
            raise typer.Exit(code=1)
        clf = loaded
    elif classifier == "ensemble":
        clf = build_default_classifier()
    else:
        console.print(
            f"[red]Unknown classifier:[/red] {classifier!r} "
            f"(choices: heuristic, ml, ensemble)"
        )
        raise typer.Exit(code=2)

    preds = clf.predict(prompt)
    preds_sorted = sorted(preds, key=lambda p: -p.confidence)
    if top > 0:
        preds_sorted = preds_sorted[:top]

    if json_output:
        # Bypass rich.Console — its line-wrapping mangles JSON output.
        import sys
        sys.stdout.write(_json.dumps(
            {
                "prompt": prompt,
                "classifier": classifier,
                "predictions": [
                    {"tag": p.tag, "confidence": p.confidence, "classifier": p.classifier}
                    for p in preds_sorted
                ],
            },
            indent=2,
        ))
        sys.stdout.write("\n")
        return

    if not preds_sorted:
        console.print(f"[yellow]No predictions[/yellow] for prompt: {prompt!r}")
        return

    t = Table(title=f"Classifier: {classifier}")
    t.add_column("tag")
    t.add_column("confidence", justify="right")
    t.add_column("from")
    for p in preds_sorted:
        conf_color = "green" if p.confidence >= 0.7 else "yellow" if p.confidence >= 0.4 else "dim"
        t.add_row(
            p.tag,
            f"[{conf_color}]{p.confidence:.2f}[/{conf_color}]",
            p.classifier,
        )
    console.print(t)


@classify_app.command("train-ml")
def classify_train_ml(
    no_backfill: bool = typer.Option(
        False,
        "--no-backfill",
        help="Skip applying the freshly-trained classifier to existing prompts.",
    ),
) -> None:
    """Train the ML classifier from labels currently in the DB, then back-fill
    ML tags on prompts that don't have them yet.

    Without the back-fill step, the freshly-trained model sits idle until the
    next `lemon classify run` reaches each prompt. We do it automatically so
    the model's predictions are visible immediately. Pass `--no-backfill` to
    inspect the trained model before applying it.
    """
    clf = MLClassifier()
    counts = clf.train_from_db()
    clf.save()
    console.print("[green]OK[/green] Trained and saved ML classifier.")
    console.print("[bold]Label balance:[/bold]")
    console.print(MLClassifier.report_balance(counts))

    if no_backfill:
        return
    from lemon_squeeze.classification.ensemble import classify_unlabeled

    stats = classify_unlabeled(clf, only_missing_classifier="ml")
    console.print(
        f"  [dim]auto-backfill:[/dim] applied ML to {stats.prompts_classified} prompts, "
        f"wrote {stats.tags_written} tags"
    )


# --- models ---


@models_app.command("register")
def models_register(
    name: str = typer.Argument(..., help="Provider-qualified name, e.g. anthropic/claude-sonnet-4-6 or lm_studio/llama-3.1-8b"),
    provider: str = typer.Option(None, "--provider", help="Override the provider (default: parsed from name)."),
    family: str = typer.Option(None, "--family"),
    size_b: float = typer.Option(None, "--size-b", help="Parameter count in billions."),
    context_window: int = typer.Option(None, "--ctx"),
    local: bool = typer.Option(False, "--local", help="Set if this is a local LM Studio model."),
    cost_in: float = typer.Option(None, "--cost-in", help="USD per million input tokens."),
    cost_out: float = typer.Option(None, "--cost-out", help="USD per million output tokens."),
) -> None:
    """Register a model so it can be used for runs."""
    from lemon_squeeze.utils import split_provider_family

    parsed_provider, parsed_family = split_provider_family(name)
    if provider is None:
        # Special-case: a bare name with --local explicitly set should default
        # to lm_studio rather than 'unknown'.
        provider = parsed_provider if "/" in name else ("lm_studio" if local else "unknown")
    if family is None:
        family = parsed_family

    with get_session() as s:
        existing = s.scalar(select(Model).where(Model.name == name))
        if existing is not None:
            for attr, val in (
                ("provider", provider),
                ("family", family),
                ("size_params_b", size_b),
                ("context_window", context_window),
                ("local", local),
                ("cost_in_per_mtok", cost_in),
                ("cost_out_per_mtok", cost_out),
            ):
                if val is not None:
                    setattr(existing, attr, val)
            console.print(f"[green]OK[/green] Updated model [bold]{name}[/bold]")
            return
        s.add(
            Model(
                name=name,
                provider=provider,
                family=family,
                size_params_b=size_b,
                context_window=context_window,
                local=local,
                cost_in_per_mtok=cost_in,
                cost_out_per_mtok=cost_out,
            )
        )
        console.print(f"[green]OK[/green] Registered [bold]{name}[/bold]")


@models_app.command("list")
def models_list() -> None:
    """List registered models."""
    with get_session() as s:
        models = list(s.scalars(select(Model).order_by(Model.size_params_b.is_(None), Model.size_params_b)).all())
    if not models:
        console.print("[yellow]No models registered.[/yellow] Use `lemon models register`.")
        return
    t = Table(title="Registered models")
    t.add_column("name"); t.add_column("provider"); t.add_column("size B", justify="right")
    t.add_column("ctx", justify="right"); t.add_column("local"); t.add_column("$ in/out per Mtok")
    for m in models:
        t.add_row(
            m.name,
            m.provider,
            f"{m.size_params_b:.1f}" if m.size_params_b else "?",
            str(m.context_window) if m.context_window else "?",
            "yes" if m.local else "no",
            f"{m.cost_in_per_mtok or 0:.2f}/{m.cost_out_per_mtok or 0:.2f}",
        )
    console.print(t)


# --- eval ---


@eval_app.command("run")
def eval_run(
    models: list[str] = typer.Option(None, "--model", "-m", help="Restrict to these model names."),
    prompts: list[int] = typer.Option(None, "--prompt", "-p", help="Restrict to these prompt IDs."),
    temperature: float = typer.Option(0.0, "--temp"),
    max_tokens: int = typer.Option(None, "--max-tokens"),
    force: bool = typer.Option(False, "--force", help="Re-run even if a Run already exists."),
) -> None:
    """Execute prompts against models and save Run rows."""
    report = fanout(
        prompt_ids=prompts or None,
        model_names=models or None,
        temperature=temperature,
        max_tokens=max_tokens,
        skip_existing=not force,
    )
    console.print(
        f"attempted: {report.attempted}, succeeded: [green]{report.succeeded}[/green], "
        f"failed: [red]{report.failed}[/red]"
    )
    for err in report.errors[:5]:
        console.print(f"  [red]![/red] {err}")


@eval_app.command("score")
def eval_score(
    rubric_path: Path = typer.Argument(..., exists=True, help="YAML/JSON rubric file."),
    force: bool = typer.Option(False, "--force", help="Append a new eval even if a row exists."),
) -> None:
    """Apply a rubric to existing runs and write Evaluation rows."""
    rubric = Rubric.from_file(rubric_path)
    console.print(
        f"[bold]Rubric:[/bold] {rubric.name}  judge={rubric.judge_kind}  "
        f"applies_to={rubric.applies_to_tags or 'all'}"
    )
    report = evaluate_runs(rubric, skip_existing=not force)
    stale_msg = (
        f", [yellow]stale re-scored: {report.stale_replaced}[/yellow]"
        if report.stale_replaced else ""
    )
    console.print(
        f"runs seen: {report.runs_seen}, evaluated: [green]{report.runs_evaluated}[/green]"
        f"{stale_msg}, skipped existing: {report.skipped_existing}, "
        f"skipped no-response: {report.skipped_no_response}, "
        f"skipped tag-mismatch: {report.skipped_tag_mismatch}"
    )
    for err in report.errors[:5]:
        console.print(f"  [red]![/red] {err}")


@eval_app.command("replay")
def eval_replay(
    rubric_path: Path = typer.Argument(..., exists=True, help="YAML/JSON rubric file."),
) -> None:
    """Delete every existing Evaluation for this rubric, then re-score everything.

    Use this when you've changed the rubric definition (different judge config,
    new threshold, etc.) and want a clean re-scoring of the historical runs.
    Unlike `--force` on `score`, this *replaces* old rows rather than appending.
    """
    rubric = Rubric.from_file(rubric_path)
    console.print(
        f"[bold]Replaying rubric:[/bold] {rubric.name}  judge={rubric.judge_kind}"
    )
    report = evaluate_runs(rubric, replace_existing=True)
    console.print(
        f"old evals deleted: [yellow]{report.replaced}[/yellow], "
        f"new evals written: [green]{report.evaluations_written}[/green], "
        f"runs seen: {report.runs_seen}, "
        f"skipped no-response: {report.skipped_no_response}, "
        f"skipped tag-mismatch: {report.skipped_tag_mismatch}"
    )
    for err in report.errors[:5]:
        console.print(f"  [red]![/red] {err}")


# --- route ---


@route_app.command("pick")
def route_pick(
    prompt: str = typer.Argument(..., help="The prompt to route."),
    threshold: float = typer.Option(0.7, "--threshold"),
    min_samples: int = typer.Option(3, "--min-samples"),
    rubric: list[str] = typer.Option(["human_pass"], "--rubric", help="Authoritative rubrics."),
    preset: str = typer.Option(
        "size", "--preset", help=f"Scoring preset: {', '.join(sorted(PRESETS))}."
    ),
    weight_size: float = typer.Option(None, "--w-size", help="Override size weight."),
    weight_cost: float = typer.Option(None, "--w-cost", help="Override cost weight."),
    weight_latency: float = typer.Option(None, "--w-latency", help="Override latency weight."),
) -> None:
    """Recommend the best model under your weights. Defaults to 'size' (smallest qualifying)."""
    try:
        weights = RouterWeights.from_preset_and_overrides(
            preset, size=weight_size, cost=weight_cost, latency=weight_latency,
        )
    except ValueError as e:
        console.print(f"[red]Unknown preset[/red] {preset!r}; known: {sorted(PRESETS)}")
        raise typer.Exit(code=2) from e
    rec = recommend(
        prompt,
        threshold=threshold,
        min_samples=min_samples,
        authoritative_rubrics=tuple(rubric),
        weights=weights,
    )
    w = rec.weights
    console.print(
        f"[bold]Tags:[/bold] {rec.tags or '(none)'}    "
        f"[bold]Weights:[/bold] size={w.size:.2f} cost={w.cost:.2f} latency={w.latency:.2f}"
    )
    if rec.picked is None:
        console.print(f"[yellow]No recommendation:[/yellow] {rec.reason}")
        return
    badge = "[yellow]FALLBACK[/yellow]" if rec.fallback else "[green]PICK[/green]"
    p = rec.picked
    score_str = f"  score={p.composite_score:.2f}" if p.composite_score is not None else ""
    console.print(
        f"{badge} [bold]{p.model_name}[/bold]  pass_rate={p.pass_rate:.0%} "
        f"over {p.sample_count} runs  avg_score={p.avg_score:.2f}{score_str}"
    )
    console.print(f"  reason: {rec.reason}")
    if rec.candidates:
        t = Table(title="All candidates")
        t.add_column("model"); t.add_column("size B", justify="right"); t.add_column("n", justify="right")
        t.add_column("pass_rate", justify="right"); t.add_column("avg_score", justify="right")
        t.add_column("$ avg", justify="right"); t.add_column("ms avg", justify="right")
        t.add_column("composite", justify="right")
        for c in sorted(rec.candidates, key=lambda s: -(s.composite_score or 0)):
            t.add_row(
                c.model_name,
                f"{c.size_params_b:.1f}" if c.size_params_b else "?",
                str(c.sample_count),
                f"{c.pass_rate:.0%}",
                f"{c.avg_score:.2f}",
                f"{c.avg_cost_usd:.4f}" if c.avg_cost_usd else "—",
                f"{c.avg_latency_ms:.0f}" if c.avg_latency_ms else "—",
                f"{c.composite_score:.2f}" if c.composite_score is not None else "—",
            )
        console.print(t)


# --- bench ---


@bench_app.command("load")
def bench_load(
    bench_dir: Path = typer.Argument(
        Path("benchmarks/starter"), exists=True, help="Path to a bench directory."
    ),
    no_classify: bool = NO_CLASSIFY_OPT,
) -> None:
    """Ingest prompts from a bench directory's prompts/*.jsonl files."""
    inserted, deduped = bench_mod.load(bench_dir)
    console.print(
        f"[bold]{bench_dir.name}[/bold] — prompts inserted: [green]{inserted}[/green], "
        f"duplicates: [yellow]{deduped}[/yellow]"
    )
    _auto_classify(no_classify)


@bench_app.command("run")
def bench_run(
    bench_dir: Path = typer.Argument(
        Path("benchmarks/starter"), exists=True, help="Path to a bench directory."
    ),
    models: list[str] = typer.Option(None, "--model", "-m", help="Restrict to these models."),
    workers: int = typer.Option(4, "--workers", "-j"),
    temperature: float = typer.Option(0.0, "--temp"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Load → fanout → score → report. Defaults to all registered models."""
    report = bench_mod.run(
        bench_dir,
        model_names=models or None,
        max_workers=workers,
        skip_existing=not force,
        temperature=temperature,
    )
    console.print(
        f"[bold]bench[/bold] {report.bench_name} — "
        f"prompts loaded: [green]{report.prompts_loaded}[/green] "
        f"(dup {report.prompts_deduped})  "
        f"runs: {report.runs_succeeded} ok / {report.runs_failed} fail / "
        f"{report.runs_attempted} attempted\n"
        f"  expected_contains evals written: {report.expected_evals_written}, "
        f"rubric evals: {report.rubric_evals_written}"
    )
    if report.per_category:
        t = Table(title="Per-category results")
        t.add_column("category")
        t.add_column("model")
        t.add_column("n", justify="right")
        t.add_column("pass", justify="right")
        t.add_column("pass_rate", justify="right")
        t.add_column("avg_score", justify="right")
        t.add_column("$ /run", justify="right")
        t.add_column("$ /pass", justify="right")
        t.add_column("ms", justify="right")
        for s in report.per_category:
            t.add_row(
                s.category,
                s.model_name,
                str(s.n_runs),
                str(s.pass_count),
                f"{s.pass_rate:.0%}",
                f"{s.avg_score:.2f}",
                f"{s.avg_cost_usd:.4f}" if s.avg_cost_usd is not None else "—",
                f"{s.cost_per_pass:.4f}" if s.cost_per_pass is not None else "—",
                f"{s.avg_latency_ms:.0f}" if s.avg_latency_ms is not None else "—",
            )
        console.print(t)
    for err in report.errors[:5]:
        console.print(f"  [red]![/red] {err}")


# --- export / import ---


@app.command("export")
def cmd_export(
    out_dir: Path = typer.Argument(..., help="Directory to write JSONL files into."),
    no_runs: bool = typer.Option(False, "--no-runs", help="Skip run rows (and their evals)."),
    no_evals: bool = typer.Option(False, "--no-evals", help="Skip evaluation rows."),
    no_taxonomy: bool = typer.Option(False, "--no-taxonomy"),
) -> None:
    """Export the DB to JSONL files for backup, sharing, or migration."""
    report = export_to_dir(
        out_dir,
        include_runs=not no_runs,
        include_evaluations=not no_evals,
        include_taxonomy=not no_taxonomy,
    )
    console.print(
        f"[green]Exported[/green] to [bold]{out_dir}[/bold]:\n"
        f"  prompts: {report.prompts}  models: {report.models}  "
        f"prompt_tags: {report.prompt_tags}\n"
        f"  runs: {report.runs}  evaluations: {report.evaluations}  "
        f"taxonomy: {report.tag_taxonomy}\n"
        f"  files: {len(report.files)}"
    )


@app.command("import")
def cmd_import(
    in_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Directory of JSONL files to import."),
) -> None:
    """Import a previously-exported directory back into the DB. Idempotent."""
    report = import_from_dir(in_dir)
    console.print(
        f"[green]Imported[/green] from [bold]{in_dir}[/bold]:\n"
        f"  prompts: [green]+{report.prompts_inserted}[/green] "
        f"(deduped {report.prompts_deduped})\n"
        f"  models:  [green]+{report.models_inserted}[/green] "
        f"(updated {report.models_updated})\n"
        f"  prompt_tags: [green]+{report.prompt_tags_inserted}[/green] "
        f"(deduped {report.prompt_tags_deduped})\n"
        f"  runs:    [green]+{report.runs_inserted}[/green] "
        f"(deduped {report.runs_deduped})\n"
        f"  evals:   [green]+{report.evaluations_inserted}[/green] "
        f"(deduped {report.evaluations_deduped})\n"
        f"  taxonomy: [green]+{report.taxonomy_inserted}[/green]"
    )
    if report.skipped:
        console.print(f"\n[yellow]Skipped {len(report.skipped)} rows:[/yellow]")
        for msg in report.skipped[:5]:
            console.print(f"  [yellow]·[/yellow] {msg}")
        if len(report.skipped) > 5:
            console.print(f"  … {len(report.skipped) - 5} more")


# --- providers ---


@providers_app.command("list")
def providers_list(
    lm_studio: bool = typer.Option(True, "--lm-studio/--no-lm-studio"),
    openrouter: bool = typer.Option(True, "--openrouter/--no-openrouter"),
    limit: int = typer.Option(50, "--limit", "-n", help="Cap rows per provider (OpenRouter has 200+)."),
) -> None:
    """Ping enabled providers and list the models they expose right now."""
    sections: list[tuple[str, list[DiscoveredModel] | str]] = []

    if lm_studio:
        try:
            models = list_lm_studio_models()
            sections.append(("LM Studio", models))
        except Exception as e:
            sections.append(("LM Studio", f"[red]unreachable: {e}[/red]"))

    if openrouter:
        try:
            models = list_openrouter_models()
            sections.append(("OpenRouter", models))
        except Exception as e:
            sections.append(("OpenRouter", f"[red]unreachable: {e}[/red]"))

    for label, result in sections:
        if isinstance(result, str):
            console.print(f"[bold]{label}[/bold] — {result}")
            continue
        if not result:
            console.print(f"[bold]{label}[/bold] — [yellow]no models reported[/yellow]")
            continue
        t = Table(title=f"{label} ({len(result)} models)")
        t.add_column("name")
        t.add_column("family")
        t.add_column("ctx", justify="right")
        t.add_column("$ in /Mtok", justify="right")
        t.add_column("$ out /Mtok", justify="right")
        for m in result[:limit]:
            t.add_row(
                m.name,
                m.family or "—",
                str(m.context_window) if m.context_window else "—",
                f"{m.cost_in_per_mtok:.2f}" if m.cost_in_per_mtok is not None else "—",
                f"{m.cost_out_per_mtok:.2f}" if m.cost_out_per_mtok is not None else "—",
            )
        if len(result) > limit:
            t.caption = f"… {len(result) - limit} more (use --limit to show)"
        console.print(t)


@providers_app.command("sync")
def providers_sync(
    lm_studio: bool = typer.Option(True, "--lm-studio/--no-lm-studio"),
    openrouter: bool = typer.Option(False, "--openrouter/--no-openrouter",
        help="OpenRouter has 200+ models — disabled by default; pass --openrouter to import."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Auto-register provider models into the DB.

    Local LM Studio models get `local=True`. OpenRouter models get pricing,
    context window, and family/provider parsed from the model id.
    """
    discovered: list[DiscoveredModel] = []
    errors: list[str] = []

    if lm_studio:
        try:
            discovered.extend(list_lm_studio_models())
        except Exception as e:
            errors.append(f"LM Studio: {e}")
    if openrouter:
        try:
            discovered.extend(list_openrouter_models())
        except Exception as e:
            errors.append(f"OpenRouter: {e}")

    for err in errors:
        console.print(f"  [red]![/red] {err}")
    if not discovered:
        console.print("[yellow]No models discovered.[/yellow]")
        return

    if dry_run:
        for m in discovered:
            console.print(
                f"  would register: [bold]{m.name}[/bold]  "
                f"provider={m.provider}  family={m.family}  ctx={m.context_window}"
            )
        console.print(f"[bold]{len(discovered)} models[/bold] (dry-run; nothing written)")
        return

    added = 0
    updated = 0
    with get_session() as s:
        for m in discovered:
            existing = s.scalar(select(Model).where(Model.name == m.name))
            if existing is not None:
                for attr, val in (
                    ("provider", m.provider),
                    ("family", m.family),
                    ("size_params_b", m.size_params_b),
                    ("context_window", m.context_window),
                    ("cost_in_per_mtok", m.cost_in_per_mtok),
                    ("cost_out_per_mtok", m.cost_out_per_mtok),
                ):
                    if val is not None:
                        setattr(existing, attr, val)
                updated += 1
                continue
            s.add(
                Model(
                    name=m.name,
                    provider=m.provider,
                    family=m.family,
                    local=(m.provider == "lm_studio"),
                    size_params_b=m.size_params_b,
                    context_window=m.context_window,
                    cost_in_per_mtok=m.cost_in_per_mtok,
                    cost_out_per_mtok=m.cost_out_per_mtok,
                )
            )
            added += 1
    console.print(
        f"[green]OK[/green] registered: {added}, updated: {updated}, "
        f"total discovered: {len(discovered)}"
    )


def _parse_duration(s: str) -> "object":
    from datetime import timedelta

    m = re.fullmatch(r"\s*(\d+)\s*([dhm])\s*", s)
    if not m:
        raise typer.BadParameter("--since must look like '7d', '24h', or '30m'.")
    n = int(m.group(1))
    unit = m.group(2)
    return {
        "d": timedelta(days=n),
        "h": timedelta(hours=n),
        "m": timedelta(minutes=n),
    }[unit]


if __name__ == "__main__":
    app()

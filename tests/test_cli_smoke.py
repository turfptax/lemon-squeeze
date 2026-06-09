"""CLI smoke tests via typer's CliRunner.

Covers the command wiring and error paths. Heavy lifting (router/report/etc.)
is tested elsewhere — these tests verify that the CLI plumbing is correctly
hooked up.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from lemon_squeeze.cli import app
from lemon_squeeze.db import Model, Prompt, PromptTag, get_session

runner = CliRunner()


# ---------- Top-level commands ----------------------------------------------


def test_version_prints_a_version_string():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "lemon-squeeze" in result.stdout


def test_help_lists_all_subcommand_groups():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("db", "ingest", "classify", "models", "eval", "route", "bench",
                "providers", "report", "compare", "export", "import"):
        assert cmd in result.stdout


# ---------- db --------------------------------------------------------------


def test_db_init_succeeds_on_clean_engine():
    result = runner.invoke(app, ["db", "init"])
    assert result.exit_code == 0
    assert "Database initialized" in result.stdout


def test_db_stats_runs_against_empty_db():
    result = runner.invoke(app, ["db", "stats"])
    assert result.exit_code == 0
    assert "Prompts" in result.stdout


# ---------- models ----------------------------------------------------------


def test_models_register_inserts_then_updates():
    r1 = runner.invoke(app, [
        "models", "register", "test/cli-model",
        "--provider", "test", "--size-b", "3", "--ctx", "4096",
    ])
    assert r1.exit_code == 0
    assert "Registered" in r1.stdout

    r2 = runner.invoke(app, [
        "models", "register", "test/cli-model", "--size-b", "5",
    ])
    assert r2.exit_code == 0
    assert "Updated" in r2.stdout

    with get_session() as s:
        m = s.scalars(__import__("sqlalchemy").select(Model).where(Model.name == "test/cli-model")).first()
    assert m is not None
    assert m.size_params_b == 5.0


def test_models_list_renders_table():
    runner.invoke(app, ["models", "register", "test/list-m", "--size-b", "1"])
    r = runner.invoke(app, ["models", "list"])
    assert r.exit_code == 0
    assert "test/list-m" in r.stdout


# ---------- doctor / report -------------------------------------------------


def test_doctor_runs_all_checks():
    r = runner.invoke(app, ["doctor"])
    # The conftest seeds the schema + taxonomy via init_db, so schema/taxonomy
    # always pass. Env file + provider may warn; that's exit 0. Any FAIL would
    # indicate a setup regression — exit_code 0 strictly required.
    assert r.exit_code == 0, f"doctor reported FAIL state:\n{r.stdout}"
    assert "schema" in r.stdout.lower()


def test_report_json_output(tmp_path: Path):
    out = tmp_path / "rep.json"
    r = runner.invoke(app, ["report", "--json", str(out)])
    assert r.exit_code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1


def test_report_html_output(tmp_path: Path):
    out = tmp_path / "rep.html"
    r = runner.invoke(app, ["report", "--html", str(out)])
    assert r.exit_code == 0
    html = out.read_text(encoding="utf-8")
    assert "<html" in html


# ---------- classify --------------------------------------------------------


def test_classify_run_handles_empty_db():
    r = runner.invoke(app, ["classify", "run"])
    assert r.exit_code == 0
    assert "Ensemble members" in r.stdout


# ---------- ingest seed -----------------------------------------------------


def test_ingest_seed_jsonl(tmp_path: Path):
    f = tmp_path / "seed.jsonl"
    f.write_text(json.dumps({"prompt": "Hello world", "intended_tag": "test"}), encoding="utf-8")
    r = runner.invoke(app, ["ingest", "seed", str(f)])
    assert r.exit_code == 0
    assert "inserted" in r.stdout


# ---------- bench load ------------------------------------------------------


def test_bench_load_starter():
    r = runner.invoke(app, ["bench", "load", "benchmarks/starter"])
    assert r.exit_code == 0
    assert "inserted" in r.stdout


# ---------- export / import -------------------------------------------------


def test_export_then_import(tmp_path: Path):
    # Seed at least one Prompt so the export has content.
    runner.invoke(app, ["ingest", "seed", str(_make_tiny_seed(tmp_path))])

    out = tmp_path / "snapshot"
    r1 = runner.invoke(app, ["export", str(out)])
    assert r1.exit_code == 0
    assert "Exported" in r1.stdout

    r2 = runner.invoke(app, ["import", str(out)])
    assert r2.exit_code == 0
    assert "Imported" in r2.stdout


def _make_tiny_seed(tmp_path: Path) -> Path:
    f = tmp_path / "tiny.jsonl"
    f.write_text(json.dumps({"prompt": "Tiny prompt", "intended_tag": "test"}), encoding="utf-8")
    return f


# ---------- route -----------------------------------------------------------


def test_route_pick_returns_no_recommendation_on_empty_db():
    r = runner.invoke(app, ["route", "pick", "Write a Python prime checker"])
    assert r.exit_code == 0
    # On empty DB: tags get classified but there's no model history.
    assert "Tags" in r.stdout


def test_route_pick_rejects_unknown_preset():
    r = runner.invoke(app, ["route", "pick", "Test prompt", "--preset", "nonsense"])
    # router.recommend raises ValueError; CLI propagates as a non-zero exit.
    assert r.exit_code != 0


# ---------- compare ---------------------------------------------------------


def test_compare_reports_unknown_models():
    r = runner.invoke(app, ["compare", "ghost", "phantom"])
    assert r.exit_code == 1
    assert "unknown" in r.stdout.lower()


# ---------- providers -------------------------------------------------------


def test_providers_list_handles_unreachable_gracefully():
    """Both LM Studio + OpenRouter unreachable should not crash the command.

    Asserts `ls.called` so a refactor that bypasses the patched name (the test
    would silently start hitting the real network and could pass on a dev
    machine without LM Studio running).
    """
    with patch("lemon_squeeze.cli.list_lm_studio_models") as ls, \
         patch("lemon_squeeze.cli.list_openrouter_models") as lo:
        ls.side_effect = httpx.ConnectError("nope")
        lo.side_effect = httpx.ConnectError("nope")
        r = runner.invoke(app, ["providers", "list"])
    assert ls.called and lo.called  # patches actually attached
    assert r.exit_code == 0
    assert "unreachable" in r.stdout


def test_providers_sync_dry_run_writes_nothing():
    from lemon_squeeze.providers import DiscoveredModel

    with patch("lemon_squeeze.cli.list_lm_studio_models") as ls:
        ls.return_value = [DiscoveredModel(provider="lm_studio", name="local/foo")]
        r = runner.invoke(app, ["providers", "sync", "--dry-run", "--no-openrouter"])
    assert r.exit_code == 0
    assert "dry-run" in r.stdout


# ---------- eval score ------------------------------------------------------


def test_eval_score_with_no_runs_in_db():
    r = runner.invoke(app, ["eval", "score", "rubrics/contains_python_block.yaml"])
    assert r.exit_code == 0
    assert "evaluated" in r.stdout


# ---------- providers sync persists size_b ----------------------------------


def test_providers_sync_persists_size_b_from_discovery(monkeypatch):
    """Caught against real LM Studio: `providers sync` only wrote
    provider/family/context_window/cost; `size_params_b` was dropped on the
    floor on both insert and update paths even when
    `DiscoveredModel.size_params_b` was populated. Net effect: router showed
    "?" for every freshly-discovered LM Studio model because the size axis
    had nothing to score against."""
    from sqlalchemy import select

    from lemon_squeeze.db import Model, get_session
    from lemon_squeeze.providers import DiscoveredModel

    def fake_list_lm_studio():
        return [
            DiscoveredModel(
                provider="lm_studio", name="qwen3.5-2b",
                family="qwen3.5", size_params_b=2.0,
            ),
            DiscoveredModel(
                provider="lm_studio", name="smollm2-135m-instruct",
                family="smollm2", size_params_b=0.135,
            ),
        ]

    monkeypatch.setattr(
        "lemon_squeeze.cli.list_lm_studio_models", fake_list_lm_studio
    )

    # Insert path: model doesn't exist yet.
    r = runner.invoke(app, ["providers", "sync"])
    assert r.exit_code == 0
    with get_session() as s:
        qwen = s.scalar(select(Model).where(Model.name == "qwen3.5-2b"))
        smol = s.scalar(select(Model).where(Model.name == "smollm2-135m-instruct"))
    assert qwen.size_params_b == 2.0, "insert path dropped size_params_b"
    assert smol.size_params_b == 0.135

    # Update path: model already exists with no size; sync should fill it in.
    with get_session() as s:
        m = s.scalar(select(Model).where(Model.name == "qwen3.5-2b"))
        m.size_params_b = None  # simulate the pre-fix state
    runner.invoke(app, ["providers", "sync"])
    with get_session() as s:
        qwen2 = s.scalar(select(Model).where(Model.name == "qwen3.5-2b"))
    assert qwen2.size_params_b == 2.0, "update path dropped size_params_b"


# ---------- bench run wires auto-classify ----------------------------------


def test_bench_run_classifies_after_fanout(monkeypatch, tmp_path: Path):
    """Caught against real LM Studio: `lemon bench run` ran load + fanout +
    score successfully, said "30 evals written", but PromptTag rows count
    stayed at 0. Downstream commands -- `report`, `route pick`, dashboard
    heatmap -- all show empty per-tag tables because the per-tag scorecard
    is computed from PromptTag, and `bench run` skipped the classify step
    that `bench load` runs.

    `bench.run()` is a low-level pipeline (load/fanout/score). Classification
    belongs at the CLI altitude, which is also where `bench load` puts it.
    This test mocks `bench.run()` and asserts the CLI calls _auto_classify
    on top, by checking PromptTag rows exist after `bench run`."""
    from sqlalchemy import func, select

    from lemon_squeeze import bench as bench_mod
    from lemon_squeeze.db import Prompt, PromptTag, get_session

    # Pre-populate one prompt so the heuristic classifier has something to
    # tag. Keep the bench.run() mock side-effect-free.
    from lemon_squeeze.utils import count_tokens, hash_prompt
    content = "Write a Python function that returns 42."
    with get_session() as s:
        s.add(Prompt(content=content, content_hash=hash_prompt(content),
                     char_count=len(content), token_count=count_tokens(content),
                     source="test"))

    def fake_bench_run(*args, **kwargs):
        return bench_mod.BenchReport(bench_name="starter")

    monkeypatch.setattr("lemon_squeeze.cli.bench_mod.run", fake_bench_run)

    bench_dir = tmp_path / "bench"
    (bench_dir / "prompts").mkdir(parents=True)
    (bench_dir / "prompts" / "coding.jsonl").write_text("")

    r = runner.invoke(app, ["bench", "run", str(bench_dir)])
    assert r.exit_code == 0
    with get_session() as s:
        tag_count = s.scalar(select(func.count()).select_from(PromptTag))
    assert tag_count > 0, (
        "bench run finished but no PromptTag rows -- classify step missing"
    )

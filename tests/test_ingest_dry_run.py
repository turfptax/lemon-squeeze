"""--dry-run flag across ingest commands — counts correctly, writes nothing."""
import json
from pathlib import Path

from sqlalchemy import func, select
from typer.testing import CliRunner

from lemon_squeeze.cli import app
from lemon_squeeze.db import Prompt, get_session
from lemon_squeeze.ingestion.self_generated import SeedFileIngester

runner = CliRunner()


def _count_prompts() -> int:
    with get_session() as s:
        return s.scalar(select(func.count()).select_from(Prompt)) or 0


# ---------- Library-level (Ingester.run(dry_run=True)) ----------------------


def test_seed_ingester_dry_run_inserts_nothing(tmp_path: Path):
    f = tmp_path / "seed.jsonl"
    f.write_text(
        "\n".join([
            json.dumps({"prompt": "Hello world"}),
            json.dumps({"prompt": "Another one"}),
        ]),
        encoding="utf-8",
    )
    before = _count_prompts()
    result = SeedFileIngester(f).run(dry_run=True)
    after = _count_prompts()

    assert result.inserted == 2  # would-have-inserted
    assert result.duplicates == 0
    assert after == before  # nothing actually written


def test_seed_ingester_dry_run_counts_existing_as_duplicates(tmp_path: Path):
    """Run once normally, then dry-run the same file — should show 2 duplicates."""
    f = tmp_path / "seed.jsonl"
    f.write_text(
        "\n".join([
            json.dumps({"prompt": "Hello world"}),
            json.dumps({"prompt": "Another one"}),
        ]),
        encoding="utf-8",
    )

    real = SeedFileIngester(f).run()
    assert real.inserted == 2

    after_real = _count_prompts()
    dry = SeedFileIngester(f).run(dry_run=True)
    after_dry = _count_prompts()

    assert dry.duplicates == 2
    assert dry.inserted == 0
    assert after_dry == after_real  # dry-run didn't add or remove


def test_dry_run_counts_intra_batch_duplicates_correctly(tmp_path: Path):
    """If the seed file itself has the same prompt twice, dry-run reports
    1 would-insert + 1 duplicate (matching real-run behavior)."""
    f = tmp_path / "seed.jsonl"
    f.write_text(
        "\n".join([
            json.dumps({"prompt": "Same prompt"}),
            json.dumps({"prompt": "Same prompt"}),  # dup within file
        ]),
        encoding="utf-8",
    )
    result = SeedFileIngester(f).run(dry_run=True)
    assert result.inserted == 1
    assert result.duplicates == 1


# ---------- CLI surface -----------------------------------------------------


def _make_seed(tmp_path: Path, n: int) -> Path:
    f = tmp_path / "seed.jsonl"
    f.write_text(
        "\n".join(json.dumps({"prompt": f"unique prompt {i}"}) for i in range(n)),
        encoding="utf-8",
    )
    return f


def test_cli_seed_dry_run_writes_nothing(tmp_path: Path):
    f = _make_seed(tmp_path, 3)
    before = _count_prompts()
    r = runner.invoke(app, ["ingest", "seed", str(f), "--dry-run"])
    assert r.exit_code == 0
    after = _count_prompts()
    assert after == before
    # Output should make clear it's a preview.
    assert "dry-run" in r.stdout
    assert "would-insert" in r.stdout
    assert "3" in r.stdout


def test_cli_seed_without_dry_run_actually_writes(tmp_path: Path):
    f = _make_seed(tmp_path, 2)
    before = _count_prompts()
    r = runner.invoke(app, ["ingest", "seed", str(f)])
    assert r.exit_code == 0
    after = _count_prompts()
    assert after == before + 2
    assert "dry-run" not in r.stdout
    assert "inserted" in r.stdout


def test_cli_dry_run_then_real_run_gives_consistent_counts(tmp_path: Path):
    """If dry-run says '5 would-insert', real run should also say '5 inserted'."""
    f = _make_seed(tmp_path, 5)
    dry_r = runner.invoke(app, ["ingest", "seed", str(f), "--dry-run"])
    real_r = runner.invoke(app, ["ingest", "seed", str(f)])
    assert dry_r.exit_code == 0 and real_r.exit_code == 0
    # Both should report "5" in their respective sections.
    assert "5" in dry_r.stdout
    assert "5" in real_r.stdout


def test_cli_dry_run_flag_present_on_all_ingest_subcommands():
    """Smoke that the flag is wired on every ingest command's --help."""
    for sub in ("seed", "claude", "grok", "openrouter", "lm-studio", "ai-harness"):
        r = runner.invoke(app, ["ingest", sub, "--help"])
        assert r.exit_code == 0, f"ingest {sub} --help failed"
        assert "--dry-run" in r.stdout, f"--dry-run flag missing on ingest {sub}"


# ---------- AI Harness dry-run (multi-table) -------------------------------


def test_ai_harness_dry_run_writes_nothing(tmp_path: Path):
    """AIHarnessImporter writes to 4 tables; dry-run must roll all of them back."""
    import sqlite3

    from sqlalchemy import func, select

    from lemon_squeeze.db import (
        Evaluation,
        Model,
        Prompt,
        Run,
        get_session,
    )
    from lemon_squeeze.ingestion.ai_harness import AIHarnessImporter

    # Build a tiny fake harness_logs.db with one run.
    harness_db = tmp_path / "harness_logs.db"
    with sqlite3.connect(harness_db) as con:
        con.execute("""
            CREATE TABLE runs (
                id TEXT PRIMARY KEY, task TEXT, supervisor_model TEXT,
                worker_model TEXT, worker_sequence TEXT,
                coder_flagged_complete INTEGER, success INTEGER,
                total_tokens INTEGER, num_loops INTEGER, timestamp TEXT,
                estimated_api_cost_usd REAL, run_duration_seconds REAL,
                complexity_score INTEGER, usefulness_score INTEGER,
                data_value_score INTEGER, scalability_score INTEGER,
                project_id TEXT, conversation_messages TEXT,
                tools_used TEXT, tool_results TEXT
            )
        """)
        con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("test-run-1", "Write hello world", "supervisor", "worker",
             None, 0, 1, 100, 1, "2025-01-01T00:00:00", 0.01, 1.0,
             None, None, None, None, None,
             '[{"role":"assistant","content":"print(\\"hello\\")"}]',
             None, None),
        )

    with get_session() as s:
        before_prompts = s.scalar(select(func.count()).select_from(Prompt))
        before_runs = s.scalar(select(func.count()).select_from(Run))
        before_models = s.scalar(select(func.count()).select_from(Model))
        before_evals = s.scalar(select(func.count()).select_from(Evaluation))

    result = AIHarnessImporter(harness_db).run(dry_run=True)

    with get_session() as s:
        after_prompts = s.scalar(select(func.count()).select_from(Prompt))
        after_runs = s.scalar(select(func.count()).select_from(Run))
        after_models = s.scalar(select(func.count()).select_from(Model))
        after_evals = s.scalar(select(func.count()).select_from(Evaluation))

    # Counters report what would have happened.
    assert result.runs_imported == 1
    assert result.prompts_inserted == 1
    assert result.models_registered == 1
    assert result.evaluations_inserted == 1  # one human_pass eval (success column)

    # But nothing got persisted.
    assert after_prompts == before_prompts
    assert after_runs == before_runs
    assert after_models == before_models
    assert after_evals == before_evals


def test_ai_harness_dry_run_then_real_run_produces_consistent_counts(tmp_path: Path):
    """If dry-run says 'would import 1 run', real run should also import 1."""
    import sqlite3

    from lemon_squeeze.ingestion.ai_harness import AIHarnessImporter

    harness_db = tmp_path / "harness_logs.db"
    with sqlite3.connect(harness_db) as con:
        con.execute("""
            CREATE TABLE runs (
                id TEXT PRIMARY KEY, task TEXT, supervisor_model TEXT,
                worker_model TEXT, worker_sequence TEXT,
                coder_flagged_complete INTEGER, success INTEGER,
                total_tokens INTEGER, num_loops INTEGER, timestamp TEXT,
                estimated_api_cost_usd REAL, run_duration_seconds REAL,
                complexity_score INTEGER, usefulness_score INTEGER,
                data_value_score INTEGER, scalability_score INTEGER,
                project_id TEXT, conversation_messages TEXT,
                tools_used TEXT, tool_results TEXT
            )
        """)
        con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("consistency-run", "Test task", "sup", "wrk",
             None, 0, 1, 50, 1, "2025-01-01T00:00:00", 0.005, 0.5,
             None, None, None, None, None, None, None, None),
        )

    dry_result = AIHarnessImporter(harness_db).run(dry_run=True)
    real_result = AIHarnessImporter(harness_db).run()

    assert dry_result.runs_imported == real_result.runs_imported
    assert dry_result.prompts_inserted == real_result.prompts_inserted
    assert dry_result.models_registered == real_result.models_registered
    assert dry_result.evaluations_inserted == real_result.evaluations_inserted

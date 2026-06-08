import json
import sqlite3
from pathlib import Path

import pytest

from lemon_squeeze.db import Evaluation, Model, Prompt, Run, get_session
from lemon_squeeze.ingestion.ai_harness import AIHarnessImporter

AI_HARNESS_SCHEMA = """
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    task TEXT,
    supervisor_model TEXT,
    worker_model TEXT,
    worker_sequence TEXT,
    coder_flagged_complete INTEGER,
    success INTEGER,
    total_tokens INTEGER,
    num_loops INTEGER,
    timestamp TEXT,
    estimated_api_cost_usd REAL,
    run_duration_seconds REAL,
    complexity_score INTEGER,
    usefulness_score INTEGER,
    data_value_score INTEGER,
    scalability_score INTEGER,
    project_id TEXT,
    conversation_messages TEXT,
    tools_used TEXT,
    tool_results TEXT
);
"""


@pytest.fixture
def fake_harness_db(tmp_path: Path) -> Path:
    db = tmp_path / "harness_logs.db"
    with sqlite3.connect(db) as con:
        con.executescript(AI_HARNESS_SCHEMA)
        con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "run-a",
                "Write a Python function to reverse a string.",
                "anthropic/claude-opus-4-6",
                "anthropic/claude-sonnet-4-6",
                json.dumps(["Coder", "Tester"]),
                1,
                1,
                12345,
                3,
                "2025-04-04T12:00:00",
                0.087,
                4.2,
                3,
                4,
                3,
                2,
                None,
                json.dumps(
                    [
                        {"role": "user", "content": "Write a Python function to reverse a string."},
                        {"role": "assistant", "content": "def rev(s): return s[::-1]"},
                    ]
                ),
                json.dumps(["python_exec"]),
                None,
            ),
        )
        # Second run, no human label, no auto-scores.
        con.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "run-b",
                "Summarize the theory of relativity.",
                "anthropic/claude-opus-4-6",
                "anthropic/claude-sonnet-4-6",
                None, 0, None, 5000, 1, "2025-04-04T13:00:00",
                0.04, 2.1, None, None, None, None, None,
                json.dumps([{"role": "assistant", "content": "Relativity says..."}]),
                None, None,
            ),
        )
    return db


def test_full_import_writes_all_tables(fake_harness_db: Path):
    result = AIHarnessImporter(fake_harness_db).run()
    assert result.runs_seen == 2
    assert result.runs_imported == 2
    assert result.prompts_inserted == 2
    assert result.models_registered == 1  # both runs share worker_model
    # run-a has human label + 4 auto-scores = 5; run-b has none.
    assert result.evaluations_inserted == 5

    with get_session() as s:
        prompts = s.query(Prompt).all()
        models = s.query(Model).all()
        runs = s.query(Run).all()
        evals = s.query(Evaluation).all()

    assert len(prompts) == 2
    assert all(p.source == "ai_harness" for p in prompts)
    assert {m.name for m in models} == {"anthropic/claude-sonnet-4-6"}
    assert all(m.provider == "anthropic" for m in models)
    assert len(runs) == 2
    # Run for run-a should have the response extracted.
    run_a = next(r for r in runs if r.run_metadata["ai_harness_id"] == "run-a")
    assert run_a.response == "def rev(s): return s[::-1]"
    assert run_a.cost_usd == 0.087
    assert run_a.latency_ms == 4200

    rubrics = {e.rubric for e in evals}
    assert "human_pass" in rubrics
    assert {"complexity", "usefulness", "data_value", "scalability"}.issubset(rubrics)


def test_importer_is_idempotent(fake_harness_db: Path):
    AIHarnessImporter(fake_harness_db).run()
    second = AIHarnessImporter(fake_harness_db).run()
    assert second.runs_imported == 0
    assert second.runs_skipped_existing == 2
    assert second.prompts_inserted == 0
    assert second.evaluations_inserted == 0


def test_importer_dedupes_prompt_against_other_sources(fake_harness_db: Path, tmp_path: Path):
    """A prompt that already exists from another source shouldn't be re-inserted —
    the AI Harness run should attach to the existing prompt."""
    from lemon_squeeze.ingestion.self_generated import SeedFileIngester

    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps(["Write a Python function to reverse a string."]))
    SeedFileIngester(seed).run()

    result = AIHarnessImporter(fake_harness_db).run()
    assert result.prompts_inserted == 1  # only run-b's task is new
    assert result.prompts_deduped == 1   # run-a's task matched the seed

    with get_session() as s:
        prompts = s.query(Prompt).all()
        # 2 total: the seed-source one + run-b's
        assert len(prompts) == 2

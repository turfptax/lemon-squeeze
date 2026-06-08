"""`lemon judge` — ad-hoc scoring without DB."""
import json
from pathlib import Path

from typer.testing import CliRunner

from lemon_squeeze.cli import app

runner = CliRunner()
RUBRICS_DIR = Path(__file__).resolve().parents[1] / "rubrics"


def test_judge_with_contains_rubric_passes_when_substring_present():
    r = runner.invoke(app, [
        "judge", str(RUBRICS_DIR / "no_refusal.yaml"),
        "--prompt", "test prompt",
        "--response", "Here is the answer: 42.",
    ])
    assert r.exit_code == 0
    assert "PASS" in r.stdout
    assert "score=1.00" in r.stdout


def test_judge_with_contains_rubric_fails_on_refusal():
    r = runner.invoke(app, [
        "judge", str(RUBRICS_DIR / "no_refusal.yaml"),
        "--prompt", "test",
        "--response", "I'm sorry, but I can't help with that.",
    ])
    assert r.exit_code == 0
    assert "FAIL" in r.stdout


def test_judge_with_per_prompt_rubric_uses_metadata():
    r = runner.invoke(app, [
        "judge", str(RUBRICS_DIR / "per_prompt_expected.yaml"),
        "--prompt", "What is 2+2?",
        "--response", "The answer is 4.",
        "--metadata", json.dumps({"expected_contains": ["4"]}),
    ])
    assert r.exit_code == 0
    assert "PASS" in r.stdout


def test_judge_per_prompt_skips_when_metadata_missing():
    r = runner.invoke(app, [
        "judge", str(RUBRICS_DIR / "per_prompt_expected.yaml"),
        "--prompt", "x",
        "--response", "y",
    ])
    assert r.exit_code == 0
    assert "SKIPPED" in r.stdout


def test_judge_requires_response_or_response_file():
    r = runner.invoke(app, [
        "judge", str(RUBRICS_DIR / "no_refusal.yaml"),
        "--prompt", "x",
    ])
    assert r.exit_code == 2
    assert "response" in r.stdout.lower()


def test_judge_response_file_works(tmp_path: Path):
    f = tmp_path / "response.txt"
    f.write_text("This is a clean response with no refusals.", encoding="utf-8")
    r = runner.invoke(app, [
        "judge", str(RUBRICS_DIR / "no_refusal.yaml"),
        "--prompt", "x",
        "--response-file", str(f),
    ])
    assert r.exit_code == 0
    assert "PASS" in r.stdout


def test_judge_invalid_metadata_json_errors_clearly():
    r = runner.invoke(app, [
        "judge", str(RUBRICS_DIR / "no_refusal.yaml"),
        "--prompt", "x",
        "--response", "y",
        "--metadata", "not-json",
    ])
    assert r.exit_code == 2
    assert "Invalid --metadata JSON" in r.stdout


def test_judge_does_not_touch_the_db():
    """Ad-hoc scoring should not write anything to the DB."""
    from sqlalchemy import func, select

    from lemon_squeeze.db import Evaluation, get_session

    with get_session() as s:
        before = s.scalar(select(func.count()).select_from(Evaluation)) or 0

    r = runner.invoke(app, [
        "judge", str(RUBRICS_DIR / "no_refusal.yaml"),
        "--prompt", "x",
        "--response", "clean answer",
    ])
    assert r.exit_code == 0

    with get_session() as s:
        after = s.scalar(select(func.count()).select_from(Evaluation)) or 0
    assert after == before  # no DB writes from `lemon judge`

"""Auto-classify after ingest + auto-backfill after train-ml.

Both are CLI ergonomics that eliminate "and then also" patterns from the docs.
"""
import json
from pathlib import Path

from sqlalchemy import func, select
from typer.testing import CliRunner

from lemon_squeeze.cli import app
from lemon_squeeze.db import Prompt, PromptTag, get_session

runner = CliRunner()


def _count_prompt_tags(classifier: str = "heuristic") -> int:
    with get_session() as s:
        return s.scalar(
            select(func.count()).select_from(PromptTag)
            .where(PromptTag.classifier == classifier)
        ) or 0


def _seed_file(tmp_path: Path, prompts: list[str]) -> Path:
    f = tmp_path / "seed.jsonl"
    f.write_text(
        "\n".join(json.dumps({"prompt": p}) for p in prompts),
        encoding="utf-8",
    )
    return f


# ---------- auto-classify after ingest --------------------------------------


def test_ingest_seed_auto_classifies_new_prompts(tmp_path: Path):
    """After `lemon ingest seed FILE`, freshly-inserted prompts should already
    have heuristic tags — no separate `classify run` needed."""
    f = _seed_file(tmp_path, [
        "Write a Python function to sort a list",  # coding
        "What is the capital of France",            # qa_factual
    ])
    before = _count_prompt_tags("heuristic")
    r = runner.invoke(app, ["ingest", "seed", str(f)])
    assert r.exit_code == 0
    after = _count_prompt_tags("heuristic")
    assert after > before
    # The summary line should mention auto-classify when tags were written.
    assert "auto-classify" in r.stdout


def test_ingest_seed_no_classify_opt_out(tmp_path: Path):
    f = _seed_file(tmp_path, [
        "Write a Python function to sort a list",
    ])
    before = _count_prompt_tags("heuristic")
    r = runner.invoke(app, ["ingest", "seed", str(f), "--no-classify"])
    assert r.exit_code == 0
    after = _count_prompt_tags("heuristic")
    assert after == before  # no heuristic tags written
    assert "auto-classify" not in r.stdout


def test_dry_run_does_not_trigger_classify(tmp_path: Path):
    f = _seed_file(tmp_path, ["Write a Python function"])
    before = _count_prompt_tags("heuristic")
    r = runner.invoke(app, ["ingest", "seed", str(f), "--dry-run"])
    assert r.exit_code == 0
    after = _count_prompt_tags("heuristic")
    assert after == before  # nothing persisted, including tags


def test_bench_load_auto_classifies(tmp_path: Path):
    """Bench load follows the same pattern."""
    bench_dir = tmp_path / "tiny_bench"
    (bench_dir / "prompts").mkdir(parents=True)
    (bench_dir / "prompts" / "x.jsonl").write_text(
        json.dumps({"prompt": "Write Python", "intended_tag": "coding"}),
        encoding="utf-8",
    )
    before = _count_prompt_tags("heuristic")
    r = runner.invoke(app, ["bench", "load", str(bench_dir)])
    assert r.exit_code == 0
    after = _count_prompt_tags("heuristic")
    assert after > before


def test_auto_classify_is_idempotent_on_already_tagged_prompts(tmp_path: Path):
    """Running ingest twice on the same file should not double-tag."""
    f = _seed_file(tmp_path, ["Write a Python function to sort a list"])
    runner.invoke(app, ["ingest", "seed", str(f)])
    after_first = _count_prompt_tags("heuristic")

    runner.invoke(app, ["ingest", "seed", str(f)])
    after_second = _count_prompt_tags("heuristic")
    assert after_second == after_first  # second pass is a no-op


# ---------- auto-backfill after train-ml ------------------------------------


def _seed_labeled_prompts() -> None:
    """Enough labels (≥3 per tag) so train-ml succeeds."""
    coding = [
        "Write a Python function fibonacci",
        "Define a class with methods for a stack",
        "Implement a recursive function in Python",
        "Build a Python script that reads files",
        "Create a Python function to sort lists",
    ]
    math = [
        "What is the integral of x squared",
        "Compute the derivative of sin x",
        "Solve 3x + 7 = 22",
        "Calculate area of triangle base 5",
        "What is 12 factorial",
    ]
    with get_session() as s:
        for i, content in enumerate(coding + math):
            p = Prompt(
                content=content,
                content_hash=f"backfill-{i}",
                char_count=len(content),
                source="test",
            )
            s.add(p); s.flush()
            tag = "coding" if i < len(coding) else "math"
            s.add(PromptTag(
                prompt_id=p.id, tag=tag, classifier="human", confidence=1.0,
            ))


def test_train_ml_auto_backfills_ml_tags():
    """After train-ml, prompts should have ML tags without a separate command."""
    _seed_labeled_prompts()

    before_ml = _count_prompt_tags("ml")
    r = runner.invoke(app, ["classify", "train-ml"])
    assert r.exit_code == 0
    after_ml = _count_prompt_tags("ml")
    assert after_ml > before_ml
    assert "auto-backfill" in r.stdout


def test_train_ml_no_backfill_opt_out():
    _seed_labeled_prompts()

    before_ml = _count_prompt_tags("ml")
    r = runner.invoke(app, ["classify", "train-ml", "--no-backfill"])
    assert r.exit_code == 0
    after_ml = _count_prompt_tags("ml")
    assert after_ml == before_ml
    assert "auto-backfill" not in r.stdout

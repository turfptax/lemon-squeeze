"""Export → import round-trip preserves the DB."""
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select

from lemon_squeeze.db import (
    Evaluation,
    Model,
    Prompt,
    PromptTag,
    Run,
    TagTaxonomy,
    get_session,
)
from lemon_squeeze.portable import (
    EXPORT_ID_KEY,
    export_to_dir,
    import_from_dir,
)


def _seed_small_graph() -> None:
    """Seed a small but representative DB graph for round-trip testing."""
    with get_session() as s:
        # TagTaxonomy isn't truncated between tests (it's session-seeded by
        # init_db). Only insert our test row if not already present.
        if not s.scalar(select(TagTaxonomy).where(TagTaxonomy.tag == "custom_export_tag")):
            s.add(TagTaxonomy(tag="custom_export_tag", description="for export test"))

        p1 = Prompt(content="Write code", content_hash="ph-code",
                    char_count=10, source="test")
        p2 = Prompt(content="Solve math", content_hash="ph-math",
                    char_count=10, source="test",
                    source_metadata={"expected_contains": ["42"]})
        s.add(p1); s.add(p2); s.flush()

        s.add(PromptTag(prompt_id=p1.id, tag="coding", classifier="heuristic", confidence=0.9))
        s.add(PromptTag(prompt_id=p2.id, tag="math", classifier="heuristic", confidence=0.95))

        m = Model(
            name="tst/m", provider="test", family="tst",
            size_params_b=3.0, context_window=4096, local=True,
            cost_in_per_mtok=0.5, cost_out_per_mtok=1.0,
        )
        s.add(m); s.flush()

        run1 = Run(prompt_id=p1.id, model_id=m.id, response="def x(): ...",
                   tokens_in=20, tokens_out=10, latency_ms=120, cost_usd=0.0001,
                   temperature=0.0, run_metadata={"system": "test"})
        run2 = Run(prompt_id=p2.id, model_id=m.id, response="The answer is 42",
                   tokens_in=15, tokens_out=8, latency_ms=80, cost_usd=0.00005,
                   error=None)
        s.add(run1); s.add(run2); s.flush()

        s.add(Evaluation(run_id=run1.id, rubric="human_pass",
                         score=1.0, passed=True, scored_by="human"))
        s.add(Evaluation(run_id=run2.id, rubric="human_pass",
                         score=1.0, passed=True, scored_by="human"))
        s.add(Evaluation(run_id=run2.id, rubric="llm_judge",
                         score=4.0, passed=None, scored_by="llm",
                         judge_model="gemini-flash"))


def _truncate_all() -> None:
    from sqlalchemy import delete
    with get_session() as s:
        for table in (Evaluation, Run, PromptTag, Prompt, Model):
            s.execute(delete(table))


# ---------- Export -----------------------------------------------------------


def test_export_writes_all_files(tmp_path: Path):
    _seed_small_graph()
    report = export_to_dir(tmp_path / "out")
    expected = {
        "prompts.jsonl", "models.jsonl", "prompt_tags.jsonl",
        "runs.jsonl", "evaluations.jsonl", "tag_taxonomy.jsonl",
        "manifest.json",
    }
    actual = {p.name for p in (tmp_path / "out").iterdir()}
    assert expected.issubset(actual)
    assert report.prompts == 2
    assert report.runs == 2
    assert report.evaluations == 3


def test_export_can_skip_runs(tmp_path: Path):
    _seed_small_graph()
    report = export_to_dir(tmp_path / "out", include_runs=False)
    assert report.runs == 0
    # Evaluations are skipped automatically when runs are excluded.
    assert report.evaluations == 0
    assert not (tmp_path / "out" / "runs.jsonl").exists()


def test_export_run_ids_persist_to_db(tmp_path: Path):
    """The export tags Run.run_metadata with lemon_export_id so subsequent
    imports can dedupe."""
    _seed_small_graph()
    export_to_dir(tmp_path / "out")
    with get_session() as s:
        runs = list(s.scalars(select(Run)).all())
    assert all(EXPORT_ID_KEY in (r.run_metadata or {}) for r in runs)


# ---------- Round-trip -------------------------------------------------------


def test_round_trip_preserves_row_counts(tmp_path: Path):
    _seed_small_graph()
    with get_session() as s:
        before = {
            "prompts": s.scalar(select(func.count()).select_from(Prompt)),
            "models": s.scalar(select(func.count()).select_from(Model)),
            "prompt_tags": s.scalar(select(func.count()).select_from(PromptTag)),
            "runs": s.scalar(select(func.count()).select_from(Run)),
            "evaluations": s.scalar(select(func.count()).select_from(Evaluation)),
        }

    export_to_dir(tmp_path / "out")
    _truncate_all()

    with get_session() as s:
        post_truncate = s.scalar(select(func.count()).select_from(Prompt))
    assert post_truncate == 0

    import_from_dir(tmp_path / "out")

    with get_session() as s:
        after = {
            "prompts": s.scalar(select(func.count()).select_from(Prompt)),
            "models": s.scalar(select(func.count()).select_from(Model)),
            "prompt_tags": s.scalar(select(func.count()).select_from(PromptTag)),
            "runs": s.scalar(select(func.count()).select_from(Run)),
            "evaluations": s.scalar(select(func.count()).select_from(Evaluation)),
        }
    assert before == after


def test_round_trip_preserves_response_content(tmp_path: Path):
    _seed_small_graph()
    export_to_dir(tmp_path / "out")
    _truncate_all()
    import_from_dir(tmp_path / "out")

    with get_session() as s:
        runs = list(s.scalars(select(Run)).all())
    responses = sorted(r.response for r in runs if r.response)
    assert "def x(): ..." in responses
    assert "The answer is 42" in responses


def test_round_trip_preserves_metadata(tmp_path: Path):
    """Prompt.source_metadata must survive — that's what per-prompt rubrics need."""
    _seed_small_graph()
    export_to_dir(tmp_path / "out")
    _truncate_all()
    import_from_dir(tmp_path / "out")

    with get_session() as s:
        math_p = s.scalar(select(Prompt).where(Prompt.content_hash == "ph-math"))
    assert math_p.source_metadata == {"expected_contains": ["42"]}


def test_import_is_idempotent(tmp_path: Path):
    _seed_small_graph()
    export_to_dir(tmp_path / "out")

    # First import: nothing new (everything already there).
    r1 = import_from_dir(tmp_path / "out")
    assert r1.prompts_inserted == 0
    assert r1.runs_inserted == 0
    assert r1.evaluations_inserted == 0
    assert r1.prompts_deduped == 2
    assert r1.runs_deduped == 2

    # Second import: same as the first.
    r2 = import_from_dir(tmp_path / "out")
    assert r2.runs_inserted == 0
    assert r2.evaluations_inserted == 0


def test_import_into_empty_db_writes_everything(tmp_path: Path):
    _seed_small_graph()
    export_to_dir(tmp_path / "out")
    _truncate_all()

    r = import_from_dir(tmp_path / "out")
    assert r.prompts_inserted == 2
    assert r.models_inserted == 1
    assert r.prompt_tags_inserted == 2
    assert r.runs_inserted == 2
    assert r.evaluations_inserted == 3


def test_import_skips_eval_when_run_missing(tmp_path: Path):
    """If somebody hand-edits runs.jsonl out, evaluations referencing those
    runs should be skipped with a message, not crash."""
    _seed_small_graph()
    export_to_dir(tmp_path / "out")
    _truncate_all()

    # Wipe runs.jsonl content.
    (tmp_path / "out" / "runs.jsonl").write_text("", encoding="utf-8")
    r = import_from_dir(tmp_path / "out")
    assert r.runs_inserted == 0
    assert r.evaluations_inserted == 0
    assert any("evaluation references unknown run" in m for m in r.skipped)


def test_round_trip_preserves_char_count_of_zero(tmp_path: Path):
    """Regression: `or len(content)` silently coerced legitimate zero counts."""
    with get_session() as s:
        s.add(Prompt(content="", content_hash="ph-empty", char_count=0, source="test"))

    export_to_dir(tmp_path / "out")
    _truncate_all()
    import_from_dir(tmp_path / "out")

    with get_session() as s:
        p = s.scalar(select(Prompt).where(Prompt.content_hash == "ph-empty"))
    assert p is not None
    assert p.char_count == 0


def test_round_trip_preserves_rubric_hash(tmp_path: Path):
    """Evaluation.rubric_hash (added with staleness detection) must survive
    export+import. Without it, imported evals get NULL hash and are treated
    as 'up-to-date' forever even when the rubric YAML has been edited on
    the importing machine, breaking staleness detection silently."""
    # Seed two evals with non-trivial rubric_hash values.
    with get_session() as s:
        p = Prompt(content="x", content_hash="rh-test", char_count=1, source="test")
        s.add(p); s.flush()
        m = Model(name="rh/m", provider="test", local=True)
        s.add(m); s.flush()
        run = Run(prompt_id=p.id, model_id=m.id, response="y")
        s.add(run); s.flush()
        s.add(Evaluation(
            run_id=run.id, rubric="rh_check",
            rubric_hash="a" * 64,
            score=1.0, passed=True, scored_by="auto",
        ))
        s.add(Evaluation(
            run_id=run.id, rubric="other_rubric",
            rubric_hash="b" * 64,
            score=0.5, passed=False, scored_by="llm",
        ))

    export_to_dir(tmp_path / "out")
    _truncate_all()
    import_from_dir(tmp_path / "out")

    with get_session() as s:
        evals = list(s.scalars(select(Evaluation)).all())
    by_rubric = {e.rubric: e for e in evals}
    assert by_rubric["rh_check"].rubric_hash == "a" * 64, (
        "rubric_hash silently dropped during export+import — staleness "
        "detection would always treat re-imported evals as up-to-date"
    )
    assert by_rubric["other_rubric"].rubric_hash == "b" * 64


def test_import_handles_naive_iso_timestamps(tmp_path: Path):
    """Round-trip should not crash on the naive datetime path —
    UTCDateTime coerces incoming naive datetimes to UTC."""
    _seed_small_graph()
    export_to_dir(tmp_path / "out")

    # Mangle one prompt's created_at to be naive ISO.
    text = (tmp_path / "out" / "prompts.jsonl").read_text(encoding="utf-8")
    text = text.replace("+00:00", "")
    (tmp_path / "out" / "prompts.jsonl").write_text(text, encoding="utf-8")

    _truncate_all()
    r = import_from_dir(tmp_path / "out")
    # Imported anyway.
    assert r.prompts_inserted == 2

    with get_session() as s:
        ps = list(s.scalars(select(Prompt)).all())
    # The naive datetime got coerced to tz-aware UTC by the column type.
    for p in ps:
        if p.created_at is not None:
            assert p.created_at.tzinfo == timezone.utc

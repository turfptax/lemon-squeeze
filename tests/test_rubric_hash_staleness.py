"""Rubric-hash staleness detection — auto-replace stale Evaluation rows
when the rubric YAML has been edited since they were written."""
from sqlalchemy import select

from lemon_squeeze.db import Evaluation, Model, Prompt, Run, get_session
from lemon_squeeze.eval.rubric import Rubric, evaluate_runs


def _seed_one_run_with_response(response: str) -> int:
    with get_session() as s:
        p = Prompt(content="p", content_hash=f"stale-{response}",
                   char_count=1, source="test")
        s.add(p); s.flush()
        m = s.scalar(select(Model).where(Model.name == "stale/m")) or Model(
            name="stale/m", provider="test", local=True
        )
        if m.id is None:
            s.add(m); s.flush()
        run = Run(prompt_id=p.id, model_id=m.id, response=response)
        s.add(run); s.flush()
        return run.id


# ---------- Rubric.config_hash() ----------------------------------------------


def test_config_hash_is_deterministic():
    r1 = Rubric(name="x", description="A", judge_kind="contains",
                judge_config={"all_of": ["foo"]})
    r2 = Rubric(name="x", description="A", judge_kind="contains",
                judge_config={"all_of": ["foo"]})
    assert r1.config_hash() == r2.config_hash()
    assert len(r1.config_hash()) == 64  # SHA-256 hex


def test_config_hash_ignores_description():
    """Editing prose shouldn't invalidate evaluations."""
    r1 = Rubric(name="x", description="original prose",
                judge_kind="contains", judge_config={"all_of": ["foo"]})
    r2 = Rubric(name="x", description="completely rewritten prose",
                judge_kind="contains", judge_config={"all_of": ["foo"]})
    assert r1.config_hash() == r2.config_hash()


def test_config_hash_responds_to_judge_config_change():
    r1 = Rubric(name="x", description="", judge_kind="contains",
                judge_config={"all_of": ["foo"]})
    r2 = Rubric(name="x", description="", judge_kind="contains",
                judge_config={"all_of": ["bar"]})
    assert r1.config_hash() != r2.config_hash()


def test_config_hash_responds_to_judge_kind_change():
    r1 = Rubric(name="x", description="", judge_kind="contains",
                judge_config={"all_of": ["foo"]})
    r2 = Rubric(name="x", description="", judge_kind="regex",
                judge_config={"all_of": ["foo"]})
    assert r1.config_hash() != r2.config_hash()


def test_config_hash_responds_to_applies_to_tags():
    r1 = Rubric(name="x", description="", judge_kind="contains",
                judge_config={"all_of": ["foo"]}, applies_to_tags=["coding"])
    r2 = Rubric(name="x", description="", judge_kind="contains",
                judge_config={"all_of": ["foo"]}, applies_to_tags=["math"])
    assert r1.config_hash() != r2.config_hash()


def test_config_hash_is_order_insensitive_on_tags():
    r1 = Rubric(name="x", description="", judge_kind="contains",
                judge_config={"all_of": ["foo"]}, applies_to_tags=["a", "b"])
    r2 = Rubric(name="x", description="", judge_kind="contains",
                judge_config={"all_of": ["foo"]}, applies_to_tags=["b", "a"])
    assert r1.config_hash() == r2.config_hash()


# ---------- evaluate_runs() integration -------------------------------------


def test_new_evals_get_current_rubric_hash():
    rid = _seed_one_run_with_response("hello world")
    r = Rubric(name="contains_world", description="d", judge_kind="contains",
               judge_config={"all_of": ["world"]})
    evaluate_runs(r)

    with get_session() as s:
        e = s.scalar(select(Evaluation).where(Evaluation.run_id == rid))
    assert e is not None
    assert e.rubric_hash == r.config_hash()


def test_stale_eval_gets_auto_replaced_when_rubric_edited():
    rid = _seed_one_run_with_response("hello world")
    # First version: looks for "world" — passes.
    v1 = Rubric(name="contains_x", description="", judge_kind="contains",
                judge_config={"all_of": ["world"]})
    report1 = evaluate_runs(v1)
    assert report1.evaluations_written == 1
    assert report1.stale_replaced == 0

    # Edit the rubric: now look for "FAIL_TOKEN" instead. Score it again.
    v2 = Rubric(name="contains_x", description="", judge_kind="contains",
                judge_config={"all_of": ["FAIL_TOKEN"]})
    report2 = evaluate_runs(v2)

    # Without staleness detection, this would silently skip (skip_existing=True
    # found a matching name). With staleness detection, it replaces.
    assert report2.stale_replaced == 1
    assert report2.evaluations_written == 1
    assert report2.skipped_existing == 0

    # The eval now reflects the new rubric's verdict (FAIL — "FAIL_TOKEN" missing).
    with get_session() as s:
        e = s.scalar(select(Evaluation).where(Evaluation.run_id == rid))
    assert e.passed is False
    assert e.rubric_hash == v2.config_hash()


def test_up_to_date_eval_is_skipped():
    _seed_one_run_with_response("hello")
    r = Rubric(name="contains_hello", description="", judge_kind="contains",
               judge_config={"all_of": ["hello"]})
    evaluate_runs(r)

    report = evaluate_runs(r)  # same rubric
    assert report.skipped_existing == 1
    assert report.evaluations_written == 0
    assert report.stale_replaced == 0


def test_null_hash_legacy_row_is_not_replaced():
    """Rows from before this feature existed have NULL rubric_hash. We don't
    want to churn legacy data — they're treated as up-to-date."""
    rid = _seed_one_run_with_response("hello world")
    r = Rubric(name="legacy_check", description="", judge_kind="contains",
               judge_config={"all_of": ["world"]})

    # Pre-populate a legacy row with NULL hash directly.
    with get_session() as s:
        s.add(Evaluation(
            run_id=rid, rubric="legacy_check", rubric_hash=None,
            score=1.0, passed=True, scored_by="human",
        ))

    report = evaluate_runs(r)
    assert report.stale_replaced == 0
    assert report.skipped_existing == 1


def test_description_edit_does_not_cause_stale_replacement():
    rid = _seed_one_run_with_response("hello")
    v1 = Rubric(name="desc_check", description="original",
                judge_kind="contains", judge_config={"all_of": ["hello"]})
    evaluate_runs(v1)

    v2 = Rubric(name="desc_check", description="rewritten prose",
                judge_kind="contains", judge_config={"all_of": ["hello"]})
    report = evaluate_runs(v2)
    assert report.stale_replaced == 0
    assert report.skipped_existing == 1


def test_replace_existing_takes_precedence_over_staleness():
    """If caller passes replace_existing=True, that's an unconditional wipe."""
    _seed_one_run_with_response("hello")
    r = Rubric(name="precedence_check", description="", judge_kind="contains",
               judge_config={"all_of": ["hello"]})
    evaluate_runs(r)

    report = evaluate_runs(r, replace_existing=True)
    assert report.replaced >= 1     # the unconditional delete fired
    assert report.evaluations_written == 1
    # stale_replaced doesn't count because we didn't compare hashes.
    assert report.stale_replaced == 0

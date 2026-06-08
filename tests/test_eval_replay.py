"""eval replay (replace_existing) behavior."""
from sqlalchemy import select

from lemon_squeeze.db import Evaluation, Model, Prompt, Run, get_session
from lemon_squeeze.eval.rubric import Rubric, evaluate_runs


def _seed_run(response: str) -> int:
    with get_session() as s:
        p = Prompt(content="t", content_hash=f"h-{response}", char_count=1, source="test")
        m = s.scalar(select(Model).where(Model.name == "test/m")) or Model(
            name="test/m", provider="test", local=True
        )
        if m.id is None:
            s.add(m); s.flush()
        s.add(p); s.flush()
        run = Run(prompt_id=p.id, model_id=m.id, response=response)
        s.add(run); s.flush()
        return run.id


def test_replay_deletes_old_evals_and_rescores():
    run_id = _seed_run("hello world")
    r1 = Rubric(name="contains_x", description="", judge_kind="contains", judge_config={"all_of": ["world"]})
    evaluate_runs(r1)  # writes one Evaluation

    with get_session() as s:
        old = list(s.scalars(select(Evaluation).where(Evaluation.rubric == "contains_x")).all())
    assert len(old) == 1
    assert old[0].passed is True

    # Now change the rubric — the same name, but stricter expectations.
    r2 = Rubric(name="contains_x", description="", judge_kind="contains", judge_config={"all_of": ["FAIL"]})
    report = evaluate_runs(r2, replace_existing=True)
    assert report.replaced == 1
    assert report.evaluations_written == 1

    with get_session() as s:
        new = list(s.scalars(select(Evaluation).where(Evaluation.rubric == "contains_x")).all())
    assert len(new) == 1  # not duplicated
    assert new[0].passed is False  # stricter rubric flips the verdict


def test_replay_with_run_ids_only_affects_those_runs():
    a_id = _seed_run("contains-target")
    b_id = _seed_run("does-not-contain")
    r = Rubric(name="rb", description="", judge_kind="contains", judge_config={"all_of": ["target"]})
    evaluate_runs(r)  # both runs evaluated

    # Replay only on run_id == a_id.
    report = evaluate_runs(r, run_ids=[a_id], replace_existing=True)
    assert report.replaced == 1  # only a's eval

    with get_session() as s:
        evals = list(s.scalars(select(Evaluation).where(Evaluation.rubric == "rb")).all())
    by_run = {e.run_id: e for e in evals}
    # B's original eval untouched.
    assert b_id in by_run
    assert a_id in by_run

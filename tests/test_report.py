"""Executive summary report."""
from sqlalchemy import select

from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run, get_session
from lemon_squeeze.report import build_report


def _seed_full(model_name: str, *, tag: str, prompt_text: str, passes: list[bool], cost: float):
    with get_session() as s:
        prompt = s.scalar(select(Prompt).where(Prompt.content == prompt_text))
        if prompt is None:
            prompt = Prompt(
                content=prompt_text,
                content_hash=f"h-{prompt_text}-{tag}",
                char_count=len(prompt_text),
                source="test",
            )
            s.add(prompt); s.flush()
            s.add(PromptTag(prompt_id=prompt.id, tag=tag, classifier="heuristic", confidence=0.9))
        model = s.scalar(select(Model).where(Model.name == model_name))
        if model is None:
            model = Model(name=model_name, provider="test", size_params_b=1.0, local=True)
            s.add(model); s.flush()
        for ok in passes:
            run = Run(prompt_id=prompt.id, model_id=model.id, response="x", cost_usd=cost)
            s.add(run); s.flush()
            s.add(
                Evaluation(
                    run_id=run.id, rubric="human_pass",
                    score=1.0 if ok else 0.0, passed=ok, scored_by="human",
                )
            )


def test_report_on_empty_db_returns_zeros():
    rep = build_report()
    assert rep.n_prompts == 0
    assert rep.n_models == 0
    assert rep.scorecards == []
    assert rep.gaps == []


def test_report_per_tag_picks_quality_and_cost_separately():
    # Same tag, two models: small cheap is slower-perfect, big expensive is also perfect.
    _seed_full("cheap-3b", tag="coding", prompt_text="p1", passes=[True] * 5, cost=0.001)
    _seed_full("dear-70b", tag="coding", prompt_text="p1", passes=[True] * 5, cost=0.05)
    rep = build_report(min_samples=3)
    coding = next(sc for sc in rep.scorecards if sc.tag == "coding")
    # Quality tied; tie-break picks lower cost.
    assert coding.quality_pick in ("cheap-3b", "dear-70b")
    assert coding.cost_pick == "cheap-3b"
    assert coding.has_qualifying is True
    assert coding.qualifying_models == 2


def test_report_flags_no_qualifying_gap():
    # 60% pass rate at default threshold 0.7 → no qualifying model.
    _seed_full(
        "weak-3b", tag="math", prompt_text="m1",
        passes=[True, True, True, False, False], cost=0.001,
    )
    rep = build_report()
    math = next(sc for sc in rep.scorecards if sc.tag == "math")
    assert math.has_qualifying is False
    gap_tags = [g.tag for g in rep.gaps]
    assert "math" in gap_tags


def test_report_excludes_tags_with_no_prompts():
    # The default taxonomy has lots of tags. Empty-DB report should show none of
    # them as scorecards (only tags that have prompts).
    rep = build_report()
    assert rep.scorecards == []


def test_report_classifies_no_runs_vs_no_evals_gap():
    # tag with prompts but no runs.
    with get_session() as s:
        p = Prompt(content="x", content_hash="x-no-runs", char_count=1, source="t")
        s.add(p); s.flush()
        s.add(PromptTag(prompt_id=p.id, tag="reasoning", classifier="heuristic", confidence=1.0))

    rep = build_report()
    no_run_gap = next(g for g in rep.gaps if g.tag == "reasoning")
    assert no_run_gap.reason == "no_runs"

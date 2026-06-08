"""Head-to-head model compare."""
import pytest
from sqlalchemy import select

from lemon_squeeze.compare import compare
from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run, get_session


def _seed(
    model_name: str,
    *,
    tag: str,
    prompt_text: str,
    passes: list[bool],
    cost: float = 0.001,
    latency_ms: int = 100,
) -> None:
    with get_session() as s:
        prompt = s.scalar(select(Prompt).where(Prompt.content == prompt_text))
        if prompt is None:
            prompt = Prompt(
                content=prompt_text,
                content_hash=f"h-{prompt_text}",
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
            run = Run(
                prompt_id=prompt.id,
                model_id=model.id,
                response="x",
                cost_usd=cost,
                latency_ms=latency_ms,
            )
            s.add(run); s.flush()
            s.add(
                Evaluation(
                    run_id=run.id,
                    rubric="human_pass",
                    score=1.0 if ok else 0.0,
                    passed=ok,
                    scored_by="human",
                )
            )


def test_compare_reports_per_tag_winner_with_enough_samples():
    # 20 samples each — wide enough CIs to declare significance on 100% vs 0%.
    _seed("alpha", tag="coding", prompt_text="code one", passes=[True] * 20)
    _seed("alpha", tag="math", prompt_text="math one", passes=[False] * 20)
    _seed("beta", tag="coding", prompt_text="code one", passes=[False] * 20)
    _seed("beta", tag="math", prompt_text="math one", passes=[True] * 20)

    report = compare("alpha", "beta")
    assert len(report.per_tag) == 2
    by_tag = {tc.tag: tc for tc in report.per_tag}

    assert by_tag["coding"].winner == "A"  # alpha wins coding
    assert by_tag["math"].winner == "B"   # beta wins math
    assert report.a_wins == 1 and report.b_wins == 1 and report.ties == 0
    assert report.overall_winner == "tie"


def test_compare_small_samples_default_to_tie_under_significance():
    """100% vs 67% over 3 samples is not significant — CIs overlap heavily."""
    _seed("a", tag="x", prompt_text="px", passes=[True, True, True])
    _seed("b", tag="x", prompt_text="px", passes=[True, True, False])
    rep = compare("a", "b")  # require_significance=True default
    assert rep.per_tag[0].winner == "tie"
    assert rep.per_tag[0].significant is False

    rep2 = compare("a", "b", require_significance=False)
    # Without the significance gate, 100% vs 67% (delta 33pp > 5pp threshold) → A wins.
    assert rep2.per_tag[0].winner == "A"


def test_compare_tie_threshold_calls_close_results_a_tie():
    """Even with enough samples to be significant, a 5pp delta is within tie threshold."""
    _seed("a", tag="x", prompt_text="px", passes=[True] * 19 + [False])  # 95%
    _seed("b", tag="x", prompt_text="px", passes=[True] * 18 + [False, False])  # 90%
    rep = compare("a", "b", require_significance=False)
    # 5pp delta == 0.05 threshold → ties (not >).
    assert rep.per_tag[0].winner == "tie"


def test_compare_skips_tags_below_min_samples():
    _seed("a", tag="rare", prompt_text="p1", passes=[True])
    _seed("b", tag="rare", prompt_text="p1", passes=[True, True, True, True])
    rep = compare("a", "b", min_samples=3)
    assert rep.per_tag == []  # `a` has only 1 sample


def test_compare_raises_for_unknown_model():
    _seed("real", tag="x", prompt_text="p", passes=[True])
    with pytest.raises(ValueError, match="unknown model"):
        compare("real", "ghost")


def test_compare_overall_winner_when_one_dominates():
    # 15 each — large enough that 100% vs 0% CIs don't overlap.
    for tag in ("t1", "t2", "t3"):
        _seed("a", tag=tag, prompt_text=f"p-{tag}", passes=[True] * 15)
        _seed("b", tag=tag, prompt_text=f"p-{tag}", passes=[False] * 15)
    rep = compare("a", "b")
    assert rep.a_wins == 3 and rep.b_wins == 0
    assert rep.overall_winner == "A"
    assert all(tc.significant for tc in rep.per_tag)

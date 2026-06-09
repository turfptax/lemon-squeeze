"""Router recommendation logic."""
from sqlalchemy import select

from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run, get_session
from lemon_squeeze.router import recommend


def _make_history(model_name: str, size_b: float, prompt_text: str, tag: str, passes: list[bool]):
    """Insert a Prompt + tag + N Run+Eval pairs for a given model."""
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
            model = Model(name=model_name, provider="test", size_params_b=size_b, local=True)
            s.add(model); s.flush()

        for ok in passes:
            run = Run(prompt_id=prompt.id, model_id=model.id, response="x")
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


def test_recommend_picks_smallest_qualifying_model():
    # 3B model and 70B model both at 100% on coding; should pick 3B.
    _make_history("local/tiny-3b", 3.0, "Write a python function that adds two numbers.", "coding", [True, True, True])
    _make_history("local/big-70b", 70.0, "Write a python function that adds two numbers.", "coding", [True, True, True])
    rec = recommend("Write a python function to multiply two numbers.")
    assert rec.picked is not None
    assert rec.picked.model_name == "local/tiny-3b"
    assert rec.fallback is False


def test_recommend_falls_back_when_nobody_meets_threshold():
    _make_history("local/poor-7b", 7.0, "Compute the integral of x squared.", "math", [False, False, True])
    rec = recommend("Compute the derivative of x cubed.", threshold=0.9)
    assert rec.picked is not None
    assert rec.fallback is True


def test_recommend_returns_none_when_no_tags():
    rec = recommend("zzzzzzz")  # heuristic returns just 'unknown' which is filtered
    assert rec.picked is None
    assert "no historical" in rec.reason or rec.tags == []


def test_recommend_respects_min_samples():
    _make_history("local/one-shot-3b", 3.0, "Translate hello into French.", "translation", [True])
    rec = recommend("Translate goodbye into German.", min_samples=3, threshold=0.5)
    # Only 1 sample exists for this tag, below min_samples. We fall back since
    # the candidate exists but doesn't qualify.
    assert rec.picked is not None
    assert rec.fallback is True


def test_recommend_uses_ensemble_not_just_heuristic(monkeypatch):
    """The router must classify incoming prompts with the full ensemble
    (heuristic + trained ML + optional LLM), not HeuristicClassifier alone.
    Found live: a trained ML model correctly tagged a syllogism prompt as
    "reasoning" via `lemon classify ask`, but `lemon route pick` returned
    Tags: (none) for the same prompt because recommend() bypassed the
    ensemble entirely -- severing the accumulate-labels -> train-ml ->
    smarter-router feedback loop at the last link."""
    from lemon_squeeze.classification.base import Classifier, TagPrediction

    class FakeEnsemble(Classifier):
        name = "fake"

        def predict(self, prompt: str) -> list[TagPrediction]:
            return [
                TagPrediction(tag="reasoning", confidence=0.7, classifier="ml"),
                TagPrediction(tag="unknown", confidence=0.1, classifier="heuristic"),
            ]

    monkeypatch.setattr(
        "lemon_squeeze.router.build_default_classifier", lambda: FakeEnsemble()
    )
    _make_history(
        "local/logic-3b", 3.0, "Premise prompt about syllogisms.", "reasoning",
        [True, True, True],
    )
    rec = recommend("If all bloops are razzies, are bloops lazzies?", threshold=0.5)
    assert rec.tags == ["reasoning"]  # "unknown" filtered, ML tag honored
    assert rec.picked is not None
    assert rec.picked.model_name == "local/logic-3b"

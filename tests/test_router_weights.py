"""Weighted multi-criteria routing."""
from sqlalchemy import select

from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run, get_session
from lemon_squeeze.router import (
    BALANCED,
    CHEAP,
    FAST,
    PRESETS,
    SIZE_ONLY,
    RouterWeights,
    recommend,
    _normalize_lower_is_better,
)


def _seed_history(
    model_name: str,
    *,
    size_b: float,
    cost: float,
    latency_ms: int,
    pass_rates: list[bool],
    tag: str = "coding",
    prompt_text: str | None = None,
) -> None:
    """Insert a Model with N (Run, Evaluation) pairs giving the desired pass rate.

    Each call appends to history. `prompt_text` defaults to "Write code: <model_name>"
    so calls land on distinct prompts (and thus the per-prompt dedup doesn't fire).
    """
    text = prompt_text or f"Write code for {model_name}"
    with get_session() as s:
        prompt = Prompt(
            content=text,
            content_hash=f"h-{text}-{model_name}",
            char_count=len(text),
            source="test",
        )
        s.add(prompt); s.flush()
        s.add(PromptTag(prompt_id=prompt.id, tag=tag, classifier="heuristic", confidence=0.9))

        model = s.scalar(select(Model).where(Model.name == model_name))
        if model is None:
            model = Model(
                name=model_name,
                provider="test",
                size_params_b=size_b,
                local=True,
            )
            s.add(model); s.flush()

        for ok in pass_rates:
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


# ---------- normalization ----------------------------------------------------


def test_normalize_lower_is_better_basic():
    assert _normalize_lower_is_better([1.0, 5.0, 10.0]) == [1.0, (10 - 5) / 9, 0.0]


def test_normalize_handles_constant_values():
    assert _normalize_lower_is_better([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]


def test_normalize_handles_empty():
    assert _normalize_lower_is_better([]) == []


# ---------- preset coverage --------------------------------------------------


def test_presets_normalize_to_one():
    for name, w in PRESETS.items():
        n = w.normalize()
        assert abs(n.size + n.cost + n.latency - 1.0) < 1e-9, name


def test_weights_zero_falls_back_to_size():
    n = RouterWeights(size=0, cost=0, latency=0).normalize()
    assert (n.size, n.cost, n.latency) == (1.0, 0.0, 0.0)


# ---------- preset semantics over real candidates ----------------------------


def test_size_preset_picks_smallest():
    _seed_history("small-3b", size_b=3.0, cost=0.005, latency_ms=1000, pass_rates=[True] * 3)
    _seed_history("big-70b", size_b=70.0, cost=0.001, latency_ms=200, pass_rates=[True] * 3)
    rec = recommend("Write a Python function.", weights="size")
    assert rec.picked is not None
    assert rec.picked.model_name == "small-3b"


def test_cheap_preset_picks_cheapest():
    _seed_history("small-3b", size_b=3.0, cost=0.005, latency_ms=1000, pass_rates=[True] * 3)
    _seed_history("big-70b", size_b=70.0, cost=0.001, latency_ms=200, pass_rates=[True] * 3)
    rec = recommend("Write a Python function.", weights="cheap")
    assert rec.picked is not None
    assert rec.picked.model_name == "big-70b"  # cheaper despite being bigger


def test_fast_preset_picks_fastest():
    _seed_history("slow-3b", size_b=3.0, cost=0.001, latency_ms=2000, pass_rates=[True] * 3)
    _seed_history("fast-7b", size_b=7.0, cost=0.002, latency_ms=300, pass_rates=[True] * 3)
    rec = recommend("Write a Python function.", weights="fast")
    assert rec.picked is not None
    assert rec.picked.model_name == "fast-7b"


def test_custom_weights_compose():
    # small but slow vs medium but fast — balanced should prefer medium
    _seed_history("small-slow", size_b=3.0, cost=0.005, latency_ms=3000, pass_rates=[True] * 3)
    _seed_history("medium-fast", size_b=8.0, cost=0.003, latency_ms=200, pass_rates=[True] * 3)
    rec = recommend(
        "Write a Python function.",
        weights=RouterWeights(size=0.3, cost=0.0, latency=0.7),
    )
    assert rec.picked is not None
    assert rec.picked.model_name == "medium-fast"


def test_unknown_preset_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown preset"):
        recommend("hi", weights="ridiculous")


# ---------- composite score is populated -------------------------------------


def test_composite_score_populated_on_qualifying_candidates():
    _seed_history("a", size_b=3.0, cost=0.001, latency_ms=100, pass_rates=[True] * 3)
    _seed_history("b", size_b=70.0, cost=0.01, latency_ms=2000, pass_rates=[True] * 3)
    rec = recommend("Write a Python function.", weights=BALANCED)
    assert rec.picked is not None
    assert rec.picked.composite_score is not None
    # Among qualifying candidates someone scored — at least one is 1.0.
    scores = [c.composite_score for c in rec.candidates if c.composite_score is not None]
    assert scores, "no candidates got a composite score"

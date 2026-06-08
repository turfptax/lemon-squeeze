"""aggregations.py — the unified per-(bucket, model) aggregation layer."""
from sqlalchemy import select

from lemon_squeeze.aggregations import (
    Aggregate,
    aggregate_by_intended_tag_model,
    aggregate_by_tag_model,
    group_by_first_key,
)
from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run, get_session


def _seed(
    *,
    model_name: str,
    tag: str | None,
    intended_tag: str | None,
    prompt_text: str,
    passes: list[bool],
    cost: float | None = None,
    latency_ms: int | None = None,
) -> None:
    with get_session() as s:
        prompt = s.scalar(select(Prompt).where(Prompt.content == prompt_text))
        if prompt is None:
            meta = {"intended_tag": intended_tag} if intended_tag else None
            prompt = Prompt(
                content=prompt_text,
                content_hash=f"h-{prompt_text}",
                char_count=len(prompt_text),
                source="test",
                source_metadata=meta,
            )
            s.add(prompt); s.flush()
            if tag:
                s.add(PromptTag(prompt_id=prompt.id, tag=tag, classifier="test", confidence=1.0))
        model = s.scalar(select(Model).where(Model.name == model_name)) or Model(
            name=model_name, provider="test", local=True, size_params_b=3.0, context_window=4096
        )
        if model.id is None:
            s.add(model); s.flush()
        for ok in passes:
            run = Run(prompt_id=prompt.id, model_id=model.id, response="x",
                      cost_usd=cost, latency_ms=latency_ms)
            s.add(run); s.flush()
            s.add(
                Evaluation(
                    run_id=run.id, rubric="human_pass",
                    score=1.0 if ok else 0.0, passed=ok, scored_by="human",
                )
            )


# ---------- aggregate_by_tag_model ------------------------------------------


def test_pass_rate_property_uses_passed_known_denominator():
    a = Aggregate(
        key=("x", "m"), n_evals=10, n_passed=3, n_passed_known=5,
        avg_score=0.5, avg_cost_usd=None, avg_latency_ms=None,
    )
    assert a.pass_rate == 0.6  # 3 / 5, not 3 / 10


def test_cost_per_pass_none_when_zero_pass():
    a = Aggregate(
        key=("x", "m"), n_evals=5, n_passed=0, n_passed_known=5,
        avg_score=0.0, avg_cost_usd=0.01, avg_latency_ms=None,
    )
    assert a.cost_per_pass is None


def test_cost_per_pass_when_known():
    a = Aggregate(
        key=("x", "m"), n_evals=10, n_passed=5, n_passed_known=10,
        avg_score=0.5, avg_cost_usd=0.01, avg_latency_ms=None,
    )
    assert abs(a.cost_per_pass - 0.02) < 1e-9


def test_aggregate_by_tag_model_groups_correctly():
    _seed(model_name="alpha", tag="coding", intended_tag=None,
          prompt_text="code 1", passes=[True, True, True], cost=0.01, latency_ms=100)
    _seed(model_name="beta", tag="coding", intended_tag=None,
          prompt_text="code 1", passes=[True, False], cost=0.02, latency_ms=200)
    _seed(model_name="alpha", tag="math", intended_tag=None,
          prompt_text="math 1", passes=[True], cost=0.005, latency_ms=50)

    aggs = aggregate_by_tag_model(rubrics=["human_pass"])
    by_key = {(a.tag, a.model_name): a for a in aggs}

    assert by_key[("coding", "alpha")].pass_rate == 1.0
    assert by_key[("coding", "alpha")].n_evals == 3
    assert by_key[("coding", "beta")].pass_rate == 0.5
    assert by_key[("math", "alpha")].pass_rate == 1.0


def test_aggregate_by_tag_model_filters_by_tag():
    _seed(model_name="alpha", tag="coding", intended_tag=None,
          prompt_text="code 1", passes=[True])
    _seed(model_name="alpha", tag="math", intended_tag=None,
          prompt_text="math 1", passes=[True])

    aggs = aggregate_by_tag_model(rubrics=["human_pass"], tags=["math"])
    assert all(a.tag == "math" for a in aggs)
    assert len(aggs) == 1


def test_aggregate_by_tag_model_filters_by_model_names():
    _seed(model_name="alpha", tag="coding", intended_tag=None,
          prompt_text="p1", passes=[True])
    _seed(model_name="beta", tag="coding", intended_tag=None,
          prompt_text="p1", passes=[True])

    aggs = aggregate_by_tag_model(rubrics=["human_pass"], model_names=["beta"])
    assert all(a.model_name == "beta" for a in aggs)


def test_aggregate_by_tag_model_empty_inputs():
    assert aggregate_by_tag_model(rubrics=[]) == []
    assert aggregate_by_tag_model(rubrics=["human_pass"], tags=[]) == []
    assert aggregate_by_tag_model(rubrics=["human_pass"], model_names=[]) == []
    assert aggregate_by_tag_model(rubrics=["human_pass"], prompt_ids=[]) == []


def test_aggregate_by_tag_model_dedupes_multi_classifier_tags():
    """Multiple classifiers can tag the same prompt with the same tag.
    The aggregator must NOT double-count: each prompt contributes once per tag.

    Regression for a bug introduced when the join inflated counts by the number
    of PromptTag rows per (prompt_id, tag).
    """
    # Seed a prompt with 3 runs (all pass). Then add two extra PromptTag rows
    # (different classifiers) — should not change n_evals.
    _seed(
        model_name="m", tag="coding", intended_tag=None,
        prompt_text="reverse a string", passes=[True, True, True],
    )
    with get_session() as s:
        prompt = s.scalar(select(Prompt).where(Prompt.content == "reverse a string"))
        s.add(PromptTag(prompt_id=prompt.id, tag="coding", classifier="ml", confidence=0.9))
        s.add(PromptTag(prompt_id=prompt.id, tag="coding", classifier="llm", confidence=0.95))

    aggs = aggregate_by_tag_model(rubrics=["human_pass"], tags=["coding"])
    coding = next(a for a in aggs if a.tag == "coding" and a.model_name == "m")
    assert coding.n_evals == 3  # NOT 9 (3 runs * 3 classifier tags)
    assert coding.pass_rate == 1.0


# ---------- aggregate_by_intended_tag_model ----------------------------------


def test_aggregate_by_intended_tag_model():
    _seed(model_name="m", tag=None, intended_tag="math",
          prompt_text="2+2?", passes=[True], cost=0.01, latency_ms=50)
    _seed(model_name="m", tag=None, intended_tag="math",
          prompt_text="3+3?", passes=[False], cost=0.01, latency_ms=50)
    _seed(model_name="m", tag=None, intended_tag="coding",
          prompt_text="def x", passes=[True, True], cost=0.02, latency_ms=100)

    with get_session() as s:
        prompt_ids = [p.id for p in s.scalars(select(Prompt)).all()]
    aggs = aggregate_by_intended_tag_model(rubrics=["human_pass"], prompt_ids=prompt_ids)
    by_key = {a.key: a for a in aggs}

    assert by_key[("math", "m")].pass_rate == 0.5
    assert by_key[("math", "m")].n_evals == 2
    assert by_key[("coding", "m")].pass_rate == 1.0


def test_aggregate_by_intended_tag_falls_back_to_uncategorized():
    _seed(model_name="m", tag=None, intended_tag=None,
          prompt_text="orphan", passes=[True])

    with get_session() as s:
        prompt_ids = [p.id for p in s.scalars(select(Prompt)).all()]
    aggs = aggregate_by_intended_tag_model(rubrics=["human_pass"], prompt_ids=prompt_ids)
    assert any(a.tag == "uncategorized" for a in aggs)


# ---------- group_by_first_key ----------------------------------------------


def test_group_by_first_key():
    a1 = Aggregate(key=("coding", "m1"), n_evals=1, n_passed=1, n_passed_known=1,
                   avg_score=1.0, avg_cost_usd=None, avg_latency_ms=None)
    a2 = Aggregate(key=("coding", "m2"), n_evals=1, n_passed=0, n_passed_known=1,
                   avg_score=0.0, avg_cost_usd=None, avg_latency_ms=None)
    a3 = Aggregate(key=("math", "m1"), n_evals=2, n_passed=1, n_passed_known=2,
                   avg_score=0.5, avg_cost_usd=None, avg_latency_ms=None)
    grouped = group_by_first_key([a1, a2, a3])
    assert set(grouped) == {"coding", "math"}
    assert len(grouped["coding"]) == 2
    assert len(grouped["math"]) == 1

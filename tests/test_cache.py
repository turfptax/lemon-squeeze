"""Cache layer + invalidation behavior."""
from sqlalchemy import select

from lemon_squeeze.aggregations import aggregate_by_tag_model
from lemon_squeeze.cache import (
    _MISS,
    _LRU,
    aggregations_cache,
    aggregations_key,
    cache_stats,
)
from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run, get_session


def _seed_one_eval(passing: bool = True) -> None:
    with get_session() as s:
        p = Prompt(content="x", content_hash=f"ph-x-{passing}-cache", char_count=1, source="test")
        s.add(p); s.flush()
        s.add(PromptTag(prompt_id=p.id, tag="coding", classifier="test", confidence=1.0))
        m = s.scalar(select(Model).where(Model.name == "cache-m")) or Model(
            name="cache-m", provider="test", local=True
        )
        if m.id is None:
            s.add(m); s.flush()
        run = Run(prompt_id=p.id, model_id=m.id, response="r")
        s.add(run); s.flush()
        s.add(
            Evaluation(
                run_id=run.id, rubric="human_pass",
                score=1.0 if passing else 0.0, passed=passing, scored_by="human",
            )
        )


# ---------- _LRU unit tests --------------------------------------------------


def test_lru_get_miss_returns_sentinel():
    lru = _LRU(max_entries=4)
    assert lru.get(("k",)) is _MISS


def test_lru_put_then_get_hits():
    lru = _LRU(max_entries=4)
    lru.put(("k",), 42)
    assert lru.get(("k",)) == 42
    assert lru.stats().hits == 1


def test_lru_evicts_oldest_when_full():
    lru = _LRU(max_entries=2)
    lru.put(("a",), 1)
    lru.put(("b",), 2)
    lru.put(("c",), 3)
    assert lru.get(("a",)) is _MISS
    assert lru.get(("b",)) == 2
    assert lru.stats().entries_evicted == 1


def test_lru_bump_version_invalidates():
    lru = _LRU(max_entries=4)
    lru.put(("k",), "v")
    lru.bump_version()
    assert lru.get(("k",)) is _MISS
    assert lru.stats().invalidations == 1


def test_lru_ttl_expires():
    import time
    lru = _LRU(max_entries=4)
    lru.put(("k",), "v", ttl_seconds=0.05)
    time.sleep(0.1)
    assert lru.get(("k",)) is _MISS


# ---------- aggregations_cache integration ----------------------------------


def test_aggregate_by_tag_model_caches_repeat_calls():
    _seed_one_eval(passing=True)
    cache = aggregations_cache()
    stats_before = cache.stats()

    a1 = aggregate_by_tag_model(rubrics=["human_pass"], tags=["coding"])
    a2 = aggregate_by_tag_model(rubrics=["human_pass"], tags=["coding"])
    assert a1 == a2

    stats_after = cache.stats()
    assert stats_after.hits > stats_before.hits
    assert stats_after.misses > stats_before.misses  # the first call missed


def test_write_invalidates_aggregation_cache():
    _seed_one_eval(passing=True)
    cache = aggregations_cache()

    first = aggregate_by_tag_model(rubrics=["human_pass"], tags=["coding"])
    assert len(first) == 1
    assert first[0].n_evals == 1

    # Insert another eval — should invalidate the cached aggregate.
    _seed_one_eval(passing=False)
    second = aggregate_by_tag_model(rubrics=["human_pass"], tags=["coding"])
    assert second[0].n_evals == 2  # would still be 1 without invalidation


def test_aggregations_key_is_stable_under_iteration_order():
    """Same args via different orderings should produce the same key."""
    k1 = aggregations_key(
        fn="x", rubrics=["a", "b"], tags=["t1", "t2"],
        model_names=["m1", "m2"], prompt_ids=[3, 1, 2],
    )
    k2 = aggregations_key(
        fn="x", rubrics=["b", "a"], tags=["t2", "t1"],
        model_names=["m2", "m1"], prompt_ids=[2, 1, 3],
    )
    assert k1 == k2


def test_cache_stats_exposes_named_caches():
    s = cache_stats()
    assert "aggregations" in s

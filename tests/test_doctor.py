"""Doctor diagnostic checks."""
from lemon_squeeze.db import Model, Prompt, PromptTag, get_session
from lemon_squeeze.doctor import run_all_checks, summarize


def _names(results):
    return {r.name: r for r in results}


def test_doctor_runs_all_checks_on_empty_db():
    results = run_all_checks()
    assert len(results) > 5
    names = _names(results)
    # The schema check should pass — conftest runs init_db.
    assert names["schema"].status == "ok"
    assert names["taxonomy"].status == "ok"
    # Empty DB → these warn, not fail.
    assert names["prompts"].status == "warn"
    assert names["models"].status == "warn"
    assert names["evaluations"].status == "warn"
    ok, warn, fail = summarize(results)
    assert ok >= 2  # schema + taxonomy
    assert warn > 0
    assert fail == 0


def test_doctor_promotes_warn_to_ok_once_data_lands():
    with get_session() as s:
        p = Prompt(content="hi", content_hash="hi-h", char_count=2, source="test")
        s.add(p); s.flush()
        s.add(PromptTag(prompt_id=p.id, tag="coding", classifier="heuristic", confidence=0.9))
        s.add(Model(name="some/model", provider="test", local=True))

    results = run_all_checks()
    names = _names(results)
    assert names["prompts"].status == "ok"
    assert names["models"].status == "ok"
    assert names["classification"].status == "ok"


def test_doctor_partial_classification_coverage_warns():
    with get_session() as s:
        for i in range(10):
            p = Prompt(content=f"p{i}", content_hash=f"p{i}-h", char_count=2, source="test")
            s.add(p)
        s.flush()
        # Tag only the first 2 — under the 50% threshold.
        s.add(PromptTag(prompt_id=1, tag="coding", classifier="heuristic", confidence=0.9))
        s.add(PromptTag(prompt_id=2, tag="coding", classifier="heuristic", confidence=0.9))

    results = run_all_checks()
    cls = _names(results)["classification"]
    assert cls.status == "warn"
    assert "20%" in cls.detail

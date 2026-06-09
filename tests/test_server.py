"""HTTP API tests via FastAPI TestClient.

Skipped automatically if fastapi isn't installed (it's an optional extra).
"""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sqlalchemy import select  # noqa: E402

from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run, get_session  # noqa: E402
from lemon_squeeze.server import create_app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _seed_routing_data() -> None:
    """A small coded prompt + sonnet model with passing evals — enough that
    /route returns a real recommendation."""
    with get_session() as s:
        p = Prompt(content="Write a Python add function.", content_hash="ph-add",
                   char_count=20, source="test")
        s.add(p); s.flush()
        s.add(PromptTag(prompt_id=p.id, tag="coding", classifier="heuristic", confidence=0.95))

        m = Model(name="anthropic/sonnet", provider="anthropic", family="claude",
                  size_params_b=70.0, context_window=200000, local=False,
                  cost_in_per_mtok=3.0, cost_out_per_mtok=15.0)
        s.add(m); s.flush()
        for _ in range(5):
            run = Run(prompt_id=p.id, model_id=m.id, response="def add(a, b): return a+b",
                     cost_usd=0.01, latency_ms=500)
            s.add(run); s.flush()
            s.add(Evaluation(run_id=run.id, rubric="human_pass",
                             score=1.0, passed=True, scored_by="human"))


# ---------- /healthz --------------------------------------------------------


def test_healthz_returns_ok(client: TestClient):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ---------- /models ---------------------------------------------------------


def test_models_lists_registered(client: TestClient):
    with get_session() as s:
        s.add(Model(name="t/m", provider="t", local=True, size_params_b=3.0))
    r = client.get("/models")
    assert r.status_code == 200
    names = [m["name"] for m in r.json()["models"]]
    assert "t/m" in names


def test_models_empty_when_nothing_registered(client: TestClient):
    r = client.get("/models")
    assert r.json()["models"] == []


# ---------- /route ----------------------------------------------------------


def test_route_returns_recommendation(client: TestClient):
    _seed_routing_data()
    r = client.post("/route", json={
        "prompt": "Write a Python function that returns 1.",
        "threshold": 0.5,
        "min_samples": 1,
        "preset": "balanced",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["picked"] is not None
    assert body["picked"]["model_name"] == "anthropic/sonnet"
    assert body["tags"] == ["coding"]
    assert "weights" in body and body["weights"]["size"] > 0


def test_route_with_explicit_weights(client: TestClient):
    _seed_routing_data()
    r = client.post("/route", json={
        "prompt": "Write a Python class.",
        "threshold": 0.5,
        "min_samples": 1,
        "weights": {"size": 0.1, "cost": 0.9, "latency": 0.0},
    })
    assert r.status_code == 200
    weights = r.json()["weights"]
    # Normalized — the values sum to 1.0 but the cost weight dominates.
    assert weights["cost"] > weights["size"]


def test_route_rejects_unknown_preset(client: TestClient):
    r = client.post("/route", json={"prompt": "x", "preset": "ridiculous"})
    assert r.status_code == 400
    assert "ridiculous" in r.json()["detail"]


def test_route_returns_no_pick_when_no_data(client: TestClient):
    """With nothing in the DB, the router should classify the prompt (tags may
    or may not fire depending on heuristic signal) but always returns picked=None
    with a specific 'no historical evaluations' reason."""
    r = client.post("/route", json={"prompt": "xx", "min_samples": 1})
    body = r.json()
    assert body["picked"] is None
    assert "no historical" in body["reason"]


# ---------- /classify -------------------------------------------------------


def test_classify_returns_predictions(client: TestClient):
    r = client.post("/classify", json={"prompt": "Write a Python function that does X."})
    body = r.json()
    tags = [p["tag"] for p in body["predictions"]]
    assert "coding" in tags


# ---------- /report ---------------------------------------------------------


def test_report_returns_schema_version(client: TestClient):
    r = client.get("/report")
    body = r.json()
    assert body["schema_version"] == 1
    assert "headline" in body
    assert "scorecards" in body


def test_report_honors_query_params(client: TestClient):
    """Caught against real LM Studio: GET /report took ZERO query
    parameters and always used build_report defaults (rubric=human_pass,
    min_samples=3, threshold=0.7). Anyone running a real bench
    (rubric=bench:expected_contains) saw scorecards=[] and gaps full of
    'no_evals' because the endpoint silently ignored their actual rubric.

    Mirror the CLI report's three flags: threshold, min_samples, rubric.
    """
    # Seed enough data that the per-tag scorecard becomes non-empty WHEN
    # the right rubric is queried.
    _seed_routing_data()

    # Default request -> uses human_pass, which the seed populates. Should
    # produce a coding scorecard at the default threshold/min_samples.
    r = client.get("/report")
    assert r.status_code == 200
    body = r.json()
    assert len(body["scorecards"]) >= 1
    assert any(sc["tag"] == "coding" for sc in body["scorecards"])

    # Query for a rubric that doesn't exist -> no scorecards (everything
    # falls into gaps). This confirms the param is actually being honored
    # rather than silently ignored.
    r = client.get("/report", params={"rubric": "nonexistent-rubric"})
    assert r.status_code == 200
    body = r.json()
    assert body["scorecards"] == [], (
        "report should honor ?rubric param; falling back to defaults means "
        "anyone using a non-human_pass rubric gets a broken report"
    )
    # The coding tag should appear in gaps now since no evals match the
    # bogus rubric.
    assert any(g["tag"] == "coding" for g in body["gaps"])


# ---------- /compare --------------------------------------------------------


def test_compare_rejects_unknown_models(client: TestClient):
    r = client.post("/compare", json={"model_a": "ghost", "model_b": "phantom"})
    assert r.status_code == 400
    assert "unknown model" in r.json()["detail"]


def test_compare_with_real_data(client: TestClient):
    _seed_routing_data()
    # Add a second model so we have something to compare against.
    with get_session() as s:
        p = s.scalar(select(Prompt).where(Prompt.content_hash == "ph-add"))
        m2 = Model(name="local/tiny", provider="lm_studio", local=True, size_params_b=3.0)
        s.add(m2); s.flush()
        for ok in [False, True, False]:
            run = Run(prompt_id=p.id, model_id=m2.id, response="...",
                     cost_usd=0.001, latency_ms=100)
            s.add(run); s.flush()
            s.add(Evaluation(run_id=run.id, rubric="human_pass",
                             score=1.0 if ok else 0.0, passed=ok, scored_by="human"))
    r = client.post("/compare", json={
        "model_a": "anthropic/sonnet",
        "model_b": "local/tiny",
        "rubric": "human_pass",
        "require_significance": False,  # small samples
    })
    assert r.status_code == 200
    body = r.json()
    assert body["model_a"] == "anthropic/sonnet"
    assert "per_tag" in body
    # Per-tag entries must expose avg_score for both models. TagComparison
    # has these fields, but _comparison_to_dict was dropping them — HTTP
    # clients could see pass rates but not average scores, which the CLI's
    # rich-rendered table does include.
    assert body["per_tag"], "expected at least one per-tag entry"
    for tc in body["per_tag"]:
        assert "a_avg_score" in tc, f"a_avg_score missing from per_tag entry: {tc}"
        assert "b_avg_score" in tc, f"b_avg_score missing from per_tag entry: {tc}"

"""/metrics endpoint behavior."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from lemon_squeeze.db import Model, Prompt, get_session  # noqa: E402
from lemon_squeeze.server import create_app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_metrics_returns_db_counts(client: TestClient):
    with get_session() as s:
        s.add(Prompt(content="p", content_hash="h-metric", char_count=1, source="t"))
        s.add(Model(name="metric/m", provider="t", local=True))
    body = client.get("/metrics").json()
    assert body["db"]["prompts"] >= 1
    assert body["db"]["models"] >= 1
    assert "runs" in body["db"]
    assert "evaluations" in body["db"]


def test_metrics_includes_request_counts(client: TestClient):
    # Hit some endpoints to bump the per-path counters.
    client.get("/healthz")
    client.get("/healthz")
    client.get("/models")
    body = client.get("/metrics").json()
    counts = body["requests_by_path"]
    assert counts.get("/healthz", 0) >= 2
    assert counts.get("/models", 0) >= 1


def test_metrics_exposes_aggregation_cache_stats(client: TestClient):
    body = client.get("/metrics").json()
    aggs = body["caches"]["aggregations"]
    for key in ("hits", "misses", "invalidations", "size", "entries_evicted"):
        assert key in aggs

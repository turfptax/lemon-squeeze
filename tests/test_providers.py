"""Provider discovery — mock httpx, verify parse logic."""
from unittest.mock import MagicMock, patch

import httpx
import pytest

from lemon_squeeze.providers import (
    DiscoveredModel,
    list_lm_studio_models,
    list_openrouter_models,
)


# ---------- LM Studio --------------------------------------------------------


def _mock_response(payload, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=resp
        )
    return resp


def test_lm_studio_parses_models():
    payload = {
        "data": [
            {"id": "llama-3.1-8b-instruct", "object": "model"},
            {"id": "mistral-7b-instruct", "object": "model"},
        ]
    }
    with patch("lemon_squeeze.providers.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.get.return_value = _mock_response(payload)
        models = list_lm_studio_models()
    assert len(models) == 2
    assert models[0].name == "llama-3.1-8b-instruct"
    assert models[0].provider == "lm_studio"
    assert models[0].family == "llama"
    assert models[1].family == "mistral"


def test_lm_studio_handles_bare_list_payload():
    payload = [{"id": "gemma-2-2b", "object": "model"}]
    with patch("lemon_squeeze.providers.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.get.return_value = _mock_response(payload)
        models = list_lm_studio_models()
    assert len(models) == 1
    assert models[0].name == "gemma-2-2b"


def test_lm_studio_raises_on_connection_failure():
    with patch("lemon_squeeze.providers.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError("nope")
        with pytest.raises(httpx.ConnectError):
            list_lm_studio_models()


def test_lm_studio_skips_malformed_entries():
    payload = {
        "data": [
            {"id": "good"},
            "bare-string",
            {"no_id": True},
            {"id": 42},  # non-string id
        ]
    }
    with patch("lemon_squeeze.providers.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.get.return_value = _mock_response(payload)
        models = list_lm_studio_models()
    assert [m.name for m in models] == ["good"]


# ---------- OpenRouter -------------------------------------------------------


def test_openrouter_converts_per_token_pricing_to_per_mtok():
    payload = {
        "data": [
            {
                "id": "anthropic/claude-sonnet-4-6",
                "context_length": 200000,
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            }
        ]
    }
    with patch("lemon_squeeze.providers.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.get.return_value = _mock_response(payload)
        models = list_openrouter_models()
    assert len(models) == 1
    m = models[0]
    assert m.name == "anthropic/claude-sonnet-4-6"
    assert m.context_window == 200000
    assert abs(m.cost_in_per_mtok - 3.0) < 1e-9
    assert abs(m.cost_out_per_mtok - 15.0) < 1e-9
    assert m.family == "claude"


def test_openrouter_handles_missing_pricing():
    payload = {"data": [{"id": "test/model"}]}
    with patch("lemon_squeeze.providers.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.get.return_value = _mock_response(payload)
        models = list_openrouter_models()
    assert models[0].cost_in_per_mtok is None
    assert models[0].cost_out_per_mtok is None


def test_openrouter_handles_non_numeric_pricing():
    payload = {
        "data": [
            {"id": "wonky/model", "pricing": {"prompt": "free!", "completion": None}}
        ]
    }
    with patch("lemon_squeeze.providers.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.get.return_value = _mock_response(payload)
        models = list_openrouter_models()
    assert models[0].cost_in_per_mtok is None


def test_openrouter_handles_pricing_as_non_dict():
    """A pricing field that's a string or list must not crash the call.
    Regression for AttributeError-on-everyone from one bad upstream record."""
    payload = {
        "data": [
            {"id": "weird/model", "pricing": "free"},
            {"id": "other/model", "pricing": ["unexpected", "list"]},
            {"id": "good/model", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
        ]
    }
    with patch("lemon_squeeze.providers.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.get.return_value = _mock_response(payload)
        models = list_openrouter_models()
    assert len(models) == 3
    assert models[0].cost_in_per_mtok is None
    assert models[1].cost_in_per_mtok is None
    assert models[2].cost_in_per_mtok == 1.0


def test_openrouter_accepts_string_or_float_context_length():
    payload = {
        "data": [
            {"id": "a/m", "context_length": "128000"},
            {"id": "b/m", "context_length": 128000.0},
            {"id": "c/m", "context_length": True},  # bool — must be rejected
            {"id": "d/m", "context_length": "not-a-number"},
        ]
    }
    with patch("lemon_squeeze.providers.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.get.return_value = _mock_response(payload)
        models = list_openrouter_models()
    by_name = {m.name: m for m in models}
    assert by_name["a/m"].context_window == 128000
    assert by_name["b/m"].context_window == 128000
    assert by_name["c/m"].context_window is None
    assert by_name["d/m"].context_window is None


# ---------- DiscoveredModel shape -------------------------------------------


def test_discovered_model_is_dataclass():
    m = DiscoveredModel(provider="lm_studio", name="x")
    assert m.provider == "lm_studio"
    assert m.family is None
    assert m.context_window is None


# ---------- size-from-name heuristic ----------------------------------------


def test_guess_size_b_parses_billions_and_millions():
    from lemon_squeeze.providers import _guess_size_b_from_name

    # Billions: real model IDs LM Studio surfaces.
    assert _guess_size_b_from_name("qwen3.5-2b") == 2.0
    assert _guess_size_b_from_name("qwen3.5-0.8b") == 0.8
    assert _guess_size_b_from_name("google/gemma-3-4b") == 4.0
    assert _guess_size_b_from_name("google/gemma-4-26b-a4b") == 26.0  # 26b, not 4b
    assert _guess_size_b_from_name("qwen/qwen3.5-35b-a3b") == 35.0
    # Millions get divided by 1000.
    assert _guess_size_b_from_name("smollm2-135m-instruct") == 0.135
    # No size token in the name → None.
    assert _guess_size_b_from_name("text-embedding-nomic-embed-text-v1.5") is None
    assert _guess_size_b_from_name("randomthing") is None


def test_lm_studio_discovery_fills_size_b(monkeypatch):
    """End-to-end: list_lm_studio_models populates size_params_b from the
    model ID. Catches both the heuristic and its plumb-through to
    DiscoveredModel."""
    from lemon_squeeze.providers import DiscoveredModel, list_lm_studio_models

    class FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"data": [
                {"id": "qwen3.5-2b"},
                {"id": "smollm2-135m-instruct"},
                {"id": "text-embedding-nomic-embed-text-v1.5"},
            ]}

    class FakeClient:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, *a, **kw): return FakeResp()

    monkeypatch.setattr("lemon_squeeze.providers.httpx.Client", FakeClient)
    models = list_lm_studio_models()
    by_name = {m.name: m for m in models}
    assert by_name["qwen3.5-2b"].size_params_b == 2.0
    assert by_name["smollm2-135m-instruct"].size_params_b == 0.135
    assert by_name["text-embedding-nomic-embed-text-v1.5"].size_params_b is None

"""ChatClient — mocked-httpx tests for the OpenAI-compatible client."""
from unittest.mock import MagicMock, patch

import httpx
import pytest

from lemon_squeeze.eval.clients import ChatClient, ChatResult, _estimate_cost


# ---------- _estimate_cost --------------------------------------------------


def test_estimate_cost_returns_none_when_token_counts_missing():
    assert _estimate_cost(None, 100, 1.0, 2.0) is None
    assert _estimate_cost(100, None, 1.0, 2.0) is None


def test_estimate_cost_returns_none_when_no_pricing():
    assert _estimate_cost(100, 50, None, None) is None


def test_estimate_cost_input_only():
    # 1000 in @ $2/Mtok = $0.002
    assert _estimate_cost(1000, 0, 2.0, None) == pytest.approx(0.002)


def test_estimate_cost_input_and_output():
    # 1M in @ $1 + 1M out @ $2 = $3
    assert _estimate_cost(1_000_000, 1_000_000, 1.0, 2.0) == pytest.approx(3.0)


# ---------- Construction ----------------------------------------------------


def test_construction_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        ChatClient("openai")


def test_construction_lm_studio_picks_up_settings():
    c = ChatClient("lm_studio")
    assert "v1" in c.base_url


def test_construction_openrouter_picks_up_settings():
    c = ChatClient("openrouter")
    assert "openrouter.ai" in c.base_url


def test_construction_explicit_overrides_win():
    c = ChatClient("lm_studio", base_url="http://custom:9000/v1", api_key="explicit-key")
    assert c.base_url == "http://custom:9000/v1"
    assert c.api_key == "explicit-key"


# ---------- chat() ----------------------------------------------------------


def _ok_response(text: str = "ok", tokens_in: int = 10, tokens_out: int = 5) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }
    resp.raise_for_status = MagicMock()
    return resp


def test_chat_returns_chat_result_with_usage():
    c = ChatClient("lm_studio")
    with patch("lemon_squeeze.eval.clients.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.return_value = _ok_response("hi", 7, 3)
        r = c.chat("local/m", "say hi")
    assert isinstance(r, ChatResult)
    assert r.text == "hi"
    assert r.tokens_in == 7
    assert r.tokens_out == 3
    assert r.latency_ms is not None
    assert r.latency_ms >= 0


def test_chat_includes_system_message_when_provided():
    c = ChatClient("lm_studio")
    captured: dict = {}
    def _capture_post(url, json=None, headers=None, **kw):
        captured["payload"] = json
        return _ok_response("done")
    with patch("lemon_squeeze.eval.clients.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.side_effect = _capture_post
        c.chat("m", "user", system="you are helpful")
    roles = [m["role"] for m in captured["payload"]["messages"]]
    assert roles == ["system", "user"]


def test_chat_passes_max_tokens_when_set():
    c = ChatClient("lm_studio")
    captured: dict = {}
    def _capture(url, json=None, headers=None, **kw):
        captured["payload"] = json
        return _ok_response()
    with patch("lemon_squeeze.eval.clients.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.side_effect = _capture
        c.chat("m", "p", max_tokens=128)
    assert captured["payload"]["max_tokens"] == 128


def test_chat_omits_max_tokens_when_unset():
    c = ChatClient("lm_studio")
    captured: dict = {}
    def _capture(url, json=None, headers=None, **kw):
        captured["payload"] = json
        return _ok_response()
    with patch("lemon_squeeze.eval.clients.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.side_effect = _capture
        c.chat("m", "p")
    assert "max_tokens" not in captured["payload"]


def test_chat_sends_bearer_auth_when_api_key_set():
    c = ChatClient("openrouter", api_key="sk-test")
    captured: dict = {}
    def _capture(url, json=None, headers=None, **kw):
        captured["headers"] = headers
        return _ok_response()
    with patch("lemon_squeeze.eval.clients.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.side_effect = _capture
        c.chat("anthropic/sonnet", "p")
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


def test_chat_omits_auth_header_when_no_key():
    c = ChatClient("openrouter", api_key="")
    captured: dict = {}
    def _capture(url, json=None, headers=None, **kw):
        captured["headers"] = headers
        return _ok_response()
    with patch("lemon_squeeze.eval.clients.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.side_effect = _capture
        c.chat("m", "p")
    assert "Authorization" not in captured["headers"]


def test_chat_computes_cost_when_pricing_provided():
    c = ChatClient("lm_studio")
    with patch("lemon_squeeze.eval.clients.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.return_value = _ok_response(
            "x", tokens_in=1_000_000, tokens_out=500_000,
        )
        r = c.chat("m", "p", cost_in_per_mtok=1.0, cost_out_per_mtok=2.0)
    # 1M @ $1 + 0.5M @ $2 = $2
    assert r.cost_usd == pytest.approx(2.0)


def test_chat_cost_is_none_when_pricing_absent():
    c = ChatClient("lm_studio")
    with patch("lemon_squeeze.eval.clients.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.return_value = _ok_response()
        r = c.chat("m", "p")
    assert r.cost_usd is None


def test_chat_handles_empty_content():
    """OpenAI servers occasionally return content=null for refusals."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": None}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 0},
    }
    resp.raise_for_status = MagicMock()
    c = ChatClient("lm_studio")
    with patch("lemon_squeeze.eval.clients.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.return_value = resp
        r = c.chat("m", "p")
    assert r.text == ""


def test_chat_propagates_http_errors():
    resp = MagicMock()
    resp.status_code = 500
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "server error", request=MagicMock(), response=resp
    )
    c = ChatClient("lm_studio")
    with patch("lemon_squeeze.eval.clients.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.return_value = resp
        with pytest.raises(httpx.HTTPError):
            c.chat("m", "p")

"""LLMClassifier — mocked tests for the LLM-based prompt classifier.

LLMClassifier uses raw httpx, not ChatClient — so we patch
`lemon_squeeze.classification.llm.httpx.Client` directly.
"""
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import httpx

from lemon_squeeze.classification.llm import LLMClassifier
from lemon_squeeze.db import TagTaxonomy, get_session


def _seed_taxonomy(*tags: str) -> None:
    from sqlalchemy import select
    with get_session() as s:
        existing = {t.tag for t in s.scalars(select(TagTaxonomy)).all()}
        for t in tags:
            if t not in existing:
                s.add(TagTaxonomy(tag=t, description=""))


def _httpx_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": text}}]}
    resp.raise_for_status = MagicMock()
    return resp


@contextmanager
def _patch_httpx(text: str):
    """Patch raw httpx.Client to return the given completion text."""
    with patch("lemon_squeeze.classification.llm.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.return_value = _httpx_response(text)
        yield


@contextmanager
def _patch_httpx_error(exc: Exception):
    with patch("lemon_squeeze.classification.llm.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.side_effect = exc
        yield


# ---------- provider=none short-circuit -------------------------------------


def test_provider_none_returns_empty_predictions():
    c = LLMClassifier(provider="none")
    assert c.predict("Write code") == []


# ---------- _parse path ----------------------------------------------------


def test_parse_well_formed_json_response():
    _seed_taxonomy("coding", "math")
    c = LLMClassifier(provider="lm_studio")
    with _patch_httpx(
        '{"tags": [{"tag":"coding","confidence":0.9},{"tag":"math","confidence":0.6}]}'
    ):
        preds = c.predict("Write a Python function to sort a list")
    tags = {p.tag for p in preds}
    assert "coding" in tags
    assert "math" in tags


def test_parse_filters_out_tags_not_in_taxonomy():
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with _patch_httpx(
        '{"tags": [{"tag":"coding","confidence":0.9},{"tag":"made_up","confidence":0.9}]}'
    ):
        preds = c.predict("Hi")
    assert {p.tag for p in preds} == {"coding"}


def test_parse_drops_predictions_below_confidence_floor():
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with _patch_httpx('{"tags": [{"tag":"coding","confidence":0.1}]}'):
        preds = c.predict("Hi")
    assert preds == []


def test_parse_clamps_confidence_to_unit_interval():
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with _patch_httpx('{"tags": [{"tag":"coding","confidence":5.0}]}'):
        preds = c.predict("p")
    assert preds[0].confidence == 1.0


def test_parse_extracts_json_from_wrapping_prose():
    """Some local models add commentary around the JSON."""
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with _patch_httpx(
        'Here is my classification: {"tags":[{"tag":"coding","confidence":0.8}]} hope this helps!'
    ):
        preds = c.predict("Write code")
    assert len(preds) == 1
    assert preds[0].tag == "coding"


def test_parse_handles_unparseable_response():
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with _patch_httpx("totally not JSON anywhere"):
        preds = c.predict("Write code")
    assert preds == []


def test_parse_handles_top_level_non_object():
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with _patch_httpx('["not","an","object"]'):
        preds = c.predict("Write code")
    assert preds == []


def test_parse_handles_tags_not_a_list():
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with _patch_httpx('{"tags":"not a list"}'):
        preds = c.predict("p")
    assert preds == []


def test_parse_skips_items_with_bad_confidence_type():
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with _patch_httpx('{"tags": [{"tag":"coding","confidence":"not-a-number"}]}'):
        preds = c.predict("p")
    assert preds == []


# ---------- HTTP failure path -----------------------------------------------


def test_http_failure_returns_empty_predictions():
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with _patch_httpx_error(httpx.ConnectError("nope")):
        preds = c.predict("p")
    assert preds == []


def _httpx_response_raw(body: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


def test_predict_handles_missing_choices_in_api_body():
    """Servers occasionally return 200 OK with a body that's missing the
    `choices` key (proxy issues, edge cases). predict() must not propagate
    the KeyError — it would crash the entire classify_unlabeled pass on a
    single bad response. Should return [] gracefully like the other failure
    modes."""
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with patch("lemon_squeeze.classification.llm.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.return_value = (
            _httpx_response_raw({"error": "rate_limited"})
        )
        # Without the fix this raises KeyError("choices")
        preds = c.predict("p")
    assert preds == []


def test_predict_handles_empty_choices_array():
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with patch("lemon_squeeze.classification.llm.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.return_value = (
            _httpx_response_raw({"choices": []})
        )
        # Without the fix this raises IndexError
        preds = c.predict("p")
    assert preds == []


def test_predict_handles_missing_message_content():
    _seed_taxonomy("coding")
    c = LLMClassifier(provider="lm_studio")
    with patch("lemon_squeeze.classification.llm.httpx.Client") as Client:
        Client.return_value.__enter__.return_value.post.return_value = (
            _httpx_response_raw({"choices": [{"finish_reason": "length"}]})
        )
        # Without the fix this raises KeyError("message")
        preds = c.predict("p")
    assert preds == []

"""LLMJudge — mocked tests for the LLM-as-judge."""
from unittest.mock import patch

import httpx

from lemon_squeeze.eval.judges.llm_judge import LLMJudge


def _chat(text: str):
    from lemon_squeeze.eval.clients import ChatResult
    return ChatResult(text=text, tokens_in=10, tokens_out=5, latency_ms=50, cost_usd=0.001)


def test_judge_returns_verdict_from_clean_json():
    j = LLMJudge(rubric_description="x", provider="lm_studio")
    with patch("lemon_squeeze.eval.judges.llm_judge.ChatClient") as Client:
        Client.return_value.chat.return_value = _chat(
            '{"score": 5, "passed": true, "reasoning": "great answer"}'
        )
        v = j.evaluate("p", "r")
    assert v.score == 5.0
    assert v.passed is True
    assert "great answer" in (v.notes or "")
    assert v.judge_model is not None


def test_judge_uses_pass_threshold_when_passed_missing():
    """If JSON omits `passed`, infer from score vs pass_threshold."""
    j = LLMJudge(rubric_description="x", provider="lm_studio", pass_threshold=4)
    with patch("lemon_squeeze.eval.judges.llm_judge.ChatClient") as Client:
        Client.return_value.chat.return_value = _chat(
            '{"score": 4, "reasoning": "ok"}'
        )
        v = j.evaluate("p", "r")
    assert v.passed is True

    with patch("lemon_squeeze.eval.judges.llm_judge.ChatClient") as Client:
        Client.return_value.chat.return_value = _chat(
            '{"score": 2, "reasoning": "weak"}'
        )
        v2 = j.evaluate("p", "r")
    assert v2.passed is False


def test_judge_extracts_json_from_wrapping_prose():
    j = LLMJudge(rubric_description="x", provider="lm_studio")
    with patch("lemon_squeeze.eval.judges.llm_judge.ChatClient") as Client:
        Client.return_value.chat.return_value = _chat(
            'My evaluation: {"score": 3, "passed": false, "reasoning": "ok"} — final.'
        )
        v = j.evaluate("p", "r")
    assert v.score == 3.0
    assert v.passed is False


def test_judge_handles_unparseable_response():
    j = LLMJudge(rubric_description="x", provider="lm_studio")
    with patch("lemon_squeeze.eval.judges.llm_judge.ChatClient") as Client:
        Client.return_value.chat.return_value = _chat("nothing parseable here")
        v = j.evaluate("p", "r")
    assert v.passed is None
    assert v.score == 0.0
    assert "unparseable" in (v.notes or "").lower()


def test_judge_handles_top_level_non_object():
    j = LLMJudge(rubric_description="x", provider="lm_studio")
    with patch("lemon_squeeze.eval.judges.llm_judge.ChatClient") as Client:
        Client.return_value.chat.return_value = _chat('[1, 2, 3]')
        v = j.evaluate("p", "r")
    assert v.passed is None


def test_judge_handles_non_numeric_score():
    j = LLMJudge(rubric_description="x", provider="lm_studio")
    with patch("lemon_squeeze.eval.judges.llm_judge.ChatClient") as Client:
        Client.return_value.chat.return_value = _chat(
            '{"score": "five", "passed": true}'
        )
        v = j.evaluate("p", "r")
    assert v.score == 0.0  # falls back to 0


def test_judge_handles_http_error_gracefully():
    j = LLMJudge(rubric_description="x", provider="lm_studio")
    with patch("lemon_squeeze.eval.judges.llm_judge.ChatClient") as Client:
        Client.return_value.chat.side_effect = httpx.ConnectError("nope")
        v = j.evaluate("p", "r")
    assert v.passed is None
    assert "failed" in (v.notes or "").lower()


def test_judge_uses_correct_default_provider_based_on_settings():
    """If openrouter key is set, defaults to openrouter; else lm_studio."""
    from lemon_squeeze.config import settings

    saved = settings.openrouter_api_key
    settings.openrouter_api_key = "sk-x"
    try:
        j = LLMJudge(rubric_description="x")
        assert j.provider == "openrouter"
    finally:
        settings.openrouter_api_key = saved

    settings.openrouter_api_key = None
    try:
        j2 = LLMJudge(rubric_description="x")
        assert j2.provider == "lm_studio"
    finally:
        settings.openrouter_api_key = saved

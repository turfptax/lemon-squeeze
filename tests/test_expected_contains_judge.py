"""ExpectedContainsJudge — per-prompt ground-truth scoring."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import select

from lemon_squeeze import bench as bench_mod
from lemon_squeeze.db import Evaluation, Model, Prompt, get_session
from lemon_squeeze.eval.clients import ChatResult
from lemon_squeeze.eval.judges import ExpectedContainsJudge, build_judge


def test_judge_skips_when_metadata_absent_by_default():
    j = ExpectedContainsJudge()
    v = j.evaluate("prompt", "response", metadata=None)
    assert v.passed is None
    assert "skipped" in v.notes.lower()


def test_judge_fails_when_metadata_absent_and_on_missing_fail():
    j = ExpectedContainsJudge(on_missing="fail")
    v = j.evaluate("prompt", "response", metadata=None)
    assert v.passed is False
    assert v.score == 0.0


def test_judge_full_match_passes():
    j = ExpectedContainsJudge()
    v = j.evaluate("p", "the answer is 42 and pi", metadata={"expected_contains": ["42", "pi"]})
    assert v.passed is True
    assert v.score == 1.0


def test_judge_partial_match_fails_with_fractional_score():
    j = ExpectedContainsJudge()
    v = j.evaluate("p", "only 42 here", metadata={"expected_contains": ["42", "pi"]})
    assert v.passed is False
    assert v.score == 0.5
    assert "pi" in v.notes  # missing field surfaced


def test_judge_case_insensitive_by_default():
    j = ExpectedContainsJudge()
    v = j.evaluate("p", "FOO BAR", metadata={"expected_contains": ["foo"]})
    assert v.passed is True


def test_judge_case_sensitive_when_configured():
    j = ExpectedContainsJudge(case_sensitive=True)
    v = j.evaluate("p", "FOO BAR", metadata={"expected_contains": ["foo"]})
    assert v.passed is False


def test_build_judge_via_registry():
    j = build_judge("expected_contains", {"on_missing": "fail"})
    assert isinstance(j, ExpectedContainsJudge)
    v = j.evaluate("p", "r", metadata=None)
    assert v.passed is False  # fail mode honored


def test_judge_rejects_invalid_on_missing():
    with pytest.raises(ValueError, match="on_missing"):
        ExpectedContainsJudge(on_missing="bogus")


def test_bench_run_uses_new_judge_via_rubric_framework(tmp_path: Path):
    """Smoke that the bench refactor — using ExpectedContainsJudge through the
    standard rubric framework — produces the same evals as before."""
    d = tmp_path / "tiny_bench"
    (d / "prompts").mkdir(parents=True)
    (d / "prompts" / "math.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"prompt": "2+2?", "intended_tag": "math", "expected_contains": ["4"]}),
                json.dumps({"prompt": "5+5?", "intended_tag": "math", "expected_contains": ["10"]}),
            ]
        ),
        encoding="utf-8",
    )
    with get_session() as s:
        s.add(Model(name="tst/m", provider="lm_studio", size_params_b=1.0, local=True))

    responses = {"2+2?": "the answer is 4", "5+5?": "hmm, 9 maybe?"}

    def fake_chat(model, prompt, **kwargs):
        return ChatResult(text=responses[prompt], tokens_in=1, tokens_out=1, latency_ms=1, cost_usd=0.001)

    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.side_effect = fake_chat
        report = bench_mod.run(d, max_workers=1)

    # One pass, one fail.
    with get_session() as s:
        evals = list(s.scalars(select(Evaluation).where(Evaluation.rubric == "bench:expected_contains")).all())
    assert len(evals) == 2
    by_pass = {e.passed for e in evals}
    assert by_pass == {True, False}
    # The evaluator passes prompt.source_metadata into the judge — verify that
    # the verdict.extra survived round-trip.
    passing = next(e for e in evals if e.passed)
    assert "expected_contains" in (passing.eval_metadata or {})

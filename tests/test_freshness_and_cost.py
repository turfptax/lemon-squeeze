"""Rubric freshness in report + cost-per-pass in bench."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import select

from lemon_squeeze import bench as bench_mod
from lemon_squeeze.db import Evaluation, Model, Prompt, Run, get_session
from lemon_squeeze.eval.clients import ChatResult
from lemon_squeeze.report import build_report


def _seed_eval(rubric: str, *, age_days: float, scored_by: str = "auto") -> None:
    """Insert one Run + Eval with a created_at offset to simulate age."""
    with get_session() as s:
        p = Prompt(
            content=f"p-{rubric}-{age_days}",
            content_hash=f"h-{rubric}-{age_days}",
            char_count=2,
            source="test",
        )
        m = s.scalar(select(Model).where(Model.name == "tst/m")) or Model(
            name="tst/m", provider="test", local=True
        )
        if m.id is None:
            s.add(m); s.flush()
        s.add(p); s.flush()
        run = Run(prompt_id=p.id, model_id=m.id, response="ok")
        s.add(run); s.flush()
        when = datetime.now(timezone.utc) - timedelta(days=age_days)
        s.add(
            Evaluation(
                run_id=run.id,
                rubric=rubric,
                score=1.0,
                passed=True,
                scored_by=scored_by,
                created_at=when,
            )
        )


def test_report_surfaces_rubric_freshness_with_scored_by_breakdown():
    _seed_eval("human_pass", age_days=2, scored_by="human")
    _seed_eval("human_pass", age_days=1, scored_by="human")
    _seed_eval("llm_judge", age_days=45, scored_by="llm")  # stale

    rep = build_report(staleness_days=30)
    by_name = {rf.rubric: rf for rf in rep.rubric_freshness}

    assert by_name["human_pass"].n_evals == 2
    assert by_name["human_pass"].stale is False
    assert by_name["human_pass"].scored_by_breakdown == [("human", 2)]

    assert by_name["llm_judge"].stale is True
    assert 44 <= by_name["llm_judge"].age_days <= 46


def test_freshness_handles_zero_evals_gracefully():
    rep = build_report()
    assert rep.rubric_freshness == []  # no evals → no rubrics tracked


def test_bench_run_populates_cost_per_pass(tmp_path: Path):
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

    # Half-pass: only the first prompt's response contains "4". Each call returns
    # a fixed-cost ChatResult so we can predict cost_per_pass exactly.
    responses = {"2+2?": "4", "5+5?": "12"}

    def fake_chat(model, prompt, **kwargs):
        return ChatResult(text=responses[prompt], tokens_in=10, tokens_out=2, latency_ms=50, cost_usd=0.001)

    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.side_effect = fake_chat
        report = bench_mod.run(d, max_workers=1)

    math_row = next(s for s in report.per_category if s.category == "math")
    assert math_row.n_runs == 2
    assert math_row.pass_count == 1
    assert math_row.pass_rate == 0.5
    assert math_row.avg_cost_usd == 0.001
    # cost-per-pass = avg_cost / pass_rate = 0.001 / 0.5 = 0.002
    assert math_row.cost_per_pass is not None
    assert abs(math_row.cost_per_pass - 0.002) < 1e-9


def test_bench_cost_per_pass_is_none_when_zero_pass(tmp_path: Path):
    """No passes → cost_per_pass should stay None (division-by-zero guard)."""
    d = tmp_path / "zero_bench"
    (d / "prompts").mkdir(parents=True)
    (d / "prompts" / "x.jsonl").write_text(
        json.dumps({"prompt": "anything", "intended_tag": "x", "expected_contains": ["nope"]}),
        encoding="utf-8",
    )
    with get_session() as s:
        s.add(Model(name="tst/m", provider="lm_studio", size_params_b=1.0, local=True))

    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.return_value = ChatResult(
            text="this doesn't contain it", tokens_in=1, tokens_out=1, latency_ms=1, cost_usd=0.01
        )
        report = bench_mod.run(d, max_workers=1)
    row = next(s for s in report.per_category if s.category == "x")
    assert row.pass_rate == 0.0
    assert row.cost_per_pass is None

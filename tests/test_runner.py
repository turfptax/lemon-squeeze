"""Run executor — mock the HTTP client and assert DB writes."""
from unittest.mock import patch

from sqlalchemy import select

from lemon_squeeze.db import Model, Prompt, Run, get_session
from lemon_squeeze.eval.clients import ChatResult
from lemon_squeeze.eval.runner import execute_run, fanout


def _seed_prompt_and_model() -> tuple[int, int]:
    with get_session() as s:
        p = Prompt(content="Hello", content_hash="hello-h", char_count=5, source="test")
        m = Model(name="local/tiny", provider="lm_studio", size_params_b=3.0, local=True,
                  cost_in_per_mtok=0.5, cost_out_per_mtok=1.0)
        s.add(p); s.add(m); s.flush()
        return p.id, m.id


def test_execute_run_persists_response_and_usage():
    pid, mid = _seed_prompt_and_model()
    fake = ChatResult(text="hi there", tokens_in=10, tokens_out=5, latency_ms=120, cost_usd=0.000010)
    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.return_value = fake
        with get_session() as s:
            prompt = s.get(Prompt, pid)
            model = s.get(Model, mid)
        run = execute_run(prompt, model)
    assert run.response == "hi there"
    assert run.tokens_in == 10 and run.tokens_out == 5
    assert run.latency_ms == 120
    assert run.error is None


def test_fanout_skips_existing_pairs():
    pid, mid = _seed_prompt_and_model()
    fake = ChatResult(text="ok", tokens_in=1, tokens_out=1, latency_ms=10)
    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.return_value = fake
        r1 = fanout()
        r2 = fanout()
    assert r1.attempted == 1
    assert r1.succeeded == 1
    assert r2.attempted == 0


def test_fanout_records_errors_without_raising():
    _seed_prompt_and_model()
    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.side_effect = ValueError("boom")
        report = fanout()
    assert report.attempted == 1
    assert report.failed == 1
    assert any("parse_error" in e or "boom" in e for e in report.errors)

    with get_session() as s:
        run = s.scalar(select(Run))
        assert run is not None


def test_execute_run_with_no_system_or_metadata_leaves_run_metadata_null():
    """execute_run was unconditionally setting run_metadata = {"system": None}
    because the dict construction always had the 'system' key — making the
    `or None` fall-through dead code. Net effect: every Run row had a
    polluted run_metadata pointing at no meaningful info, and any
    `WHERE run_metadata IS NOT NULL` query returned all rows.
    """
    pid, mid = _seed_prompt_and_model()
    with get_session() as s:
        prompt = s.get(Prompt, pid)
        model = s.get(Model, mid)
    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.return_value = ChatResult(
            text="hi", tokens_in=2, tokens_out=2, latency_ms=10,
        )
        # No system, no extra_metadata.
        run = execute_run(prompt, model)

    assert run.run_metadata is None, (
        f"run_metadata should be NULL when no system or extra metadata "
        f"was provided; got {run.run_metadata!r}"
    )


def test_execute_run_with_system_records_system_in_metadata():
    pid, mid = _seed_prompt_and_model()
    with get_session() as s:
        prompt = s.get(Prompt, pid)
        model = s.get(Model, mid)
    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.return_value = ChatResult(
            text="hi", tokens_in=2, tokens_out=2, latency_ms=10,
        )
        run = execute_run(prompt, model, system="be terse")

    assert run.run_metadata == {"system": "be terse"}


def test_execute_run_merges_extra_metadata():
    pid, mid = _seed_prompt_and_model()
    with get_session() as s:
        prompt = s.get(Prompt, pid)
        model = s.get(Model, mid)
    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.return_value = ChatResult(
            text="hi", tokens_in=2, tokens_out=2, latency_ms=10,
        )
        run = execute_run(
            prompt, model,
            system="be terse",
            extra_metadata={"trace_id": "abc-123"},
        )

    assert run.run_metadata == {"system": "be terse", "trace_id": "abc-123"}

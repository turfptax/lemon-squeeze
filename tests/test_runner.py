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


def test_execute_run_records_indexerror_as_parse_error():
    """ChatClient.chat reads data['choices'][0] without guard. If a 200 OK
    response comes back with `choices: []`, IndexError fires. Previously
    execute_run caught (KeyError, ValueError, TypeError) but missed
    IndexError, so the whole batch would crash on a single bad response."""
    pid, mid = _seed_prompt_and_model()
    with get_session() as s:
        prompt = s.get(Prompt, pid)
        model = s.get(Model, mid)
    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.side_effect = IndexError("list index out of range")
        # Should not raise — should record as parse error on the run row.
        run = execute_run(prompt, model)
    assert run.error is not None
    assert "parse_error" in run.error
    assert "IndexError" in run.error


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


def test_execute_run_collapses_multiline_httpx_error_to_one_line():
    """Caught against real LM Studio: when the server returns 500 (model
    swapping in/out of VRAM), httpx.HTTPStatusError stringifies as TWO
    lines -- the actual error, then "For more information check: <MDN URL>".
    With 18 errors in a bench run, the per-row error printout becomes
    unreadable garbage as each error spans 3-4 terminal lines. Collapse
    to one line at storage so the DB column, bench output, JSON export,
    and HTML report are all clean."""
    import httpx

    pid, mid = _seed_prompt_and_model()
    with get_session() as s:
        prompt = s.get(Prompt, pid)
        model = s.get(Model, mid)

    fake_response = httpx.Response(500, request=httpx.Request("POST", "http://x/y"))
    err = httpx.HTTPStatusError(
        "Server error '500 Internal Server Error' for url 'http://x/y'\n"
        "For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/500",
        request=fake_response.request, response=fake_response,
    )
    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.side_effect = err
        run = execute_run(prompt, model)

    assert run.error is not None
    assert "\n" not in run.error, (
        f"Run.error should be single-line; got {run.error!r}"
    )
    # The actual error info is preserved.
    assert "500" in run.error
    assert "http://x/y" in run.error
    # The MDN URL boilerplate is dropped (pure noise, same URL for every
    # status code in that class).
    assert "developer.mozilla.org" not in run.error
    assert "For more information check" not in run.error

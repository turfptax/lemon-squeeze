"""Concurrency tests for the ThreadPoolExecutor-backed fanout."""
import threading
import time
from unittest.mock import patch

from sqlalchemy import select

from lemon_squeeze.db import Model, Prompt, Run, get_session
from lemon_squeeze.eval.clients import ChatResult
from lemon_squeeze.eval.runner import fanout


def _seed(n_prompts: int = 5, n_models: int = 3) -> None:
    with get_session() as s:
        for i in range(n_prompts):
            s.add(Prompt(content=f"prompt-{i}", content_hash=f"ph-{i}", char_count=8, source="test"))
        for j in range(n_models):
            s.add(
                Model(
                    name=f"local/m-{j}",
                    provider="lm_studio",
                    size_params_b=float(j + 1),
                    local=True,
                )
            )


def test_fanout_actually_runs_concurrently():
    _seed(n_prompts=4, n_models=2)
    # Each "chat" sleeps; if we ran serially the total wall time would be
    # 8 * 0.05 = 0.4s. With 4 workers we expect ~0.1s.
    barrier = threading.Barrier(parties=4, timeout=5.0)
    seen_threads: set[int] = set()
    lock = threading.Lock()

    def fake_chat(*args, **kwargs):
        with lock:
            seen_threads.add(threading.get_ident())
        # All 4 workers must reach this barrier before any can finish — proves
        # parallelism, doesn't depend on a fragile wall-clock measurement.
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            raise AssertionError("workers did not run concurrently")
        return ChatResult(text="ok", tokens_in=1, tokens_out=1, latency_ms=1)

    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.side_effect = fake_chat
        report = fanout(max_workers=4)

    assert report.attempted == 8
    assert report.succeeded == 8
    # At least 4 distinct threads handled work.
    assert len(seen_threads) >= 4


def test_fanout_max_workers_one_falls_back_to_serial():
    _seed(n_prompts=2, n_models=2)
    times: list[float] = []

    def fake_chat(*args, **kwargs):
        times.append(time.perf_counter())
        time.sleep(0.02)
        return ChatResult(text="ok", tokens_in=1, tokens_out=1, latency_ms=1)

    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.side_effect = fake_chat
        report = fanout(max_workers=1)
    assert report.attempted == 4
    assert report.succeeded == 4
    # Serial = each call starts only after the previous finished, so gaps >= 0.02s.
    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    assert all(g >= 0.015 for g in gaps)


def test_fanout_aggregates_errors_thread_safely():
    _seed(n_prompts=3, n_models=2)
    counter = {"n": 0}
    counter_lock = threading.Lock()

    def fake_chat(*args, **kwargs):
        with counter_lock:
            counter["n"] += 1
            should_fail = counter["n"] % 2 == 0
        if should_fail:
            raise ValueError("simulated failure")
        return ChatResult(text="ok", tokens_in=1, tokens_out=1, latency_ms=1)

    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.side_effect = fake_chat
        report = fanout(max_workers=4)
    assert report.attempted == 6
    # Half succeed, half fail. Exact split depends on thread scheduling but
    # the totals must sum.
    assert report.succeeded + report.failed == 6
    assert report.failed > 0

    # Every attempt produced a Run row (even errors).
    with get_session() as s:
        all_runs = s.scalars(select(Run)).all()
    assert len(list(all_runs)) == 6

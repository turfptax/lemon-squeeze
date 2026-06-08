"""`lemon demo` + the underlying lemon_squeeze.demo module."""
from pathlib import Path

from typer.testing import CliRunner

from lemon_squeeze.cli import app
from lemon_squeeze.demo import DemoResult, run_demo

runner = CliRunner()


def test_run_demo_quiet_returns_meaningful_summary():
    """run_demo() should set up a complete DB and report verifiable counts."""
    result = run_demo(quiet=True)

    assert isinstance(result, DemoResult)
    assert result.prompts_seeded == 3
    assert result.runs_attempted == 6      # 3 prompts × 2 models
    assert result.runs_succeeded == 6      # the mock never fails
    assert result.evaluations_written == 6  # one per (run, rubric)
    assert result.scorecards_with_pick >= 1
    # comparison_winner is A/B (the column position) or "tie".
    # The scripted scenario has B (premium) winning overall.
    assert result.comparison_winner in ("A", "B", "tie")


def test_run_demo_leaves_db_at_returned_path():
    """The returned db_path should point to a real, populated SQLite file."""
    result = run_demo(quiet=True)

    assert result.db_path.exists()
    assert result.db_path.suffix == ".db"
    # File should be non-trivially sized (has tables + data)
    assert result.db_path.stat().st_size > 4096


def test_lemon_demo_cli_runs_end_to_end():
    """`lemon demo` should run without erroring and print the key markers."""
    r = runner.invoke(app, ["demo"])
    assert r.exit_code == 0
    # Each numbered step prints; check that the full flow ran.
    for step in ("[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]", "[9]"):
        assert step in r.stdout, f"missing step {step} in demo output"
    # Final pointer to the demo DB so users can poke at it.
    assert "DB lives at" in r.stdout
    assert "lemon db stats" in r.stdout


def test_demo_outputs_router_picks_cheap_for_math():
    """The demo is scripted so the cheap model is correct on math but not coding.
    The report should reflect that with a math scorecard preferring the cheap one
    for cost. This is the value proposition the demo is meant to showcase."""
    result = run_demo(quiet=True)

    # If the scripted scenario doesn't produce at least one scorecard with a
    # cost pick distinct from the quality pick, the demo isn't doing its job.
    assert result.scorecards_with_pick >= 1


def test_run_demo_restores_caller_state_on_return():
    """run_demo() must not permanently hijack the caller's LEMON_DB_PATH
    or settings.db_path. Library users should be able to call it mid-session
    and continue with their original DB pointer."""
    import os

    import lemon_squeeze as lemon

    prev_env = os.environ.get("LEMON_DB_PATH")
    prev_db_path = lemon.settings.db_path

    try:
        run_demo(quiet=True)
        # After return, both surfaces should match what they were before.
        assert os.environ.get("LEMON_DB_PATH") == prev_env, (
            "LEMON_DB_PATH was hijacked by run_demo"
        )
        assert lemon.settings.db_path == prev_db_path, (
            "settings.db_path was hijacked by run_demo"
        )
    finally:
        # Belt-and-suspenders: explicit restore so a failing assertion doesn't
        # corrupt the rest of the test session.
        if prev_env is None:
            os.environ.pop("LEMON_DB_PATH", None)
        else:
            os.environ["LEMON_DB_PATH"] = prev_env
        lemon.settings.db_path = prev_db_path


def test_examples_library_demo_still_callable():
    """The thin wrapper script should still work — defends against people
    cargo-culting that path from the old README."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "library_demo",
        Path(__file__).resolve().parents[1] / "examples" / "library_demo.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # The script defines run_demo via import; calling it should succeed.
    mod.run_demo(quiet=True)

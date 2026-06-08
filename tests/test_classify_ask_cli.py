"""`lemon classify ask` — one-shot classification without DB."""
import json

from typer.testing import CliRunner

from lemon_squeeze.cli import app

runner = CliRunner()


def test_classify_ask_heuristic_finds_coding():
    r = runner.invoke(app, [
        "classify", "ask",
        "Write a Python function to sort a list.",
        "--classifier", "heuristic",
    ])
    assert r.exit_code == 0
    assert "coding" in r.stdout


def test_classify_ask_default_ensemble():
    r = runner.invoke(app, [
        "classify", "ask",
        "What is the capital of Australia?",
    ])
    assert r.exit_code == 0
    # qa_factual is the heuristic match for "what is the capital of"
    assert "qa_factual" in r.stdout or "qa" in r.stdout.lower()


def test_classify_ask_json_output_parses():
    r = runner.invoke(app, [
        "classify", "ask",
        "Translate 'hello' into French.",
        "--classifier", "heuristic",
        "--json",
    ])
    assert r.exit_code == 0
    body = json.loads(r.stdout)
    assert body["prompt"] == "Translate 'hello' into French."
    assert body["classifier"] == "heuristic"
    tags = [p["tag"] for p in body["predictions"]]
    assert "translation" in tags


def test_classify_ask_top_n_limits_output():
    r = runner.invoke(app, [
        "classify", "ask",
        "Summarize how Python list comprehensions work using a code example.",
        "--classifier", "heuristic",
        "--json",
        "--top", "1",
    ])
    body = json.loads(r.stdout)
    assert len(body["predictions"]) == 1


def test_classify_ask_unknown_classifier_errors():
    r = runner.invoke(app, [
        "classify", "ask", "x", "--classifier", "nonsense",
    ])
    assert r.exit_code == 2
    assert "Unknown classifier" in r.stdout


def test_classify_ask_ml_without_trained_model():
    """If ML classifier isn't trained, fail with a helpful hint."""
    # The conftest fixture doesn't ship a trained model, but the test that
    # actually trains one may have written joblib to data/models/. To make
    # this test robust we patch MLClassifier.load to return None.
    from unittest.mock import patch

    with patch("lemon_squeeze.classification.MLClassifier.load", return_value=None):
        r = runner.invoke(app, ["classify", "ask", "x", "--classifier", "ml"])
    assert r.exit_code == 1
    assert "train-ml" in r.stdout


def test_classify_ask_does_not_write_to_db():
    """Ad-hoc classify should not write anything to PromptTag."""
    from sqlalchemy import func, select

    from lemon_squeeze.db import PromptTag, get_session

    with get_session() as s:
        before = s.scalar(select(func.count()).select_from(PromptTag)) or 0

    r = runner.invoke(app, [
        "classify", "ask",
        "Write a Python function that prints hello world.",
    ])
    assert r.exit_code == 0

    with get_session() as s:
        after = s.scalar(select(func.count()).select_from(PromptTag)) or 0
    assert after == before  # no DB writes from `classify ask`


def test_classify_ask_no_predictions_returns_message():
    """A bizarre prompt with no heuristic signals shouldn't crash."""
    r = runner.invoke(app, [
        "classify", "ask",
        "xx yy zz",
        "--classifier", "heuristic",
    ])
    assert r.exit_code == 0
    # heuristic always returns at least 'unknown' for unmatched prompts.
    assert "unknown" in r.stdout or "No predictions" in r.stdout

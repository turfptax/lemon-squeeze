"""MLClassifier — full train → save → load → predict cycle."""
from pathlib import Path

import pytest

from lemon_squeeze.classification.ml import MIN_EXAMPLES_PER_TAG, MLClassifier
from lemon_squeeze.db import Prompt, PromptTag, get_session


def _seed_labeled(content: str, tag: str, classifier: str = "human", *, confidence: float = 1.0) -> None:
    with get_session() as s:
        p = Prompt(
            content=content,
            content_hash=f"ml-{content[:30]}-{tag}",
            char_count=len(content),
            source="ml-test",
        )
        s.add(p); s.flush()
        s.add(PromptTag(
            prompt_id=p.id, tag=tag, classifier=classifier, confidence=confidence,
        ))


# ---------- Empty / insufficient data ---------------------------------------


def test_train_with_no_labels_raises():
    clf = MLClassifier()
    with pytest.raises(RuntimeError, match="No labeled prompts"):
        clf.train_from_db()


def test_train_with_insufficient_examples_per_tag_raises():
    """Each tag needs ≥3 examples; below that they're dropped."""
    # Two prompts per tag — below the threshold.
    for i in range(2):
        _seed_labeled(f"Coding sample {i} write code", "coding")
        _seed_labeled(f"Math sample {i} compute x", "math")

    clf = MLClassifier()
    with pytest.raises(RuntimeError, match=f"≥ {MIN_EXAMPLES_PER_TAG}"):
        clf.train_from_db()


def test_predict_on_untrained_classifier_returns_empty():
    clf = MLClassifier()
    assert clf.predict("anything") == []


def test_save_untrained_raises():
    clf = MLClassifier()
    with pytest.raises(RuntimeError, match="train first"):
        clf.save(Path("/tmp/never-written.joblib"))


# ---------- Successful train + predict --------------------------------------


def _seed_enough_data() -> None:
    """Enough labels per tag to clear MIN_EXAMPLES_PER_TAG."""
    coding_samples = [
        "Write a Python function to compute fibonacci",
        "Implement a recursive function in Python",
        "Define a class with methods for a stack",
        "Create a Python script that reads a file",
        "Build a simple Python function for sorting",
    ]
    math_samples = [
        "What is the integral of x squared",
        "Compute the derivative of sin x",
        "Solve the equation 3x plus 7 equals 22",
        "Calculate the area of a triangle with base 5",
        "What is 12 factorial",
    ]
    for s in coding_samples:
        _seed_labeled(s, "coding")
    for s in math_samples:
        _seed_labeled(s, "math")


def test_train_succeeds_with_enough_data():
    _seed_enough_data()
    clf = MLClassifier()
    counts = clf.train_from_db()
    assert counts["coding"] == 5
    assert counts["math"] == 5


def test_predict_returns_confidence_scores_after_training():
    _seed_enough_data()
    clf = MLClassifier(threshold=0.0)  # accept any prediction so we can verify shape
    clf.train_from_db()
    preds = clf.predict("Write a Python program to print hello world")
    assert preds, "untrained model should never reach here, and threshold=0 returns all classes"
    assert all(p.classifier == "ml" for p in preds)
    assert all(0.0 <= p.confidence <= 1.0 for p in preds)


def test_predict_with_lower_threshold_returns_more():
    _seed_enough_data()
    clf_high = MLClassifier(threshold=0.5)
    clf_high.train_from_db()
    clf_low = MLClassifier(threshold=0.05)
    # Re-seed not needed — training reads from DB and is independent.
    clf_low.train_from_db()
    pl = clf_low.predict("Something math related, integral of x")
    ph = clf_high.predict("Something math related, integral of x")
    assert len(pl) >= len(ph)


# ---------- Save / load round-trip ------------------------------------------


def test_save_then_load_round_trip(tmp_path: Path):
    _seed_enough_data()
    path = tmp_path / "model.joblib"
    clf = MLClassifier(threshold=0.4)
    clf.train_from_db()
    clf.save(path)

    assert path.exists()
    loaded = MLClassifier.load(path)
    assert loaded is not None
    assert loaded.threshold == 0.4
    # Predict produces same shape (we don't check exact equality of confidences
    # because sklearn doesn't guarantee bit-exact reload, only that the model
    # is functionally equivalent).
    p_orig = clf.predict("Compute the integral of x")
    p_loaded = loaded.predict("Compute the integral of x")
    assert {p.tag for p in p_orig} == {p.tag for p in p_loaded}


def test_load_returns_none_when_path_missing(tmp_path: Path):
    assert MLClassifier.load(tmp_path / "nope.joblib") is None


# ---------- Label-source preference -----------------------------------------


def test_human_labels_preferred_over_heuristic():
    """When both human and heuristic labels exist for the same prompt,
    human wins."""
    _seed_labeled("This is a special prompt", "coding", classifier="human")
    with get_session() as s:
        # Same prompt also has a heuristic tag we'd ignore.
        p = s.scalars(__import__("sqlalchemy").select(Prompt)).first()
        s.add(PromptTag(
            prompt_id=p.id, tag="math", classifier="heuristic", confidence=0.5,
        ))

    examples = MLClassifier._collect_examples()
    assert len(examples) == 1
    content, tags = examples[0]
    assert tags == ["coding"]  # human label wins


def test_heuristic_fallback_when_no_human_label():
    _seed_labeled("falls back", "coding", classifier="heuristic", confidence=0.8)

    examples = MLClassifier._collect_examples()
    assert len(examples) == 1
    assert examples[0][1] == ["coding"]


def test_heuristic_highest_confidence_wins():
    """If only heuristic labels exist, the highest-confidence one is used."""
    with get_session() as s:
        p = Prompt(content="multi-heuristic prompt", content_hash="multi-h",
                   char_count=10, source="test")
        s.add(p); s.flush()
        s.add(PromptTag(prompt_id=p.id, tag="coding", classifier="heuristic", confidence=0.3))
        s.add(PromptTag(prompt_id=p.id, tag="math", classifier="heuristic", confidence=0.9))

    examples = MLClassifier._collect_examples()
    assert examples[0][1] == ["math"]  # higher confidence wins


def test_unknown_heuristic_tag_excluded():
    _seed_labeled("xxx zzz qqq", "unknown", classifier="heuristic", confidence=0.1)

    examples = MLClassifier._collect_examples()
    assert examples == []  # 'unknown' tags are not training material


# ---------- report_balance --------------------------------------------------


def test_report_balance_empty_input():
    assert MLClassifier.report_balance({}) == "(no labels)"


def test_report_balance_shows_per_tag_bars():
    report = MLClassifier.report_balance({"coding": 10, "math": 5, "summarization": 2})
    assert "coding" in report
    assert "math" in report
    assert "summarization" in report
    # The bar widths should be proportional — the largest bucket is wider than
    # the smallest. We don't depend on a specific glyph or absolute width.
    lines = report.splitlines()
    by_tag = {line.split()[0]: line for line in lines}
    bar_width = lambda line: sum(1 for ch in line if not ch.isascii() or ch == "=")
    assert bar_width(by_tag["coding"]) >= bar_width(by_tag["summarization"])

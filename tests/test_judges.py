"""Judge behavior tests — no network."""
from lemon_squeeze.eval.judges import build_judge


def test_contains_all_of_and_none_of():
    j = build_judge("contains", {"all_of": ["python", "function"], "none_of": ["sorry"]})
    v = j.evaluate("", "Here is a Python function that does X.")
    assert v.passed is True
    assert v.score == 1.0

    v = j.evaluate("", "Sorry, I can't help.")
    assert v.passed is False
    assert v.score == 0.0


def test_contains_partial_score():
    j = build_judge("contains", {"all_of": ["alpha", "beta", "gamma"]})
    v = j.evaluate("", "alpha and beta but not the third one")
    assert v.passed is False
    assert v.score == 2 / 3


def test_exact_match_normalizes_whitespace():
    j = build_judge("exact_match", {"expected": "Hello world"})
    assert j.evaluate("", "hello   WORLD").passed is True
    assert j.evaluate("", "hi world").passed is False


def test_regex_judge():
    j = build_judge("regex", {"pattern": r"^\d+$"})
    assert j.evaluate("", "42").passed is True
    assert j.evaluate("", "42 things").passed is False


def test_json_valid_with_required_keys():
    j = build_judge("json_valid", {"required_keys": ["name", "age"]})
    v = j.evaluate("", '{"name": "x", "age": 1}')
    assert v.passed is True
    v = j.evaluate("", '{"name": "x"}')
    assert v.passed is False
    assert v.score == 0.5


def test_json_valid_strips_code_fence():
    j = build_judge("json_valid", {})
    v = j.evaluate("", "```json\n{\"x\": 1}\n```")
    assert v.passed is True


def test_unknown_judge_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown judge kind"):
        build_judge("does_not_exist", {})

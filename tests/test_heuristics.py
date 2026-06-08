from lemon_squeeze.classification.heuristics import HeuristicClassifier


def _top_tags(prompt: str) -> list[str]:
    return [p.tag for p in HeuristicClassifier().predict(prompt)]


def test_coding_signal_from_fenced_block():
    prompt = "Why does this not work?\n```python\nfor i in range(10):\n  print(i\n```"
    assert "coding" in _top_tags(prompt)


def test_summarization_signal():
    assert "summarization" in _top_tags("Please summarize this article in two sentences.")


def test_translation_signal():
    assert "translation" in _top_tags("Translate the following paragraph into French.")


def test_math_signal():
    assert "math" in _top_tags("Compute the derivative of 3x^2 + 2x + 1.")


def test_multilabel_coding_plus_summarization():
    prompt = "Summarize what this Python function does:\n```python\ndef f(x): return x*2\n```"
    tags = _top_tags(prompt)
    assert "coding" in tags
    assert "summarization" in tags


def test_unknown_when_no_signals_match():
    assert _top_tags("hi") == ["unknown"]


def test_cpp_keyword_actually_fires_coding_signal():
    """SIGNALS["coding"] has _kw(..., "c\\+\\+", ...). The author manually
    escaped the `+` regex metacharacters, but `_kw` already passes the word
    through `re.escape`, so the pattern double-escapes and matches the
    literal 5-char sequence `c\\+\\+` (with backslashes), not "c++". Net
    effect: prompts mentioning C++ never gain the c++ signal's weight.

    Compare the same prompt with and without "C++" to assert the signal
    fires (raising the coding score above what other keywords contribute).
    Without the fix both come back at identical confidence."""

    def _coding_conf(prompt: str) -> float:
        for p in HeuristicClassifier().predict(prompt):
            if p.tag == "coding":
                return p.confidence
        return 0.0

    base_conf = _coding_conf("Write a function to compute fibonacci.")
    cpp_conf = _coding_conf("Write a C++ function to compute fibonacci.")
    assert cpp_conf > base_conf, (
        f"adding 'C++' to a coding prompt should raise the coding signal "
        f"(_kw double-escape bug otherwise); got base={base_conf} cpp={cpp_conf}"
    )

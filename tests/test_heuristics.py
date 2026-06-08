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

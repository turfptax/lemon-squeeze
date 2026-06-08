from lemon_squeeze.classification.base import Classifier, TagPrediction
from lemon_squeeze.classification.ensemble import EnsembleClassifier, classify_unlabeled
from lemon_squeeze.classification.heuristics import HeuristicClassifier
from lemon_squeeze.db import PromptTag, get_session
from lemon_squeeze.ingestion.base import Ingester, RawPrompt


class _StaticIngester(Ingester):
    source_name = "test"

    def __init__(self, items):
        self.items = items

    def iter_prompts(self):
        for content in self.items:
            yield RawPrompt(content=content, source=self.source_name)


class _FakeLLM(Classifier):
    name = "llm"

    def __init__(self, tag: str):
        self.tag = tag
        self.calls = 0

    def predict(self, prompt):
        self.calls += 1
        return [TagPrediction(tag=self.tag, confidence=0.9, classifier=self.name)]


def test_classify_writes_tags():
    _StaticIngester(
        [
            "Summarize the Federalist Papers in two paragraphs.",
            "Write a Python function to reverse a string.",
        ]
    ).run()

    stats = classify_unlabeled()
    assert stats.tags_written > 0

    with get_session() as s:
        tags = {(t.prompt_id, t.tag) for t in s.query(PromptTag).all()}
    tagged_tags = {t for _, t in tags}
    assert "summarization" in tagged_tags
    assert "coding" in tagged_tags


def test_ensemble_skips_llm_when_others_agree():
    fake = _FakeLLM(tag="creative")
    ens = EnsembleClassifier(
        members=[HeuristicClassifier(), fake], llm_only_on_disagreement=True
    )
    # Clear-cut prompt — heuristic confidently picks 'coding'; nothing to disagree with.
    ens.predict("Write a Python function that returns the nth Fibonacci number.")
    assert fake.calls == 0


def test_ensemble_invokes_llm_on_ambiguity():
    fake = _FakeLLM(tag="creative")
    ens = EnsembleClassifier(
        members=[HeuristicClassifier(), fake], llm_only_on_disagreement=True
    )
    # No heuristic signal -> heuristic returns 'unknown' -> no agreement -> LLM runs.
    ens.predict("xy zzy plugh")
    assert fake.calls == 1

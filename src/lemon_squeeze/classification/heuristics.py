"""Regex + keyword scoring classifier.

Deterministic baseline. Each tag has a set of weighted signals; the classifier
returns every tag whose normalized score crosses a threshold. Multi-label by
design — a prompt asking for code that summarizes a paper is both `coding` and
`summarization`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from lemon_squeeze.classification.base import Classifier, TagPrediction


@dataclass
class Signal:
    pattern: re.Pattern[str]
    weight: float


def _kw(*words: str, weight: float = 1.0) -> Signal:
    body = "|".join(re.escape(w) for w in words)
    return Signal(re.compile(rf"\b({body})\b", re.IGNORECASE), weight)


def _re(pattern: str, weight: float = 1.5) -> Signal:
    return Signal(re.compile(pattern, re.IGNORECASE | re.MULTILINE), weight)


# Weights are rough — tune as labeled data accumulates.
SIGNALS: dict[str, list[Signal]] = {
    "coding": [
        _re(r"```[\w+-]*\n", weight=3.0),
        _re(r"(^|\n)\s{4}\S", weight=1.0),  # indented code
        _kw("function", "method", "class", "variable", "compile", "debug", "stacktrace"),
        _kw("python", "javascript", "typescript", "rust", "go", "java", "c\\+\\+", "sql"),
        _kw("bug", "fix", "refactor", "implement", "snippet"),
    ],
    "reasoning": [
        _kw("why", "because", "therefore", "implies", "deduce", "reasoning", "logic", "step by step"),
        _re(r"\b(if .+ then|prove that|show that)\b", weight=2.0),
    ],
    "math": [
        _re(r"\$.+?\$", weight=2.0),
        _re(r"\b\d+\s*[+\-*/^]\s*\d+", weight=2.0),
        _kw("integral", "derivative", "equation", "matrix", "vector", "theorem", "calculate", "compute"),
    ],
    "summarization": [
        _kw("summarize", "summarise", "summary", "tl;dr", "tldr", "condense", "abstract", "key points"),
        _re(r"in (\d+|one|two|three|five) (sentence|paragraph|bullet)", weight=2.0),
    ],
    "extraction": [
        _kw("extract", "pull out", "list all", "find every", "parse"),
        _re(r"return (a |the )?(json|yaml|csv|list|table)", weight=2.0),
    ],
    "classification": [
        _kw("classify", "categorize", "categorise", "label", "tag", "which category"),
    ],
    "creative": [
        _kw("write a story", "poem", "haiku", "screenplay", "lyrics", "creative", "imagine", "fictional"),
        _re(r"\bstory about\b", weight=2.0),
    ],
    "conversation": [
        _kw("how are you", "let's chat", "your opinion", "what do you think", "tell me about yourself"),
    ],
    "instruction": [
        _kw("how do i", "how to", "steps to", "guide me", "walk me through", "instructions for"),
    ],
    "translation": [
        _re(r"\btranslate\b.+\binto\b", weight=3.0),
        _kw("in french", "in spanish", "in german", "in japanese", "in mandarin"),
    ],
    "qa_factual": [
        _re(r"^(what|who|when|where|which) (is|was|are|were)\b", weight=2.0),
        _kw("capital of", "population of", "year did", "who invented"),
    ],
    "rewrite": [
        _kw("rewrite", "rephrase", "reword", "make this more", "tone of", "polish"),
    ],
    "planning": [
        _kw("plan", "outline", "schedule", "roadmap", "agenda", "itinerary", "checklist"),
    ],
}

# Tags with score below this (after length normalization) are dropped.
SCORE_THRESHOLD = 1.0
# Confidence saturates at this raw score.
SATURATION_SCORE = 5.0


class HeuristicClassifier(Classifier):
    name = "heuristic"

    def predict(self, prompt: str) -> list[TagPrediction]:
        if not prompt:
            return []
        scores: dict[str, float] = {}
        for tag, signals in SIGNALS.items():
            total = 0.0
            for sig in signals:
                matches = sig.pattern.findall(prompt)
                if matches:
                    total += sig.weight * len(matches)
            if total > 0:
                scores[tag] = total

        predictions = [
            TagPrediction(
                tag=tag,
                confidence=min(score / SATURATION_SCORE, 1.0),
                classifier=self.name,
            )
            for tag, score in scores.items()
            if score >= SCORE_THRESHOLD
        ]
        if not predictions:
            return [TagPrediction(tag="unknown", confidence=0.1, classifier=self.name)]
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions

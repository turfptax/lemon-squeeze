"""TF-IDF + LogisticRegression multi-label classifier.

Trained on whatever labeled prompts the DB already has — labels come from
`prompt_tags` rows in declining order of trust: `classifier='human'` first,
then `classifier='bench'` (the category a benchmark file declared outright),
then the highest-confidence heuristic tag. The model serializes to a single
joblib file under `data/models/`.

This is intentionally simple. The ML classifier earns its keep once a few
hundred curated examples accumulate; before then the heuristic carries the load.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from lemon_squeeze.classification.base import Classifier, TagPrediction
from lemon_squeeze.config import PROJECT_ROOT
from lemon_squeeze.db import Prompt, PromptTag, get_session

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "prompt_classifier.joblib"
DEFAULT_THRESHOLD = 0.35
MIN_EXAMPLES_PER_TAG = 3


class MLClassifier(Classifier):
    name = "ml"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._pipeline: Pipeline | None = None
        self._mlb: MultiLabelBinarizer | None = None

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> MLClassifier | None:
        if not path.exists():
            return None
        bundle = joblib.load(path)
        inst = cls(threshold=bundle.get("threshold", DEFAULT_THRESHOLD))
        inst._pipeline = bundle["pipeline"]
        inst._mlb = bundle["mlb"]
        return inst

    def save(self, path: Path = MODEL_PATH) -> None:
        if self._pipeline is None or self._mlb is None:
            raise RuntimeError("Nothing to save — train first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self._pipeline, "mlb": self._mlb, "threshold": self.threshold},
            path,
        )

    def predict(self, prompt: str) -> list[TagPrediction]:
        if self._pipeline is None or self._mlb is None:
            return []
        proba = self._pipeline.predict_proba([prompt])[0]
        return [
            TagPrediction(tag=tag, confidence=float(p), classifier=self.name)
            for tag, p in zip(self._mlb.classes_, proba, strict=True)
            if p >= self.threshold
        ]

    def train_from_db(self) -> dict[str, int]:
        """Pull labeled prompts from the DB and fit. Returns a per-tag count report."""
        examples = self._collect_examples()
        if not examples:
            raise RuntimeError(
                "No labeled prompts in DB. Add `human` PromptTag rows or run the "
                "heuristic classifier first to bootstrap labels."
            )

        # Drop tags with too few examples — they overfit and hurt overall scores.
        counts: dict[str, int] = defaultdict(int)
        for _, tags in examples:
            for t in tags:
                counts[t] += 1
        keep = {t for t, c in counts.items() if c >= MIN_EXAMPLES_PER_TAG}
        examples = [(text, [t for t in tags if t in keep]) for text, tags in examples]
        examples = [(text, tags) for text, tags in examples if tags]

        if not examples:
            raise RuntimeError(
                f"No tags have ≥ {MIN_EXAMPLES_PER_TAG} examples. Label more prompts."
            )

        texts = [t for t, _ in examples]
        labels = [tags for _, tags in examples]

        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(labels)

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        sublinear_tf=True,
                        strip_accents="unicode",
                    ),
                ),
                (
                    "clf",
                    OneVsRestClassifier(
                        LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
                    ),
                ),
            ]
        )
        pipeline.fit(texts, y)

        self._pipeline = pipeline
        self._mlb = mlb
        return dict(counts)

    @staticmethod
    def _collect_examples() -> list[tuple[str, list[str]]]:
        """Labels in declining order of trust: human > bench > heuristic.

        Bench tags are the category a benchmark JSONL declared outright
        (confidence 1.0) -- as trustworthy as a human label. Without this
        rung, training on a freshly-benched DB fell through to the heuristic
        guesses and the ML model inherited their blind spots: it could never
        learn "reasoning" because the heuristic has no reasoning signal, so
        the router stayed unable to tag incoming reasoning prompts even with
        4 declared examples sitting in the DB.
        """
        with get_session() as session:
            prompts = session.query(Prompt).all()
            tags_by_prompt: dict[int, list[PromptTag]] = defaultdict(list)
            for t in session.query(PromptTag).all():
                tags_by_prompt[t.prompt_id].append(t)

            examples: list[tuple[str, list[str]]] = []
            for p in prompts:
                tags = tags_by_prompt.get(p.id, [])
                human = [t.tag for t in tags if t.classifier == "human"]
                if human:
                    examples.append((p.content, sorted(set(human))))
                    continue
                bench = [t.tag for t in tags if t.classifier == "bench"]
                if bench:
                    examples.append((p.content, sorted(set(bench))))
                    continue
                heur = [t for t in tags if t.classifier == "heuristic" and t.tag != "unknown"]
                if heur:
                    best = max(heur, key=lambda t: t.confidence)
                    examples.append((p.content, [best.tag]))
            return examples

    @staticmethod
    def report_balance(counts: dict[str, int]) -> str:
        if not counts:
            return "(no labels)"
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        max_count = max(counts.values())
        lines = []
        for tag, c in items:
            bar = "█" * int(np.ceil(20 * c / max_count))
            lines.append(f"  {tag:18s} {c:5d}  {bar}")
        return "\n".join(lines)

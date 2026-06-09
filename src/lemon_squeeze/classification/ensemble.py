"""Ensemble classifier and DB-write pipeline.

Each member classifier writes its predictions independently (keyed by
`(prompt_id, tag, classifier)` so they don't collide). The ensemble's role is
to fan out the predict call, optionally invoking the LLM only when the
heuristic and ML members disagree — saving on inference cost.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sqlalchemy import select

from lemon_squeeze.classification.base import Classifier, TagPrediction
from lemon_squeeze.classification.heuristics import HeuristicClassifier
from lemon_squeeze.classification.llm import LLMClassifier
from lemon_squeeze.classification.ml import MLClassifier
from lemon_squeeze.config import settings
from lemon_squeeze.db import Prompt, PromptTag, get_session


@dataclass
class ClassifyStats:
    prompts_seen: int = 0
    prompts_classified: int = 0
    tags_written: int = 0
    tags_skipped_existing: int = 0


class EnsembleClassifier(Classifier):
    name = "ensemble"

    def __init__(
        self,
        members: Sequence[Classifier],
        *,
        llm_only_on_disagreement: bool = True,
    ) -> None:
        self.members = list(members)
        self.llm_only_on_disagreement = llm_only_on_disagreement

    def predict(self, prompt: str) -> list[TagPrediction]:
        results: list[TagPrediction] = []
        non_llm = [m for m in self.members if m.name != "llm"]
        llm_members = [m for m in self.members if m.name == "llm"]

        agreement: set[str] | None = None
        for member in non_llm:
            preds = member.predict(prompt)
            results.extend(preds)
            tags = {p.tag for p in preds if p.tag != "unknown"}
            agreement = tags if agreement is None else agreement & tags

        run_llm = bool(llm_members) and (
            not self.llm_only_on_disagreement or not agreement
        )
        if run_llm:
            for member in llm_members:
                results.extend(member.predict(prompt))
        return results


def build_default_classifier() -> EnsembleClassifier:
    members: list[Classifier] = [HeuristicClassifier()]
    ml = MLClassifier.load()
    if ml is not None:
        members.append(ml)
    if settings.classifier_llm_provider != "none":
        members.append(LLMClassifier())
    return EnsembleClassifier(members)


def resolve_classifier(name: str) -> Classifier:
    """Map a user-supplied classifier name to an instance.

    Shared by the CLI (`lemon classify ask --classifier`) and the HTTP
    server (`POST /classify`) so the two surfaces accept the same names
    and fail with the same message. Raises ValueError for an unknown name
    (usage error) and FileNotFoundError for "ml" with no trained model
    (precondition error); callers translate those to their surface's
    convention (CLI exit 2 / exit 1, HTTP 400).
    """
    if name == "heuristic":
        return HeuristicClassifier()
    if name == "ml":
        loaded = MLClassifier.load()
        if loaded is None:
            raise FileNotFoundError(
                "No trained ML classifier found; run `lemon classify train-ml` first."
            )
        return loaded
    if name == "ensemble":
        return build_default_classifier()
    raise ValueError(
        f"Unknown classifier: {name!r} (choices: heuristic, ml, ensemble)"
    )


def classify_unlabeled(
    classifier: Classifier | None = None,
    *,
    limit: int | None = None,
    only_missing_classifier: str | None = None,
) -> ClassifyStats:
    """Run the classifier over prompts in the DB and persist tag predictions.

    `only_missing_classifier`: if set, skip prompts that already have a tag from
    that classifier. Useful for re-running only the ML or LLM step.
    """
    classifier = classifier or build_default_classifier()
    stats = ClassifyStats()

    with get_session() as session:
        q = select(Prompt)
        if limit is not None:
            q = q.limit(limit)
        prompts = list(session.scalars(q).all())

        existing_index: dict[int, set[tuple[str, str]]] = {}
        for t in session.query(PromptTag).all():
            existing_index.setdefault(t.prompt_id, set()).add((t.tag, t.classifier))

        for prompt in prompts:
            stats.prompts_seen += 1
            existing = existing_index.get(prompt.id, set())
            if only_missing_classifier and any(
                c == only_missing_classifier for _, c in existing
            ):
                continue

            predictions = classifier.predict(prompt.content)
            if not predictions:
                continue
            stats.prompts_classified += 1
            for pred in _dedupe(predictions):
                if (pred.tag, pred.classifier) in existing:
                    stats.tags_skipped_existing += 1
                    continue
                session.add(
                    PromptTag(
                        prompt_id=prompt.id,
                        tag=pred.tag,
                        classifier=pred.classifier,
                        confidence=pred.confidence,
                    )
                )
                existing.add((pred.tag, pred.classifier))
                stats.tags_written += 1
    return stats


def _dedupe(predictions: Iterable[TagPrediction]) -> list[TagPrediction]:
    """Within a single classify call, keep the highest-confidence per (tag, classifier)."""
    best: dict[tuple[str, str], TagPrediction] = {}
    for p in predictions:
        key = (p.tag, p.classifier)
        if key not in best or p.confidence > best[key].confidence:
            best[key] = p
    return list(best.values())

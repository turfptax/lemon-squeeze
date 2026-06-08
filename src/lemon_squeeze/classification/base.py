"""Classifier interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TagPrediction:
    tag: str
    confidence: float
    classifier: str


class Classifier(ABC):
    """Implement `predict(prompt)` returning zero or more tag predictions."""

    name: str = "unknown"

    @abstractmethod
    def predict(self, prompt: str) -> list[TagPrediction]:
        ...

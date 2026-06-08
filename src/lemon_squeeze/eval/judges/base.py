"""Judge ABC + verdict type.

Judges are stateless: given a (prompt, response, metadata) tuple they return
a JudgeVerdict. Most judges only need (prompt, response); the optional
`metadata` parameter exists so per-prompt judges can read ground truth
attached to the Prompt row (e.g. `ExpectedContainsJudge` pulls
`metadata["expected_contains"]`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgeVerdict:
    score: float           # arbitrary numeric score (judges set their own scale)
    passed: bool | None    # whether the run is "good enough" by this judge's standard
    notes: str | None = None
    judge_model: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Judge(ABC):
    """Stateless evaluator: given a (prompt, response, metadata) tuple, return a verdict.

    Most judges ignore `metadata`. Per-prompt judges (e.g. ExpectedContainsJudge)
    read ground truth from Prompt.source_metadata at score time.
    """

    name: str = "unknown"

    @abstractmethod
    def evaluate(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        ...

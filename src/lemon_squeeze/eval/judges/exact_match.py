"""Exact-string match judge, optionally after light normalization."""
from __future__ import annotations

from lemon_squeeze.eval.judges.base import Judge, JudgeVerdict


class ExactMatchJudge(Judge):
    name = "exact_match"

    def __init__(
        self,
        expected: str,
        normalize_whitespace: bool = True,
        case_sensitive: bool = False,
    ) -> None:
        self.expected = expected
        self.normalize_whitespace = normalize_whitespace
        self.case_sensitive = case_sensitive

    def _norm(self, s: str) -> str:
        if self.normalize_whitespace:
            s = " ".join(s.split())
        if not self.case_sensitive:
            s = s.lower()
        return s.strip()

    def evaluate(
        self, prompt: str, response: str, metadata: dict | None = None
    ) -> JudgeVerdict:
        match = self._norm(self.expected) == self._norm(response)
        return JudgeVerdict(
            score=1.0 if match else 0.0,
            passed=match,
            notes=None if match else f"expected={self.expected!r}",
        )

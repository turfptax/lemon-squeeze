"""Per-prompt expected-substring judge.

Each prompt brings its own ground truth — a list of strings that must all
appear in the response, stored under `expected_contains` in its
`source_metadata`. Replaces the bench-specific `_score_expected_contains`
helper so seed prompts and AI-harness prompts can carry per-prompt scoring
too without re-implementing the pattern.

The metadata key is configurable so callers can co-exist different ground-truth
fields (e.g. `expected_keywords` vs `expected_contains`).
"""
from __future__ import annotations

from typing import Any

from lemon_squeeze.eval.judges.base import Judge, JudgeVerdict


class ExpectedContainsJudge(Judge):
    name = "expected_contains"

    def __init__(
        self,
        metadata_key: str = "expected_contains",
        case_sensitive: bool = False,
        # If the prompt doesn't carry the metadata key, what to do:
        #   "skip"  — return passed=None (signals "not applicable")
        #   "fail"  — score 0.0, passed=False
        on_missing: str = "skip",
    ) -> None:
        self.metadata_key = metadata_key
        self.case_sensitive = case_sensitive
        if on_missing not in ("skip", "fail"):
            raise ValueError(f"on_missing must be 'skip' or 'fail', got {on_missing!r}")
        self.on_missing = on_missing

    def evaluate(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> JudgeVerdict:
        expected = (metadata or {}).get(self.metadata_key)
        if not expected or not isinstance(expected, list):
            if self.on_missing == "skip":
                return JudgeVerdict(
                    score=0.0,
                    passed=None,
                    notes=f"no {self.metadata_key!r} in prompt metadata; skipped",
                )
            return JudgeVerdict(
                score=0.0,
                passed=False,
                notes=f"missing required metadata key {self.metadata_key!r}",
            )

        hits = [
            e for e in expected
            if isinstance(e, str) and self._present(e, response)
        ]
        score = len(hits) / len(expected)
        passed = len(hits) == len(expected)
        missing = [e for e in expected if e not in hits]
        return JudgeVerdict(
            score=score,
            passed=passed,
            notes=f"matched {len(hits)}/{len(expected)}: missing={missing}",
            extra={"expected_contains": expected, "hits": hits},
        )

    def _present(self, needle: str, haystack: str) -> bool:
        if self.case_sensitive:
            return needle in haystack
        return needle.lower() in haystack.lower()

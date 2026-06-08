"""Substring presence judge — does response contain expected text?

`all_of`: list of strings that must ALL appear (AND).
`any_of`: list of strings where at least ONE must appear (OR).
`none_of`: list of forbidden strings that must NOT appear.

Score = fraction of `all_of` matched (1.0 if no `all_of` given), penalized
to 0 if any `none_of` hits or no `any_of` matches.
"""
from __future__ import annotations

from collections.abc import Sequence

from lemon_squeeze.eval.judges.base import Judge, JudgeVerdict


class ContainsJudge(Judge):
    name = "contains"

    def __init__(
        self,
        all_of: Sequence[str] = (),
        any_of: Sequence[str] = (),
        none_of: Sequence[str] = (),
        case_sensitive: bool = False,
    ) -> None:
        self.all_of = list(all_of)
        self.any_of = list(any_of)
        self.none_of = list(none_of)
        self.case_sensitive = case_sensitive

    def _present(self, needle: str, haystack: str) -> bool:
        if self.case_sensitive:
            return needle in haystack
        return needle.lower() in haystack.lower()

    def evaluate(
        self, prompt: str, response: str, metadata: dict | None = None
    ) -> JudgeVerdict:
        forbidden_hits = [n for n in self.none_of if self._present(n, response)]
        if forbidden_hits:
            return JudgeVerdict(
                score=0.0,
                passed=False,
                notes=f"forbidden tokens present: {forbidden_hits}",
            )
        all_hits = [n for n in self.all_of if self._present(n, response)]
        any_hit = any(self._present(n, response) for n in self.any_of) if self.any_of else True

        score = (len(all_hits) / len(self.all_of)) if self.all_of else 1.0
        if not any_hit:
            score = 0.0
        passed = bool(all_hits == self.all_of and any_hit)
        return JudgeVerdict(
            score=score,
            passed=passed,
            notes=(
                f"all_of matched {len(all_hits)}/{len(self.all_of)}; "
                f"any_of satisfied={any_hit}"
            ),
        )

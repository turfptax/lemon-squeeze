"""Regex-match judge."""
from __future__ import annotations

import re

from lemon_squeeze.eval.judges.base import Judge, JudgeVerdict


class RegexJudge(Judge):
    name = "regex"

    def __init__(self, pattern: str, flags: str = "") -> None:
        compiled_flags = 0
        for ch in flags:
            compiled_flags |= {
                "i": re.IGNORECASE,
                "m": re.MULTILINE,
                "s": re.DOTALL,
                "x": re.VERBOSE,
            }.get(ch, 0)
        self.pattern_str = pattern
        self.pattern = re.compile(pattern, compiled_flags)

    def evaluate(
        self, prompt: str, response: str, metadata: dict | None = None
    ) -> JudgeVerdict:
        m = self.pattern.search(response)
        return JudgeVerdict(
            score=1.0 if m else 0.0,
            passed=m is not None,
            notes=f"matched: {m.group(0)[:80]!r}" if m else f"no match for /{self.pattern_str}/",
        )

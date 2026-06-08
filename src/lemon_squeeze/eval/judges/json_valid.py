"""Is the response valid JSON? Optionally with required keys."""
from __future__ import annotations

import json
import re
from collections.abc import Sequence

from lemon_squeeze.eval.judges.base import Judge, JudgeVerdict

# Strip ``` fences so judges work against models that wrap JSON in code fences.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class JsonValidJudge(Judge):
    name = "json_valid"

    def __init__(
        self,
        required_keys: Sequence[str] = (),
        allow_fenced: bool = True,
        require_object: bool = False,
    ) -> None:
        self.required_keys = list(required_keys)
        self.allow_fenced = allow_fenced
        self.require_object = require_object

    def evaluate(
        self, prompt: str, response: str, metadata: dict | None = None
    ) -> JudgeVerdict:
        text = response.strip()
        if self.allow_fenced:
            text = _FENCE_RE.sub("", text).strip()
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            return JudgeVerdict(score=0.0, passed=False, notes=f"json error: {e.msg}")

        if self.require_object and not isinstance(obj, dict):
            return JudgeVerdict(
                score=0.0,
                passed=False,
                notes=f"expected object, got {type(obj).__name__}",
            )

        if self.required_keys:
            if not isinstance(obj, dict):
                return JudgeVerdict(
                    score=0.0,
                    passed=False,
                    notes="required_keys set but parsed JSON isn't an object",
                )
            missing = [k for k in self.required_keys if k not in obj]
            score = 1.0 - (len(missing) / len(self.required_keys))
            return JudgeVerdict(
                score=score,
                passed=not missing,
                notes=None if not missing else f"missing keys: {missing}",
            )
        return JudgeVerdict(score=1.0, passed=True)

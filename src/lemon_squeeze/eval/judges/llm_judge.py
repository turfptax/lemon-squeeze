"""LLM-as-judge.

Asks a separate model to score a response 1-5 against a rubric description and
return strict JSON. Defaults to OpenRouter + a cheap Gemini Flash model — same
pattern AI Harness uses, which is why the data we imported is already
LLM-scored on the same scale.
"""
from __future__ import annotations

import json

import httpx

from lemon_squeeze.config import settings
from lemon_squeeze.eval.clients import ChatClient
from lemon_squeeze.eval.judges.base import Judge, JudgeVerdict

SYSTEM_PROMPT = """\
You are an evaluator scoring an AI assistant's response against a rubric.

Reply with ONLY JSON in this shape:
{"score": <integer 1-5>, "passed": <bool>, "reasoning": "<one short sentence>"}

Where:
  - 1 = response fails completely; 5 = response is excellent.
  - passed = true if the response is acceptable for production use of this rubric.\
"""

USER_TEMPLATE = """\
Rubric: {rubric_description}

User prompt:
\"\"\"
{prompt}
\"\"\"

Assistant response:
\"\"\"
{response}
\"\"\"\
"""

DEFAULT_PASS_THRESHOLD = 4


class LLMJudge(Judge):
    name = "llm"

    def __init__(
        self,
        rubric_description: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        pass_threshold: int = DEFAULT_PASS_THRESHOLD,
    ) -> None:
        self.rubric_description = rubric_description
        self.provider = provider or (
            "openrouter" if settings.openrouter_api_key else "lm_studio"
        )
        self.model = model or "google/gemini-2.0-flash-001"
        self.pass_threshold = pass_threshold

    def evaluate(
        self, prompt: str, response: str, metadata: dict | None = None
    ) -> JudgeVerdict:
        client = ChatClient(self.provider)  # type: ignore[arg-type]
        try:
            result = client.chat(
                self.model,
                USER_TEMPLATE.format(
                    rubric_description=self.rubric_description,
                    prompt=prompt[:4000],
                    response=response[:4000],
                ),
                system=SYSTEM_PROMPT,
                temperature=0.0,
            )
        except httpx.HTTPError as e:
            return JudgeVerdict(
                score=0.0, passed=None, notes=f"judge call failed: {e}", judge_model=self.model
            )

        parsed = _parse_json(result.text)
        if parsed is None:
            return JudgeVerdict(
                score=0.0,
                passed=None,
                notes=f"judge returned unparseable: {result.text[:120]!r}",
                judge_model=self.model,
            )
        try:
            score = float(parsed.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        passed = bool(parsed.get("passed", score >= self.pass_threshold))
        return JudgeVerdict(
            score=score,
            passed=passed,
            notes=parsed.get("reasoning"),
            judge_model=self.model,
            extra={"raw": result.text},
        )


def _parse_json(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None

"""LLM-assisted classifier.

Calls a chat-completion endpoint (LM Studio or OpenRouter) with a strict
JSON-output prompt and parses the returned tag list. Off by default — flip
`CLASSIFIER_LLM_PROVIDER` in `.env` to enable. The intended use is as a
*tiebreaker* in the ensemble: when heuristics and ML disagree, ask a model.
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from lemon_squeeze.classification.base import Classifier, TagPrediction
from lemon_squeeze.config import settings
from lemon_squeeze.db import TagTaxonomy, get_session

SYSTEM_PROMPT = """\
You are a prompt classifier. Given a user prompt, return a JSON object listing \
which categories apply, with confidences in [0, 1]. Multiple categories may apply.

Respond with ONLY valid JSON in this shape:
{"tags": [{"tag": "<one of the allowed tags>", "confidence": <float>}, ...]}\
"""

USER_TEMPLATE = """\
Allowed tags: {tags}

Prompt to classify:
\"\"\"
{prompt}
\"\"\"\
"""

CONFIDENCE_FLOOR = 0.3


class LLMClassifier(Classifier):
    name = "llm"

    def __init__(self, provider: str | None = None, model: str | None = None) -> None:
        self.provider = provider or settings.classifier_llm_provider
        self.model = model or settings.classifier_llm_model

    def predict(self, prompt: str) -> list[TagPrediction]:
        if self.provider == "none":
            return []
        allowed = self._allowed_tags()
        try:
            raw = self._call(prompt, allowed)
        except (httpx.HTTPError, ValueError):
            return []
        return self._parse(raw, allowed)

    def _allowed_tags(self) -> list[str]:
        with get_session() as session:
            return sorted(row.tag for row in session.query(TagTaxonomy).all())

    def _call(self, prompt: str, allowed: list[str]) -> str:
        if self.provider == "lm_studio":
            base = settings.lm_studio_base_url
            key = settings.lm_studio_api_key
        elif self.provider == "openrouter":
            base = settings.openrouter_base_url
            key = settings.openrouter_api_key or ""
        else:
            raise ValueError(f"Unknown LLM classifier provider: {self.provider}")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(tags=", ".join(allowed), prompt=prompt),
                },
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{base}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _parse(self, raw: str, allowed: list[str]) -> list[TagPrediction]:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Some local models wrap JSON in prose — last-ditch extraction.
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1:
                return []
            try:
                obj = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return []
        items = obj.get("tags") if isinstance(obj, dict) else None
        if not isinstance(items, list):
            return []
        allowed_set = set(allowed)
        out: list[TagPrediction] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            tag = item.get("tag")
            conf = item.get("confidence", 0.5)
            if tag not in allowed_set:
                continue
            try:
                c = float(conf)
            except (TypeError, ValueError):
                continue
            if c < CONFIDENCE_FLOOR:
                continue
            out.append(TagPrediction(tag=tag, confidence=min(max(c, 0.0), 1.0), classifier=self.name))
        return out

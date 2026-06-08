from lemon_squeeze.eval.judges.base import Judge, JudgeVerdict
from lemon_squeeze.eval.judges.contains import ContainsJudge
from lemon_squeeze.eval.judges.exact_match import ExactMatchJudge
from lemon_squeeze.eval.judges.expected_contains import ExpectedContainsJudge
from lemon_squeeze.eval.judges.json_valid import JsonValidJudge
from lemon_squeeze.eval.judges.llm_judge import LLMJudge
from lemon_squeeze.eval.judges.regex import RegexJudge

JUDGE_REGISTRY: dict[str, type[Judge]] = {
    "contains": ContainsJudge,
    "exact_match": ExactMatchJudge,
    "expected_contains": ExpectedContainsJudge,
    "json_valid": JsonValidJudge,
    "regex": RegexJudge,
    "llm": LLMJudge,
}


def build_judge(kind: str, config: dict) -> Judge:
    """Construct a judge by registry name."""
    try:
        cls = JUDGE_REGISTRY[kind]
    except KeyError as e:
        raise ValueError(
            f"unknown judge kind: {kind!r}. Known: {sorted(JUDGE_REGISTRY)}"
        ) from e
    return cls(**config)


__all__ = [
    "JUDGE_REGISTRY",
    "ContainsJudge",
    "ExactMatchJudge",
    "ExpectedContainsJudge",
    "JsonValidJudge",
    "Judge",
    "JudgeVerdict",
    "LLMJudge",
    "RegexJudge",
    "build_judge",
]

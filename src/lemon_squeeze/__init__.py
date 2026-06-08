"""Lemon Squeeze — LLM performance harness.

Public Python API. Anything in this module's `__all__` is considered stable;
internals may move. Two ways to use it:

    1. CLI: `lemon <command>` (see `lemon --help`)
    2. Library:

        import lemon_squeeze as lemon

        lemon.init_db()                              # create the SQLite schema
        lemon.classify_unlabeled()                   # tag prompts already in DB

        # Score historical runs against a rubric.
        rubric = lemon.Rubric.from_file("rubrics/contains_python_block.yaml")
        lemon.evaluate_runs(rubric)

        # Ask the router what to use for a new prompt.
        rec = lemon.recommend(
            "Write a Python function to reverse a string.",
            weights="balanced",
        )
        print(rec.picked.model_name if rec.picked else "no recommendation")

        # Head-to-head + executive summary.
        rep = lemon.compare("anthropic/claude-sonnet-4-6", "lm_studio/llama-3.1-8b")
        summary = lemon.build_report()

The DB path comes from `LEMON_DB_PATH` in your `.env` (default: `./data/lemon.db`).
"""
from __future__ import annotations

__version__ = "0.2.4"

# Core data plumbing
from lemon_squeeze.config import settings
from lemon_squeeze.db import (
    Base,
    Evaluation,
    Model,
    Prompt,
    PromptTag,
    Run,
    TagTaxonomy,
    get_session,
    init_db,
)

# Classification
from lemon_squeeze.classification import (
    Classifier,
    EnsembleClassifier,
    HeuristicClassifier,
    LLMClassifier,
    MLClassifier,
    TagPrediction,
    build_default_classifier,
)
from lemon_squeeze.classification.ensemble import classify_unlabeled

# Evaluation
from lemon_squeeze.eval.clients import ChatClient, ChatResult
from lemon_squeeze.eval.judges import (
    ContainsJudge,
    ExactMatchJudge,
    ExpectedContainsJudge,
    JsonValidJudge,
    Judge,
    JudgeVerdict,
    LLMJudge,
    RegexJudge,
    build_judge,
)
from lemon_squeeze.eval.rubric import EvalReport, Rubric, evaluate_runs
from lemon_squeeze.eval.runner import RunReport, execute_run, fanout

# Router + analytics
from lemon_squeeze.router import (
    BALANCED,
    CHEAP,
    FAST,
    PRESETS,
    SIZE_ONLY,
    ModelStats,
    Recommendation,
    RouterWeights,
    recommend,
    stats_by_tag,
)
from lemon_squeeze.compare import ComparisonReport, TagComparison, compare
from lemon_squeeze.report import (
    CoverageGap,
    Report,
    RubricFreshness,
    TagScorecard,
    build_report,
    headline_stats,
)
from lemon_squeeze.bench import BenchReport, CategoryStat
from lemon_squeeze.providers import (
    DiscoveredModel,
    list_lm_studio_models,
    list_openrouter_models,
)
from lemon_squeeze.portable import (
    ExportReport,
    ImportReport,
    export_to_dir,
    import_from_dir,
)

__all__ = [
    "__version__",
    # config + db
    "settings",
    "Base",
    "Evaluation",
    "Model",
    "Prompt",
    "PromptTag",
    "Run",
    "TagTaxonomy",
    "get_session",
    "init_db",
    # classification
    "Classifier",
    "EnsembleClassifier",
    "HeuristicClassifier",
    "LLMClassifier",
    "MLClassifier",
    "TagPrediction",
    "build_default_classifier",
    "classify_unlabeled",
    # evaluation
    "ChatClient",
    "ChatResult",
    "ContainsJudge",
    "ExactMatchJudge",
    "ExpectedContainsJudge",
    "JsonValidJudge",
    "Judge",
    "JudgeVerdict",
    "LLMJudge",
    "RegexJudge",
    "build_judge",
    "EvalReport",
    "Rubric",
    "evaluate_runs",
    "RunReport",
    "execute_run",
    "fanout",
    # router + analytics
    "BALANCED",
    "CHEAP",
    "FAST",
    "PRESETS",
    "SIZE_ONLY",
    "ModelStats",
    "Recommendation",
    "RouterWeights",
    "recommend",
    "stats_by_tag",
    "ComparisonReport",
    "TagComparison",
    "compare",
    "CoverageGap",
    "Report",
    "RubricFreshness",
    "TagScorecard",
    "build_report",
    "headline_stats",
    "BenchReport",
    "CategoryStat",
    "DiscoveredModel",
    "list_lm_studio_models",
    "list_openrouter_models",
    "ExportReport",
    "ImportReport",
    "export_to_dir",
    "import_from_dir",
]

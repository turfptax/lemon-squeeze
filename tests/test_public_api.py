"""The public API surface — re-exports from `lemon_squeeze` work as advertised."""
import lemon_squeeze as lemon


def test_top_level_classes_importable():
    # Core data plumbing
    assert lemon.Prompt is not None
    assert lemon.Model is not None
    assert lemon.Run is not None
    assert lemon.Evaluation is not None
    assert lemon.PromptTag is not None
    assert lemon.TagTaxonomy is not None
    assert lemon.get_session is not None
    assert lemon.init_db is not None
    assert lemon.settings is not None


def test_classifier_classes_importable():
    assert lemon.Classifier is not None
    assert lemon.HeuristicClassifier is not None
    assert lemon.MLClassifier is not None
    assert lemon.LLMClassifier is not None
    assert lemon.EnsembleClassifier is not None
    assert lemon.classify_unlabeled is not None
    assert lemon.build_default_classifier is not None
    assert lemon.TagPrediction is not None


def test_eval_classes_importable():
    assert lemon.ChatClient is not None
    assert lemon.ChatResult is not None
    assert lemon.Judge is not None
    assert lemon.JudgeVerdict is not None
    assert lemon.ContainsJudge is not None
    assert lemon.ExactMatchJudge is not None
    assert lemon.RegexJudge is not None
    assert lemon.JsonValidJudge is not None
    assert lemon.LLMJudge is not None
    assert lemon.build_judge is not None
    assert lemon.Rubric is not None
    assert lemon.EvalReport is not None
    assert lemon.evaluate_runs is not None
    assert lemon.execute_run is not None
    assert lemon.fanout is not None
    assert lemon.RunReport is not None


def test_router_and_analytics_importable():
    assert lemon.recommend is not None
    assert lemon.stats_by_tag is not None
    assert lemon.compare is not None
    assert lemon.build_report is not None
    assert lemon.RouterWeights is not None
    assert lemon.Recommendation is not None
    assert lemon.ModelStats is not None
    assert lemon.ComparisonReport is not None
    assert lemon.TagComparison is not None
    assert lemon.Report is not None
    assert lemon.TagScorecard is not None
    assert lemon.CoverageGap is not None
    assert lemon.BALANCED is not None
    assert lemon.PRESETS is not None


def test_version_string():
    assert isinstance(lemon.__version__, str)
    assert lemon.__version__.count(".") >= 2


def test_smoke_classification_via_public_api():
    """Use only the public API surface to classify a prompt."""
    classifier = lemon.HeuristicClassifier()
    preds = classifier.predict("Write a Python function that adds two numbers.")
    tags = [p.tag for p in preds]
    assert "coding" in tags


def test_smoke_build_report_via_public_api():
    """An empty-DB report must return without raising."""
    report = lemon.build_report()
    assert report is not None
    assert report.n_prompts == 0

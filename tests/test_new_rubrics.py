"""Newly-shipped starter rubrics parse and apply correctly."""
from pathlib import Path

from lemon_squeeze.eval.rubric import Rubric

RUBRICS_DIR = Path(__file__).resolve().parents[1] / "rubrics"


def test_no_refusal_rubric_loads():
    r = Rubric.from_file(RUBRICS_DIR / "no_refusal.yaml")
    assert r.judge_kind == "contains"
    assert "I can't" in r.judge_config["none_of"]


def test_concise_rubric_loads():
    r = Rubric.from_file(RUBRICS_DIR / "concise.yaml")
    assert r.judge_kind == "regex"
    # PyYAML correctly unescapes \\s\\S → \s\S; the pattern matches whitespace too.
    assert r.judge_config["pattern"] == r"^[\s\S]{0,600}$"


def test_factual_quality_rubric_loads_with_llm_judge():
    r = Rubric.from_file(RUBRICS_DIR / "factual_quality.yaml")
    assert r.judge_kind == "llm"
    assert r.applies_to_tags == ["qa_factual"]
    assert r.judge_config["pass_threshold"] == 4


def test_no_refusal_judge_passes_clean_response():
    """Spot-check that the wired judge actually rejects refusals."""
    from lemon_squeeze.eval.judges import build_judge

    r = Rubric.from_file(RUBRICS_DIR / "no_refusal.yaml")
    j = build_judge(r.judge_kind, r.judge_config)
    assert j.evaluate("p", "Here's the answer: 42.").passed is True
    assert j.evaluate("p", "I'm sorry, but I can't help with that.").passed is False


def test_concise_judge_rejects_long_response():
    from lemon_squeeze.eval.judges import build_judge

    r = Rubric.from_file(RUBRICS_DIR / "concise.yaml")
    j = build_judge(r.judge_kind, r.judge_config)
    assert j.evaluate("p", "Short.").passed is True
    assert j.evaluate("p", "x" * 1000).passed is False
    # \s\S handles whitespace correctly (newlines, tabs, etc.) without DOTALL.
    assert j.evaluate("p", "Short\nresponse\nwith\nlines.").passed is True


def test_bullet_list_summary_rubric_uses_pyyaml_features():
    """The bullet-list rubric exercises features the old parser couldn't:
    multi-line `description: |` block scalar + backslash regex escapes."""
    r = Rubric.from_file(RUBRICS_DIR / "bullet_list_summary.yaml")
    # Block scalar preserves newlines.
    assert "\n" in r.description
    assert "Useful for prompts" in r.description
    # Backslash escapes survive: \s, \n, \\.
    assert r.judge_kind == "regex"
    assert r"\s" in r.judge_config["pattern"]


def test_bullet_list_summary_judge_accepts_real_bullets():
    from lemon_squeeze.eval.judges import build_judge

    r = Rubric.from_file(RUBRICS_DIR / "bullet_list_summary.yaml")
    j = build_judge(r.judge_kind, r.judge_config)
    good = "Summary:\n- First point\n- Second point\n- Third point"
    bad = "This is just a single line of running prose with no bullets."
    one_bullet = "Summary:\n- only one bullet here"
    assert j.evaluate("p", good).passed is True
    assert j.evaluate("p", bad).passed is False
    assert j.evaluate("p", one_bullet).passed is False  # need ≥ 3

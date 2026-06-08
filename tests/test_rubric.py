"""Rubric loading + evaluator behavior."""
from pathlib import Path

from sqlalchemy import select

from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run, get_session
from lemon_squeeze.eval.rubric import Rubric, evaluate_runs


YAML_TEXT = """\
name: contains_python
description: Response contains python block
judge: regex
config:
  pattern: "```python"
  flags: ""
applies_to:
  tags: [coding]
"""


def test_rubric_from_yaml(tmp_path: Path):
    f = tmp_path / "rubric.yaml"
    f.write_text(YAML_TEXT, encoding="utf-8")
    r = Rubric.from_file(f)
    assert r.name == "contains_python"
    assert r.judge_kind == "regex"
    assert r.judge_config["pattern"] == "```python"
    assert r.applies_to_tags == ["coding"]


def _seed_prompt_with_run(prompt_text: str, response: str, tag: str | None = None) -> int:
    with get_session() as s:
        p = Prompt(content=prompt_text, content_hash=str(hash(prompt_text)), char_count=len(prompt_text), source="test")
        m = s.scalar(select(Model).where(Model.name == "test/model")) or Model(
            name="test/model", provider="test", local=True
        )
        if m.id is None:
            s.add(m)
        s.add(p)
        s.flush()
        run = Run(prompt_id=p.id, model_id=m.id, response=response)
        s.add(run)
        if tag:
            s.add(PromptTag(prompt_id=p.id, tag=tag, classifier="heuristic", confidence=0.9))
        s.flush()
        return run.id


def test_evaluator_writes_evaluation_and_respects_skip_existing():
    rubric = Rubric(
        name="contains_hello",
        description="x",
        judge_kind="contains",
        judge_config={"all_of": ["hello"]},
    )
    _seed_prompt_with_run("Say hi", "hello world")
    report = evaluate_runs(rubric)
    assert report.runs_evaluated == 1
    assert report.evaluations_written == 1

    # Idempotent
    report2 = evaluate_runs(rubric)
    assert report2.skipped_existing == 1
    assert report2.evaluations_written == 0


def test_evaluator_applies_to_tags_filter():
    rubric = Rubric(
        name="contains_def",
        description="x",
        judge_kind="contains",
        judge_config={"all_of": ["def "]},
        applies_to_tags=["coding"],
    )
    _seed_prompt_with_run("write a function", "def foo(): pass", tag="coding")
    _seed_prompt_with_run("summarize x", "X is a topic.", tag="summarization")
    report = evaluate_runs(rubric)
    assert report.runs_evaluated == 1
    assert report.skipped_tag_mismatch == 1


def test_evaluator_skips_runs_without_response():
    rubric = Rubric(name="any", description="x", judge_kind="contains", judge_config={"all_of": ["x"]})
    with get_session() as s:
        p = Prompt(content="t", content_hash="empty-resp", char_count=1, source="test")
        m = s.scalar(select(Model).where(Model.name == "test/model")) or Model(
            name="test/model", provider="test", local=True
        )
        if m.id is None:
            s.add(m); s.flush()
        s.add(p); s.flush()
        s.add(Run(prompt_id=p.id, model_id=m.id, response=None))
    report = evaluate_runs(rubric)
    assert report.skipped_no_response == 1
    # Confirm no eval written.
    with get_session() as s:
        assert s.scalar(select(Evaluation)) is None

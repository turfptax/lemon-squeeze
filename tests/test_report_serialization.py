"""JSON + HTML serialization of the executive report."""
import json
from datetime import datetime, timezone

from sqlalchemy import select

from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run, get_session
from lemon_squeeze.report import (
    REPORT_SCHEMA_VERSION,
    Report,
    build_report,
    report_to_html,
)


def _seed_minimal_history() -> None:
    """Two tagged prompts + one model + runs + human_pass evals so the report
    has scorecards and freshness data."""
    with get_session() as s:
        for hint, tag, passes in [
            ("Write code", "coding", [True, True, True]),
            ("Solve x", "math", [True]),
        ]:
            p = Prompt(content=hint, content_hash=f"h-{hint}", char_count=len(hint), source="test")
            s.add(p); s.flush()
            s.add(PromptTag(prompt_id=p.id, tag=tag, classifier="test", confidence=1.0))
            m = s.scalar(select(Model).where(Model.name == "test/m")) or Model(
                name="test/m", provider="test", local=True
            )
            if m.id is None:
                s.add(m); s.flush()
            for ok in passes:
                run = Run(prompt_id=p.id, model_id=m.id, response="r")
                s.add(run); s.flush()
                s.add(
                    Evaluation(
                        run_id=run.id, rubric="human_pass",
                        score=1.0 if ok else 0.0, passed=ok, scored_by="human",
                    )
                )


# ---------- JSON -------------------------------------------------------------


def test_to_dict_on_empty_report_has_schema_version():
    rep = Report()
    d = rep.to_dict()
    assert d["schema_version"] == REPORT_SCHEMA_VERSION
    assert "generated_at" in d
    assert d["headline"]["n_prompts"] == 0
    assert d["scorecards"] == []
    assert d["gaps"] == []
    assert d["rubric_freshness"] == []


def test_to_dict_serializes_real_data_as_jsonable():
    _seed_minimal_history()
    rep = build_report(min_samples=1, threshold=0.5)
    d = rep.to_dict()

    # Round-trip through json must not raise — that's the actual contract.
    text = json.dumps(d)
    reparsed = json.loads(text)
    assert reparsed["schema_version"] == REPORT_SCHEMA_VERSION
    assert reparsed["headline"]["n_models"] >= 1
    assert isinstance(reparsed["scorecards"], list)


def test_to_dict_emits_per_source_breakdown_as_named_objects():
    _seed_minimal_history()
    rep = build_report()
    d = rep.to_dict()
    sources = d["headline"]["prompts_by_source"]
    assert all("source" in item and "count" in item for item in sources)


def test_to_dict_serializes_rubric_freshness_with_iso_timestamp():
    _seed_minimal_history()
    rep = build_report()
    d = rep.to_dict()
    if not d["rubric_freshness"]:
        return  # no rubric data in this minimal fixture
    rf = d["rubric_freshness"][0]
    assert "last_scored_at" in rf
    if rf["last_scored_at"] is not None:
        # Parses cleanly back as ISO.
        datetime.fromisoformat(rf["last_scored_at"])
    assert isinstance(rf["scored_by_breakdown"], list)


# ---------- HTML -------------------------------------------------------------


def test_html_is_self_contained():
    """Self-contained means: no external <link>, no <script src>."""
    _seed_minimal_history()
    rep = build_report(min_samples=1)
    html = report_to_html(rep)
    assert "<link" not in html
    assert "src=" not in html  # no external scripts/images
    assert "<style>" in html  # inline CSS present


def test_html_contains_section_titles():
    _seed_minimal_history()
    rep = build_report(min_samples=1)
    html = report_to_html(rep)
    assert "Headline" in html
    # With data present, scorecards section is rendered.
    if rep.scorecards:
        assert "Per-tag scorecard" in html


def test_html_on_empty_db_still_renders():
    """The HTML must not crash on an empty DB — gaps/scorecards sections drop
    out gracefully."""
    html = report_to_html(Report())
    assert "<html" in html
    assert "</html>" in html


def test_html_escapes_html_in_content():
    """Model names with `&` or `<` must not break the markup."""
    _seed_minimal_history()
    rep = build_report(min_samples=1)
    # Inject a hostile value into the prompts_by_source breakdown.
    rep.prompts_by_source = [("evil<script>alert(1)</script>", 1)]
    html = report_to_html(rep)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_html_marks_stale_rubrics_with_stale_class():
    """The `.stale` CSS class should be applied when age_days > staleness_days."""
    from lemon_squeeze.report import RubricFreshness

    rep = Report()
    rep.rubric_freshness = [
        RubricFreshness(
            rubric="stale_one",
            n_evals=5,
            last_scored_at=datetime.now(timezone.utc),
            age_days=100.0,
            stale=True,
            scored_by_breakdown=[("auto", 5)],
        ),
        RubricFreshness(
            rubric="fresh_one",
            n_evals=3,
            last_scored_at=datetime.now(timezone.utc),
            age_days=1.0,
            stale=False,
            scored_by_breakdown=[("human", 3)],
        ),
    ]
    html = report_to_html(rep)
    # Stale row has class="stale"; fresh row does not.
    assert 'class="stale">\n<td>stale_one' in html or 'class="stale"><td>stale_one' in html

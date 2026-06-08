"""UTCDateTime — timestamps always come back tz-aware."""
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from lemon_squeeze.db import Evaluation, Model, Prompt, Run, get_session


def _seed_prompt_run_eval(when: datetime) -> int:
    with get_session() as s:
        m = s.scalar(select(Model).where(Model.name == "utc/test")) or Model(
            name="utc/test", provider="test", local=True
        )
        if m.id is None:
            s.add(m); s.flush()
        p = Prompt(content="t", content_hash=f"h-{when.isoformat()}", char_count=1, source="t")
        s.add(p); s.flush()
        run = Run(prompt_id=p.id, model_id=m.id, response="x")
        s.add(run); s.flush()
        s.add(
            Evaluation(
                run_id=run.id, rubric="r",
                score=1.0, passed=True, scored_by="auto",
                created_at=when,
            )
        )
        return run.id


def test_aware_datetime_in_aware_datetime_out():
    when = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    _seed_prompt_run_eval(when)
    with get_session() as s:
        e = s.scalars(select(Evaluation)).first()
    assert e.created_at.tzinfo is not None
    assert e.created_at == when


def test_naive_datetime_in_aware_utc_out():
    """Even when callers store a naive datetime, reads must return aware."""
    naive = datetime(2025, 6, 7, 9, 30, 0)
    _seed_prompt_run_eval(naive)
    with get_session() as s:
        e = s.scalars(select(Evaluation)).first()
    assert e.created_at.tzinfo is timezone.utc
    # Value is preserved (treated as UTC since it was naive).
    assert (e.created_at.replace(tzinfo=None)) == naive


def test_age_math_works_without_inline_tz_fix():
    """The whole point: no naive-vs-aware TypeError in age computations."""
    past = datetime.now(timezone.utc) - timedelta(days=10)
    _seed_prompt_run_eval(past)
    with get_session() as s:
        e = s.scalars(select(Evaluation)).first()
    # Subtract aware from aware — would crash if e.created_at were naive.
    age = datetime.now(timezone.utc) - e.created_at
    assert age.total_seconds() >= 9 * 86400

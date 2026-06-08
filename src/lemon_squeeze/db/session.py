"""Engine, session factory, and one-shot DB init."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from lemon_squeeze.config import settings
from lemon_squeeze.db.models import Base, TagTaxonomy

DEFAULT_TAXONOMY: list[tuple[str, str]] = [
    ("coding", "Write, fix, or explain code"),
    ("reasoning", "Multi-step logical or mathematical reasoning"),
    ("math", "Arithmetic, algebra, calculus, proofs"),
    ("summarization", "Condense or paraphrase given text"),
    ("extraction", "Pull structured data from unstructured text"),
    ("classification", "Assign labels to provided content"),
    ("creative", "Fiction, poetry, brainstorming, ideation"),
    ("conversation", "Open-ended chit-chat, role-play, advice"),
    ("instruction", "Follow procedural instructions, tool use"),
    ("translation", "Convert text between human languages"),
    ("qa_factual", "Direct factual question expecting a known answer"),
    ("rewrite", "Edit or rephrase given text"),
    ("planning", "Produce a plan, outline, or schedule"),
    ("unknown", "Could not be classified"),
]


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False is safe because each Session uses its own pooled
    # connection — we never share a single Connection between threads. Required
    # for the ThreadPoolExecutor in eval/runner.py:fanout to write concurrently.
    return create_engine(
        settings.db_url,
        future=True,
        connect_args={"check_same_thread": False},
    )


@lru_cache(maxsize=1)
def _sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


@contextmanager
def get_session() -> Iterator[Session]:
    session = _sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(seed_taxonomy: bool = True) -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)

    # Wire cache-invalidation hooks the first time the schema exists. Doing
    # this here (rather than at engine creation) avoids a circular import:
    # `cache` imports from `db.models`, which doesn't exist until tables are
    # registered. `install_invalidation_hooks` is idempotent.
    from lemon_squeeze.cache import install_invalidation_hooks
    install_invalidation_hooks()

    if not seed_taxonomy:
        return
    with get_session() as s:
        existing = {row.tag for row in s.query(TagTaxonomy).all()}
        for tag, desc in DEFAULT_TAXONOMY:
            if tag not in existing:
                s.add(TagTaxonomy(tag=tag, description=desc))

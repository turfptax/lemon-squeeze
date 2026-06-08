"""Pytest fixtures — isolated SQLite DB, freshly truncated for each test.

CRITICAL: the engine + sessionmaker are `@lru_cache`d, and `db/session.py`
imports `settings` by name at module load. Any prior import binds to that
original instance. Replacing `config_module.settings = Settings()` does NOT
update those local references — so callers would silently hit the production
DB `./data/lemon.db` instead of the tmp test DB. We learned this the hard way
when a `models_registered == 1` assertion was failing because the production
DB still had a stale model row from CLI usage.

Fix: MUTATE the existing settings instance in place. All modules already hold
a reference to it.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_db(tmp_path_factory: pytest.TempPathFactory):
    db_path = tmp_path_factory.mktemp("lemon") / "test.db"
    os.environ["LEMON_DB_PATH"] = str(db_path)

    from lemon_squeeze.config import settings
    from lemon_squeeze.db import session as session_module

    # Mutate in place — every other module's `settings` reference picks it up.
    settings.db_path = db_path

    # Caches were possibly populated by a previous test session or import-time
    # side effect; force a rebuild against the new path.
    session_module.get_engine.cache_clear()
    session_module._sessionmaker.cache_clear()

    from lemon_squeeze.db import init_db

    init_db()
    yield
    # Windows holds the SQLite file open via the pool; dispose first.
    session_module.get_engine().dispose()
    try:
        Path(db_path).unlink(missing_ok=True)
    except PermissionError:
        # Best-effort: pytest's tmp_path is auto-cleaned eventually anyway.
        pass


@pytest.fixture(autouse=True)
def _truncate_between_tests():
    """Wipe rows between tests so tests can't pollute each other via dedup hashes."""
    from sqlalchemy import delete

    from lemon_squeeze.db import Evaluation, Model, Prompt, PromptTag, Run, get_session

    yield
    with get_session() as s:
        # Delete in FK-safe order.
        for table in (Evaluation, Run, PromptTag, Prompt, Model):
            s.execute(delete(table))

"""`lemon db init` now auto-stamps at current Alembic head."""
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from lemon_squeeze.cli import app

runner = CliRunner()


def test_db_init_reports_stamp_in_output():
    """The CLI message should make it clear the DB was stamped too."""
    r = runner.invoke(app, ["db", "init"])
    assert r.exit_code == 0
    assert "stamped at head" in r.stdout


def test_db_init_no_stamp_skips_stamp_step():
    r = runner.invoke(app, ["db", "init", "--no-stamp"])
    assert r.exit_code == 0
    assert "no stamp" in r.stdout
    assert "stamped at head" not in r.stdout


def test_db_init_actually_stamps_against_real_db(tmp_path: Path):
    """End-to-end: after `db init`, `db current` should report a revision.

    Verifies the stamp is real — not just a printed message.
    """
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import create_engine

    from lemon_squeeze.config import settings

    db = tmp_path / "init_stamp.db"
    saved = settings.db_path
    settings.db_path = db
    try:
        # Clear engine + sessionmaker cache so the CLI picks up the new path.
        from lemon_squeeze.db import session as session_module
        session_module.get_engine.cache_clear()
        session_module._sessionmaker.cache_clear()

        r = runner.invoke(app, ["db", "init"])
        assert r.exit_code == 0

        # Inspect alembic_version table directly.
        engine = create_engine(f"sqlite:///{db}", future=True)
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn)
            current = ctx.get_current_revision()
        assert current is not None  # was None before this feature
        assert len(current) == 12   # alembic uses 12-hex revisions
    finally:
        settings.db_path = saved
        session_module.get_engine.cache_clear()
        session_module._sessionmaker.cache_clear()


def test_db_init_handles_missing_alembic_files_gracefully(tmp_path: Path):
    """When alembic.ini isn't present (wheel install scenario), init still
    succeeds — it just notes the stamp was skipped."""
    with patch("lemon_squeeze.cli._alembic_config",
               side_effect=FileNotFoundError("alembic.ini not found")):
        r = runner.invoke(app, ["db", "init"])
    assert r.exit_code == 0
    assert "Database initialized" in r.stdout
    assert "stamp skipped" in r.stdout

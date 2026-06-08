"""Doctor checks — exercise the OK and FAIL branches that the empty-DB
test in test_doctor.py doesn't reach."""
from pathlib import Path
from unittest.mock import patch

from sqlalchemy.exc import OperationalError

from lemon_squeeze.doctor import (
    _check_db_path_writable,
    _check_env_file,
    _check_ml_classifier_present,
    _check_openrouter_or_lmstudio,
    _gather_counts,
    run_all_checks,
    summarize,
)


def test_env_file_ok_when_path_exists(tmp_path: Path):
    """Happy path: when .env exists, check returns ok."""
    env_file = tmp_path / ".env"
    env_file.write_text("KEY=value")
    fake_settings = type("S", (), {"model_config": {"env_file": str(env_file)}})()
    with patch("lemon_squeeze.doctor.settings", fake_settings):
        result = _check_env_file()
    assert result.status == "ok"
    assert "loaded from" in result.detail


def test_env_file_warn_when_no_env_file_configured():
    fake_settings = type("S", (), {})()  # no model_config attr at all
    with patch("lemon_squeeze.doctor.settings", fake_settings):
        result = _check_env_file()
    assert result.status == "warn"


def test_db_path_writable_fail_when_oserror():
    """When mkdir raises OSError, the check fails with a remediation hint."""
    fake_settings = type(
        "S", (), {"db_path": Path("/this/should/not/be/writable/x.db")}
    )()
    with patch("lemon_squeeze.doctor.settings", fake_settings):
        with patch.object(Path, "mkdir", side_effect=OSError("read-only filesystem")):
            result = _check_db_path_writable()
    assert result.status == "fail"
    assert result.hint is not None


def test_openrouter_check_ok_when_real_key_set():
    fake_settings = type("S", (), {"openrouter_api_key": "sk-real-key-here"})()
    with patch("lemon_squeeze.doctor.settings", fake_settings):
        result = _check_openrouter_or_lmstudio()
    assert result.status == "ok"


def test_openrouter_check_warn_when_placeholder_key():
    fake_settings = type("S", (), {"openrouter_api_key": "your_key_here"})()
    with patch("lemon_squeeze.doctor.settings", fake_settings):
        result = _check_openrouter_or_lmstudio()
    assert result.status == "warn"


def test_ml_classifier_check_ok_when_file_exists(tmp_path: Path):
    model_path = tmp_path / "fake.joblib"
    model_path.write_text("not really a model")
    with patch("lemon_squeeze.doctor.ML_MODEL_PATH", model_path):
        result = _check_ml_classifier_present()
    assert result.status == "ok"


def test_gather_counts_falls_back_on_operational_error():
    """When the schema is missing (e.g. DB never initialized), gather_counts
    returns schema_ok=False with the error message — doesn't crash."""
    from contextlib import contextmanager

    @contextmanager
    def _bad_session():
        # SQLAlchemy's OperationalError needs three args (statement, params, orig).
        raise OperationalError("SELECT 1", {}, Exception("no such table: prompts"))
        yield  # pragma: no cover

    with patch("lemon_squeeze.doctor.get_session", _bad_session):
        counts = _gather_counts()
    assert counts.schema_ok is False
    assert counts.schema_error is not None


def test_run_all_checks_with_schema_missing_returns_fail_statuses():
    """When _gather_counts reports schema missing, the per-table checks
    propagate the fail. Summarize counts non-zero fails."""
    from lemon_squeeze.doctor import _DbCounts

    fake_counts = _DbCounts(
        schema_ok=False, schema_error="boom",
        taxonomy=0, prompts=0, tagged_prompts=0, models=0, evaluations=0,
    )
    with patch("lemon_squeeze.doctor._gather_counts", return_value=fake_counts):
        results = run_all_checks()
    ok, warn, fail = summarize(results)
    # Schema, taxonomy, prompts, models, classification, evaluations all fail.
    assert fail >= 6

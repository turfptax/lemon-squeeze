"""OpenRouter ingester — file mode and mocked-httpx live mode."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from sqlalchemy import select

from lemon_squeeze.db import Prompt, get_session
from lemon_squeeze.ingestion.openrouter import OpenRouterIngester


def _prompts() -> list[Prompt]:
    with get_session() as s:
        return list(s.scalars(select(Prompt)).all())


# ---------- File mode (offline) ---------------------------------------------


def test_file_mode_extracts_prompt_from_top_level_field(tmp_path: Path):
    f = tmp_path / "history.json"
    f.write_text(
        json.dumps([
            {
                "id": "gen-1",
                "model": "anthropic/claude-sonnet",
                "prompt": "Write a haiku about lemons.",
                "created_at": "2025-04-01T10:00:00Z",
                "total_cost": 0.001,
            },
        ]),
        encoding="utf-8",
    )
    result = OpenRouterIngester(history_file=f).run()
    assert result.inserted == 1
    p = _prompts()[0]
    assert p.content == "Write a haiku about lemons."


def test_file_mode_extracts_from_messages_field(tmp_path: Path):
    f = tmp_path / "history.json"
    f.write_text(
        json.dumps([
            {
                "id": "gen-2",
                "messages": [
                    {"role": "system", "content": "ignored"},
                    {"role": "user", "content": "User question 1."},
                    {"role": "user", "content": "Follow-up."},
                ],
            },
        ]),
        encoding="utf-8",
    )
    result = OpenRouterIngester(history_file=f).run()
    assert result.inserted == 1
    # User messages are joined.
    p = _prompts()[0]
    assert "User question 1." in p.content
    assert "Follow-up." in p.content


def test_file_mode_handles_wrapper_object(tmp_path: Path):
    f = tmp_path / "history.json"
    f.write_text(
        json.dumps({"data": [{"id": "gen-3", "input": "wrapped"}]}),
        encoding="utf-8",
    )
    result = OpenRouterIngester(history_file=f).run()
    assert result.inserted == 1


def test_file_mode_uses_user_prompt_field(tmp_path: Path):
    f = tmp_path / "history.json"
    f.write_text(
        json.dumps([{"id": "x", "user_prompt": "from user_prompt key"}]),
        encoding="utf-8",
    )
    result = OpenRouterIngester(history_file=f).run()
    assert result.inserted == 1


def test_file_mode_skips_records_without_extractable_prompt(tmp_path: Path):
    f = tmp_path / "history.json"
    f.write_text(
        json.dumps([
            {"id": "no-prompt", "tokens_prompt": 100},
            {"id": "has-prompt", "prompt": "real one"},
        ]),
        encoding="utf-8",
    )
    result = OpenRouterIngester(history_file=f).run()
    assert result.inserted == 1


def test_file_mode_preserves_metadata(tmp_path: Path):
    f = tmp_path / "history.json"
    f.write_text(
        json.dumps([
            {
                "id": "gen-meta",
                "model": "anthropic/sonnet",
                "prompt": "test",
                "tokens_prompt": 100,
                "tokens_completion": 50,
                "total_cost": 0.001,
            },
        ]),
        encoding="utf-8",
    )
    OpenRouterIngester(history_file=f).run()
    p = _prompts()[0]
    assert p.source_metadata["model"] == "anthropic/sonnet"
    assert p.source_metadata["tokens_prompt"] == 100
    assert p.source_metadata["total_cost"] == 0.001


# ---------- Since filter ----------------------------------------------------


def test_since_filter_excludes_old_records(tmp_path: Path):
    now = datetime.now(timezone.utc)
    f = tmp_path / "history.json"
    f.write_text(
        json.dumps([
            {
                "id": "recent",
                "prompt": "recent prompt",
                "created_at": now.isoformat(),
            },
            {
                "id": "old",
                "prompt": "old prompt",
                "created_at": (now - timedelta(days=30)).isoformat(),
            },
        ]),
        encoding="utf-8",
    )
    result = OpenRouterIngester(history_file=f, since=timedelta(days=7)).run()
    assert result.inserted == 1
    p = _prompts()[0]
    assert "recent" in p.content


# ---------- Timestamps ------------------------------------------------------


def test_unix_timestamp_seconds(tmp_path: Path):
    f = tmp_path / "history.json"
    f.write_text(
        json.dumps([{"id": "ts1", "prompt": "x", "created": 1736942400}]),
        encoding="utf-8",
    )
    OpenRouterIngester(history_file=f).run()
    p = _prompts()[0]
    assert p.created_at is not None
    assert p.created_at.year == 2025


def test_unix_timestamp_milliseconds(tmp_path: Path):
    f = tmp_path / "history.json"
    f.write_text(
        json.dumps([{"id": "ts2", "prompt": "x", "created": 1736942400000}]),
        encoding="utf-8",
    )
    OpenRouterIngester(history_file=f).run()
    p = _prompts()[0]
    assert p.created_at is not None
    assert p.created_at.year == 2025


# ---------- Live (mocked httpx) mode ----------------------------------------


def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def test_live_mode_no_api_key_returns_nothing():
    """If no OpenRouter API key configured AND no history file, ingester is a no-op."""
    from lemon_squeeze.config import settings

    saved = settings.openrouter_api_key
    settings.openrouter_api_key = None
    try:
        result = OpenRouterIngester().run()
        assert result.inserted == 0
    finally:
        settings.openrouter_api_key = saved


def test_live_mode_pages_through_history():
    from lemon_squeeze.config import settings

    saved = settings.openrouter_api_key
    settings.openrouter_api_key = "fake-key"
    try:
        page1 = {"data": [{"id": f"gen-{i}", "prompt": f"prompt {i}"} for i in range(3)]}
        # Page 2 is a short page → terminator.
        page2 = {"data": [{"id": "gen-3", "prompt": "last one"}]}

        client_mock = MagicMock()
        client_mock.get.side_effect = [
            _mock_response(page1),
            _mock_response(page2),
        ]
        with patch("lemon_squeeze.ingestion.openrouter.httpx.Client") as Client:
            Client.return_value.__enter__.return_value = client_mock
            result = OpenRouterIngester(page_size=3, max_pages=5).run()
        assert result.inserted == 4
    finally:
        settings.openrouter_api_key = saved


def test_live_mode_respects_max_pages():
    """Stops after max_pages even if pages keep coming back full."""
    from lemon_squeeze.config import settings

    saved = settings.openrouter_api_key
    settings.openrouter_api_key = "fake-key"
    try:
        full_page = {"data": [{"id": f"x-{i}", "prompt": f"p {i}"} for i in range(3)]}
        client_mock = MagicMock()
        client_mock.get.return_value = _mock_response(full_page)
        with patch("lemon_squeeze.ingestion.openrouter.httpx.Client") as Client:
            Client.return_value.__enter__.return_value = client_mock
            OpenRouterIngester(page_size=3, max_pages=2).run()
        # 2 pages × 3 entries each. But all have unique IDs, so 6 unique
        # inserts — except prompts get deduped by content hash. Since each
        # has a unique `prompt` value, all 6 insert.
        assert client_mock.get.call_count == 2
    finally:
        settings.openrouter_api_key = saved

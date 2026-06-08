"""LM Studio ingester — file-system + JSON-shape robustness."""
import json
from pathlib import Path

from sqlalchemy import select

from lemon_squeeze.db import Prompt, get_session
from lemon_squeeze.ingestion.lm_studio import LMStudioIngester


def _all_prompts() -> list[Prompt]:
    with get_session() as s:
        return list(s.scalars(select(Prompt)).all())


# ---------- Construction + missing directory --------------------------------


def test_ingester_with_nonexistent_dir_emits_nothing(tmp_path: Path):
    ing = LMStudioIngester(logs_dir=tmp_path / "does-not-exist")
    assert list(ing.iter_prompts()) == []
    result = ing.run()
    assert result.inserted == 0


def test_ingester_with_empty_dir(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    result = LMStudioIngester(logs_dir=tmp_path / "empty").run()
    assert result.inserted == 0


def test_ingester_with_no_logs_dir_configured_emits_nothing():
    ing = LMStudioIngester(logs_dir=None)
    # Even if LM_STUDIO_LOGS_DIR is unset, no crash.
    list(ing.iter_prompts())


# ---------- JSON shape variations -------------------------------------------


def test_messages_in_top_level_messages_key(tmp_path: Path):
    payload = {
        "id": "conv-1",
        "model": "llama-3.1-8b",
        "messages": [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ],
    }
    (tmp_path / "conv-1.json").write_text(json.dumps(payload), encoding="utf-8")
    result = LMStudioIngester(logs_dir=tmp_path).run()
    assert result.inserted == 2  # two user messages only


def test_messages_via_turns_key(tmp_path: Path):
    payload = {
        "id": "conv-turns",
        "turns": [
            {"role": "user", "content": "Question one"},
            {"role": "user", "content": "Question two"},
        ],
    }
    (tmp_path / "conv-turns.json").write_text(json.dumps(payload), encoding="utf-8")
    result = LMStudioIngester(logs_dir=tmp_path).run()
    assert result.inserted == 2


def test_messages_as_bare_list_at_top_level(tmp_path: Path):
    payload = [
        {"role": "user", "content": "bare list user message"},
        {"role": "system", "content": "ignore me"},
    ]
    (tmp_path / "bare.json").write_text(json.dumps(payload), encoding="utf-8")
    result = LMStudioIngester(logs_dir=tmp_path).run()
    assert result.inserted == 1


def test_messages_with_list_content_concatenated(tmp_path: Path):
    """LM Studio sometimes serializes content as a list of typed parts."""
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Part one. "},
                    {"type": "text", "text": "Part two."},
                ],
            }
        ]
    }
    (tmp_path / "parts.json").write_text(json.dumps(payload), encoding="utf-8")
    result = LMStudioIngester(logs_dir=tmp_path).run()
    assert result.inserted == 1
    contents = [p.content for p in _all_prompts()]
    assert any("Part one" in c and "Part two" in c for c in contents)


def test_sender_field_works_like_role(tmp_path: Path):
    payload = {
        "messages": [
            {"sender": "user", "text": "Message via sender field"},
            {"sender": "ai", "text": "ignored"},
        ]
    }
    (tmp_path / "sender.json").write_text(json.dumps(payload), encoding="utf-8")
    result = LMStudioIngester(logs_dir=tmp_path).run()
    assert result.inserted == 1


def test_invalid_json_files_skipped_without_crash(tmp_path: Path):
    (tmp_path / "good.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": "ok"}]}), encoding="utf-8"
    )
    (tmp_path / "bad.json").write_text("{ this is not json", encoding="utf-8")
    result = LMStudioIngester(logs_dir=tmp_path).run()
    assert result.inserted == 1


def test_empty_content_is_skipped(tmp_path: Path):
    payload = {
        "messages": [
            {"role": "user", "content": ""},
            {"role": "user", "content": "   "},
            {"role": "user", "content": "real prompt"},
        ]
    }
    (tmp_path / "x.json").write_text(json.dumps(payload), encoding="utf-8")
    result = LMStudioIngester(logs_dir=tmp_path).run()
    assert result.inserted == 1


# ---------- Timestamp parsing -----------------------------------------------


def test_iso_timestamp_parsed(tmp_path: Path):
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "with timestamp",
                "timestamp": "2025-01-15T12:30:00Z",
            }
        ]
    }
    (tmp_path / "ts.json").write_text(json.dumps(payload), encoding="utf-8")
    LMStudioIngester(logs_dir=tmp_path).run()
    prompts = _all_prompts()
    assert any(p.created_at is not None for p in prompts)


def test_unix_seconds_timestamp_parsed(tmp_path: Path):
    payload = {
        "messages": [
            {"role": "user", "content": "unix seconds", "timestamp": 1736942400},
        ]
    }
    (tmp_path / "ts.json").write_text(json.dumps(payload), encoding="utf-8")
    LMStudioIngester(logs_dir=tmp_path).run()
    p = _all_prompts()[0]
    assert p.created_at is not None
    assert p.created_at.year == 2025


def test_unix_milliseconds_timestamp_parsed(tmp_path: Path):
    payload = {
        "messages": [
            {"role": "user", "content": "unix ms", "timestamp": 1736942400000},
        ]
    }
    (tmp_path / "ts.json").write_text(json.dumps(payload), encoding="utf-8")
    LMStudioIngester(logs_dir=tmp_path).run()
    p = _all_prompts()[0]
    assert p.created_at is not None
    assert p.created_at.year == 2025


def test_malformed_timestamp_falls_through_to_none(tmp_path: Path):
    payload = {
        "messages": [
            {"role": "user", "content": "bad ts", "timestamp": "yesterday"},
        ]
    }
    (tmp_path / "ts.json").write_text(json.dumps(payload), encoding="utf-8")
    LMStudioIngester(logs_dir=tmp_path).run()
    p = _all_prompts()[0]
    assert p.created_at is None


# ---------- Recursive walking -----------------------------------------------


def test_recursive_directory_walk(tmp_path: Path):
    sub = tmp_path / "2026" / "01"
    sub.mkdir(parents=True)
    (sub / "deep.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": "deep prompt"}]}),
        encoding="utf-8",
    )
    result = LMStudioIngester(logs_dir=tmp_path).run()
    assert result.inserted == 1

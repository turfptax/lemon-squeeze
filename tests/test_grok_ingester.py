"""Grok export ingester — supports several JSON shapes."""
import json
from pathlib import Path

from sqlalchemy import select

from lemon_squeeze.db import Prompt, get_session
from lemon_squeeze.ingestion.grok_export import GrokExportIngester


def _prompts() -> list[Prompt]:
    with get_session() as s:
        return list(s.scalars(select(Prompt)).all())


# ---------- Per-conversation-array shape ------------------------------------


def test_array_of_conversations_each_with_messages(tmp_path: Path):
    f = tmp_path / "export.json"
    f.write_text(
        json.dumps([
            {
                "id": "conv-1",
                "messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                ],
            },
            {
                "id": "conv-2",
                "messages": [
                    {"role": "user", "content": "Q2"},
                ],
            },
        ]),
        encoding="utf-8",
    )
    result = GrokExportIngester(f).run()
    assert result.inserted == 2


def test_conversations_wrapper_object(tmp_path: Path):
    f = tmp_path / "export.json"
    f.write_text(
        json.dumps({
            "conversations": [
                {"id": "c1", "messages": [{"role": "user", "content": "Hello"}]},
            ]
        }),
        encoding="utf-8",
    )
    result = GrokExportIngester(f).run()
    assert result.inserted == 1


def test_bare_message_list(tmp_path: Path):
    """Falls back to treating the whole file as a single conversation's messages."""
    f = tmp_path / "export.json"
    f.write_text(
        json.dumps([
            {"role": "user", "content": "bare list user"},
            {"role": "assistant", "content": "ignored"},
        ]),
        encoding="utf-8",
    )
    result = GrokExportIngester(f).run()
    assert result.inserted == 1


def test_directory_of_per_conversation_files(tmp_path: Path):
    d = tmp_path / "exports"
    d.mkdir()
    (d / "a.json").write_text(
        json.dumps({"id": "a", "messages": [{"role": "user", "content": "from A"}]}),
        encoding="utf-8",
    )
    (d / "b.json").write_text(
        json.dumps({"id": "b", "messages": [{"role": "user", "content": "from B"}]}),
        encoding="utf-8",
    )
    # Non-JSON file alongside.
    (d / "readme.md").write_text("# notes", encoding="utf-8")

    result = GrokExportIngester(d).run()
    assert result.inserted == 2


# ---------- Role variations -------------------------------------------------


def test_sender_field_works(tmp_path: Path):
    f = tmp_path / "export.json"
    f.write_text(
        json.dumps([
            {"messages": [
                {"sender": "user", "text": "via sender"},
            ]},
        ]),
        encoding="utf-8",
    )
    result = GrokExportIngester(f).run()
    assert result.inserted == 1


def test_author_field_works(tmp_path: Path):
    f = tmp_path / "export.json"
    f.write_text(
        json.dumps([
            {"messages": [
                {"author": "human", "content": "via author"},
            ]},
        ]),
        encoding="utf-8",
    )
    result = GrokExportIngester(f).run()
    assert result.inserted == 1


def test_only_user_messages_kept(tmp_path: Path):
    f = tmp_path / "export.json"
    f.write_text(
        json.dumps([
            {"messages": [
                {"role": "user", "content": "keep"},
                {"role": "assistant", "content": "drop"},
                {"role": "system", "content": "drop"},
                {"role": "user", "content": "keep two"},
            ]},
        ]),
        encoding="utf-8",
    )
    result = GrokExportIngester(f).run()
    assert result.inserted == 2


# ---------- Robustness ------------------------------------------------------


def test_malformed_json_file_skipped(tmp_path: Path):
    d = tmp_path / "exports"
    d.mkdir()
    (d / "broken.json").write_text("not valid json{", encoding="utf-8")
    (d / "good.json").write_text(
        json.dumps([{"messages": [{"role": "user", "content": "ok"}]}]),
        encoding="utf-8",
    )
    result = GrokExportIngester(d).run()
    assert result.inserted == 1


def test_empty_content_skipped(tmp_path: Path):
    f = tmp_path / "export.json"
    f.write_text(
        json.dumps([
            {"messages": [
                {"role": "user", "content": ""},
                {"role": "user", "content": "valid"},
            ]},
        ]),
        encoding="utf-8",
    )
    result = GrokExportIngester(f).run()
    assert result.inserted == 1


def test_non_string_content_skipped(tmp_path: Path):
    """Content must be a string — lists/numbers are skipped."""
    f = tmp_path / "export.json"
    f.write_text(
        json.dumps([
            {"messages": [
                {"role": "user", "content": [1, 2, 3]},
                {"role": "user", "content": 42},
                {"role": "user", "content": "real"},
            ]},
        ]),
        encoding="utf-8",
    )
    result = GrokExportIngester(f).run()
    assert result.inserted == 1


# ---------- Timestamps ------------------------------------------------------


def test_timestamps_parsed_from_iso(tmp_path: Path):
    f = tmp_path / "export.json"
    f.write_text(
        json.dumps([
            {"messages": [
                {
                    "role": "user", "content": "with ts",
                    "timestamp": "2025-03-15T08:00:00Z",
                },
            ]},
        ]),
        encoding="utf-8",
    )
    GrokExportIngester(f).run()
    p = _prompts()[0]
    assert p.created_at is not None
    assert p.created_at.year == 2025

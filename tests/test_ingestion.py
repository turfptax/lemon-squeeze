import json
from pathlib import Path

from lemon_squeeze.db import Prompt, get_session
from lemon_squeeze.ingestion.claude_export import ClaudeExportIngester
from lemon_squeeze.ingestion.self_generated import SeedFileIngester, TemplateIngester


def _all_prompts() -> list[Prompt]:
    with get_session() as s:
        return list(s.query(Prompt).all())


def test_seed_file_ingest_and_dedup(tmp_path: Path):
    f = tmp_path / "seed.jsonl"
    f.write_text(
        "\n".join(
            [
                json.dumps({"prompt": "Summarize quantum computing.", "tag": "summarization"}),
                json.dumps({"prompt": "Write a haiku about lemons."}),
                json.dumps({"prompt": "Summarize quantum computing."}),  # duplicate
            ]
        ),
        encoding="utf-8",
    )
    res = SeedFileIngester(f).run()
    assert res.inserted == 2
    assert res.duplicates == 1

    # Re-running should hit duplicates for every record, no inserts.
    res2 = SeedFileIngester(f).run()
    assert res2.inserted == 0
    assert res2.duplicates == 3


def test_template_ingester_expands_cartesian():
    ing = TemplateIngester(
        templates=["Translate '{phrase}' into {lang}."],
        slots={"phrase": ["hello", "goodbye"], "lang": ["French", "German"]},
        intended_tag="translation",
    )
    items = list(ing.iter_prompts())
    contents = sorted(p.content for p in items)
    assert contents == sorted(
        [
            "Translate 'hello' into French.",
            "Translate 'hello' into German.",
            "Translate 'goodbye' into French.",
            "Translate 'goodbye' into German.",
        ]
    )
    assert all(p.metadata.get("intended_tag") == "translation" for p in items)


def test_claude_export_ingest(tmp_path: Path):
    export = tmp_path / "conversations.json"
    export.write_text(
        json.dumps(
            [
                {
                    "uuid": "conv-1",
                    "name": "Test",
                    "chat_messages": [
                        {"sender": "human", "text": "What is the capital of Australia?"},
                        {"sender": "assistant", "text": "Canberra."},
                        {
                            "sender": "human",
                            "content": [{"type": "text", "text": "Why not Sydney?"}],
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    res = ClaudeExportIngester(export).run()
    assert res.inserted == 2

    prompts = _all_prompts()
    contents = {p.content for p in prompts}
    assert "What is the capital of Australia?" in contents
    assert "Why not Sydney?" in contents

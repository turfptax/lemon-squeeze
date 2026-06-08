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


def test_template_ingester_source_refs_are_deterministic_across_processes():
    """source_ref must be stable across processes. Python's built-in `hash()`
    is randomized per-process for strings (PYTHONHASHSEED), so using it for
    source_ref made the same template produce different source_refs each run
    -- breaking the observability contract that source_refs are stable
    identifiers. Two TemplateIngesters built from identical inputs in the
    same process happen to give the same refs (same PYTHONHASHSEED), so the
    real check is: the source_ref string must NOT depend on Python's hash()."""
    import subprocess
    import sys

    code = (
        "from lemon_squeeze.ingestion.self_generated import TemplateIngester\n"
        "ing = TemplateIngester(templates=['Hello {x}.'], slots={'x': ['world']})\n"
        "for p in ing.iter_prompts():\n"
        "    print(p.source_ref)\n"
    )
    # Two child processes with DIFFERENT randomized hash seeds.
    import os
    refs = set()
    for seed in ("12345", "98765"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["PYTHONIOENCODING"] = "utf-8"
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, check=True,
        )
        refs.add(out.stdout.strip())
    assert len(refs) == 1, (
        f"source_ref differs across PYTHONHASHSEED values: {refs} -- "
        f"means TemplateIngester is using Python's process-randomized hash() "
        f"instead of a deterministic one"
    )


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

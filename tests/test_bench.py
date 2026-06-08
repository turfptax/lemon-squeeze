"""Bench loader + per-prompt expected_contains scoring."""
import json
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import select

from lemon_squeeze import bench as bench_mod
from lemon_squeeze.db import Evaluation, Model, Prompt, get_session
from lemon_squeeze.eval.clients import ChatResult


def _make_bench(tmp_path: Path) -> Path:
    d = tmp_path / "mybench"
    (d / "prompts").mkdir(parents=True)
    (d / "prompts" / "math.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "prompt": "What is 2+2?",
                        "intended_tag": "math",
                        "expected_contains": ["4"],
                    }
                ),
                json.dumps(
                    {
                        "prompt": "What is 5*5?",
                        "intended_tag": "math",
                        "expected_contains": ["25"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    (d / "prompts" / "qa.jsonl").write_text(
        json.dumps(
            {
                "prompt": "Capital of Australia?",
                "intended_tag": "qa_factual",
                "expected_contains": ["Canberra"],
            }
        ),
        encoding="utf-8",
    )
    return d


def test_load_ingests_all_jsonl_files(tmp_path: Path):
    d = _make_bench(tmp_path)
    inserted, deduped = bench_mod.load(d)
    assert inserted == 3
    assert deduped == 0

    with get_session() as s:
        prompts = list(s.scalars(select(Prompt)).all())
    # Each prompt's metadata carries `expected_contains`.
    metas = [p.source_metadata or {} for p in prompts]
    assert all("expected_contains" in m for m in metas)
    assert any(m["intended_tag"] == "math" for m in metas)


def test_run_scores_per_prompt_expected(tmp_path: Path):
    d = _make_bench(tmp_path)
    with get_session() as s:
        s.add(Model(name="local/tiny", provider="lm_studio", size_params_b=1.0, local=True))

    responses = {
        "What is 2+2?": "The answer is 4.",
        "What is 5*5?": "It is 25.",
        "Capital of Australia?": "I think it's Sydney.",  # wrong
    }

    def fake_chat(model, prompt, **kwargs):
        return ChatResult(text=responses.get(prompt, ""), tokens_in=1, tokens_out=1, latency_ms=1)

    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.side_effect = fake_chat
        report = bench_mod.run(d, max_workers=2)

    assert report.runs_succeeded == 3
    assert report.expected_evals_written == 3

    with get_session() as s:
        evals = list(
            s.scalars(
                select(Evaluation).where(Evaluation.rubric == "bench:expected_contains")
            ).all()
        )
    passed_count = sum(1 for e in evals if e.passed)
    assert passed_count == 2  # math both pass, qa fails

    # Per-category breakdown reflects this: math = 100%, qa_factual = 0%.
    by_cat = {(s.category, s.model_name): s for s in report.per_category}
    assert by_cat[("math", "local/tiny")].pass_rate == 1.0
    assert by_cat[("qa_factual", "local/tiny")].pass_rate == 0.0


def test_bench_does_not_collect_prompts_from_overlapping_filenames(tmp_path: Path):
    """Two benches with the same jsonl filenames must not see each other's
    prompts. SeedFileIngester sets source_ref = '<filename>:<idx>' (no path),
    so filtering by source_ref prefix alone collides across benches that share
    a naming convention (the starter bench has `coding.jsonl`, `math.jsonl`,
    etc — likely to be copied)."""
    bench_a = tmp_path / "bench_a"
    bench_b = tmp_path / "bench_b"
    for d, content in [(bench_a, "Prompt unique to bench A"),
                        (bench_b, "Prompt unique to bench B")]:
        (d / "prompts").mkdir(parents=True)
        (d / "prompts" / "coding.jsonl").write_text(
            json.dumps({"prompt": content, "intended_tag": "coding"}),
            encoding="utf-8",
        )

    bench_mod.load(bench_a)
    bench_mod.load(bench_b)

    ids_a = bench_mod._bench_prompt_ids(bench_a)
    ids_b = bench_mod._bench_prompt_ids(bench_b)

    # Each bench should see ONLY its own prompts.
    assert len(ids_a) == 1, (
        f"bench A grabbed bench B's prompts via filename overlap (got {len(ids_a)})"
    )
    assert len(ids_b) == 1
    assert set(ids_a).isdisjoint(set(ids_b)), (
        "bench prompt-ID sets must be disjoint when files happen to share names"
    )


def test_run_is_idempotent(tmp_path: Path):
    d = _make_bench(tmp_path)
    with get_session() as s:
        s.add(Model(name="local/tiny", provider="lm_studio", size_params_b=1.0, local=True))

    def fake_chat(model, prompt, **kwargs):
        return ChatResult(text="4 25 Canberra", tokens_in=1, tokens_out=1, latency_ms=1)

    with patch("lemon_squeeze.eval.runner.ChatClient") as cls:
        cls.return_value.chat.side_effect = fake_chat
        first = bench_mod.run(d, max_workers=2)
        second = bench_mod.run(d, max_workers=2)

    assert first.runs_attempted == 3
    assert second.runs_attempted == 0
    assert second.expected_evals_written == 0
    assert second.prompts_deduped == 3

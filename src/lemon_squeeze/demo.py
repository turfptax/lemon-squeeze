"""Zero-config offline demo of the full Lemon Squeeze pipeline.

`lemon demo` calls `run_demo()` to set up a temporary DB, seed prompts,
mock LLM responses, fan out across two fake models, score, compare, and
print router recommendations + an executive report. No external services
required.

The demo is deliberately scripted with `cheap/small-3b` getting math right
but coding wrong, while `premium/big-70b` gets everything right — so the
router's tradeoff math has something interesting to chew on (`math` should
pick the cheap model, `coding` should pick the premium one).
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch


@dataclass
class DemoResult:
    db_path: Path
    prompts_seeded: int
    runs_succeeded: int
    runs_attempted: int
    evaluations_written: int
    comparison_winner: str
    scorecards_with_pick: int


def run_demo(*, quiet: bool = False) -> DemoResult:
    """Run the full offline demo and return summary stats.

    Set `quiet=True` to suppress the step-by-step prints (useful in tests).
    Returns a `DemoResult` regardless. The temp DB is left in place for
    inspection — its path is in the return value.
    """
    log = (lambda *_a, **_k: None) if quiet else print

    # --- 1. Fresh DB ----------------------------------------------------------
    tmp = Path(tempfile.mkdtemp(prefix="lemon-demo-"))
    db_path = tmp / "demo.db"
    os.environ["LEMON_DB_PATH"] = str(db_path)
    log(f"[1] DB at {db_path}\n")

    import lemon_squeeze as lemon
    from lemon_squeeze.db.session import _sessionmaker, get_engine

    lemon.settings.db_path = db_path
    get_engine.cache_clear(); _sessionmaker.cache_clear()
    lemon.init_db()

    # --- 2. Seed prompts ------------------------------------------------------
    from sqlalchemy import select
    seed_data = [
        ("Write a Python function that returns the nth Fibonacci.", "coding"),
        ("Write a SQL query returning the top 5 customers by revenue.", "coding"),
        ("What is 27 * 49?", "math"),
    ]
    with lemon.get_session() as s:
        for content, tag in seed_data:
            p = lemon.Prompt(
                content=content,
                content_hash=f"h-{content[:30]}",
                char_count=len(content),
                source="demo",
                source_metadata={"intended_tag": tag},
            )
            s.add(p); s.flush()
            s.add(lemon.PromptTag(
                prompt_id=p.id, tag=tag, classifier="seed", confidence=1.0,
            ))
    log(f"[2] Seeded {len(seed_data)} prompts\n")

    # --- 3. Sample heuristic prediction ---------------------------------------
    classifier = lemon.HeuristicClassifier()
    sample = "Write a Python function to reverse a string."
    preds = classifier.predict(sample)
    log(f"[3] HeuristicClassifier({sample!r}):")
    for p in preds:
        log(f"      {p.tag} @ {p.confidence:.2f} ({p.classifier})")
    log()

    # --- 4. Register two models with different cost profiles ------------------
    with lemon.get_session() as s:
        s.add(lemon.Model(
            name="cheap/small-3b", provider="lm_studio", size_params_b=3.0,
            context_window=4096, local=True,
            cost_in_per_mtok=0.10, cost_out_per_mtok=0.20,
        ))
        s.add(lemon.Model(
            name="premium/big-70b", provider="anthropic", size_params_b=70.0,
            context_window=200000, local=False,
            cost_in_per_mtok=3.0, cost_out_per_mtok=15.0,
        ))
    log("[4] Registered 2 models: cheap/small-3b, premium/big-70b\n")

    # --- 5–6. Mocked fan-out -------------------------------------------------
    cheap_responses = {
        "Write a Python function that returns the nth Fibonacci.": "def fib(n): return n",
        "Write a SQL query returning the top 5 customers by revenue.": "SELECT * FROM customers",
        "What is 27 * 49?": "1323",
    }
    premium_responses = {
        "Write a Python function that returns the nth Fibonacci.":
            "def fib(n):\n    return n if n<2 else fib(n-1)+fib(n-2)",
        "Write a SQL query returning the top 5 customers by revenue.":
            "SELECT customer_id, SUM(amount) AS total FROM orders "
            "GROUP BY customer_id ORDER BY total DESC LIMIT 5",
        "What is 27 * 49?": "1323",
    }

    def fake_chat(model_name, prompt, **kwargs):
        text = (cheap_responses if "cheap" in model_name else premium_responses)[prompt]
        cost = 0.0001 if "cheap" in model_name else 0.01
        ms = 80 if "cheap" in model_name else 600
        return lemon.ChatResult(
            text=text, tokens_in=20, tokens_out=15, latency_ms=ms, cost_usd=cost,
        )

    with patch("lemon_squeeze.eval.runner.ChatClient") as Client:
        Client.return_value.chat.side_effect = fake_chat
        fanout_report = lemon.fanout(max_workers=2)
    log(f"[5] Fanout: {fanout_report.succeeded}/{fanout_report.attempted} succeeded\n")

    # --- 7. Score with a per-prompt rubric -----------------------------------
    expectations = {
        "Write a Python function that returns the nth Fibonacci.": ["def fib", "fib(n-1)"],
        "Write a SQL query returning the top 5 customers by revenue.":
            ["SELECT", "ORDER BY", "LIMIT"],
        "What is 27 * 49?": ["1323"],
    }
    with lemon.get_session() as s:
        for p in s.scalars(select(lemon.Prompt)).all():
            meta = dict(p.source_metadata or {})
            meta["expected_contains"] = expectations.get(p.content, [])
            p.source_metadata = meta

    rubric = lemon.Rubric(
        name="human_pass",
        description="per-prompt expected_contains",
        judge_kind="expected_contains",
        judge_config={"on_missing": "skip"},
    )
    er = lemon.evaluate_runs(rubric, rescored_by="human")
    log(f"[6] Scored {er.evaluations_written} evaluations\n")

    # --- 8. Compare models head-to-head --------------------------------------
    cmp = lemon.compare(
        "cheap/small-3b", "premium/big-70b",
        rubric="human_pass",
        require_significance=False,
    )
    log(f"[7] Head-to-head ({cmp.model_a} vs {cmp.model_b}):")
    for tc in cmp.per_tag:
        log(f"      {tc.tag:10s} A={tc.a_pass_rate:.0%} B={tc.b_pass_rate:.0%} -> {tc.winner}")
    log(f"    Overall winner: {cmp.overall_winner}\n")

    # --- 9. Router under three weight regimes --------------------------------
    log("[8] Router recommendations:")
    for preset in ("size", "balanced", "cheap"):
        rec = lemon.recommend(
            "Write a Python function to compute factorial.",
            threshold=0.5, min_samples=1, weights=preset,
        )
        if rec.picked:
            log(
                f"      preset={preset:10s} -> {rec.picked.model_name}  "
                f"pass={rec.picked.pass_rate:.0%}, $={rec.picked.avg_cost_usd:.5f}"
            )
        else:
            log(f"      preset={preset:10s} -> (no recommendation: {rec.reason})")
    log()

    # --- 10. Executive report ------------------------------------------------
    rep = lemon.build_report(threshold=0.5, min_samples=1)
    log("[9] Executive report:")
    log(
        f"      prompts={rep.n_prompts}  models={rep.n_models}  "
        f"runs={rep.n_runs}  evals={rep.n_evals}"
    )
    picks = 0
    for sc in rep.scorecards:
        if sc.has_qualifying:
            picks += 1
            log(
                f"      tag={sc.tag:12s} quality={sc.quality_pick} "
                f"({sc.quality_pass_rate:.0%}) cheap={sc.cost_pick} "
                f"(${sc.cost_pick_avg_cost:.5f}/run)"
            )
    log()
    log(
        f"DB lives at {db_path} — open with "
        f"`LEMON_DB_PATH={db_path} lemon db stats`"
    )

    return DemoResult(
        db_path=db_path,
        prompts_seeded=len(seed_data),
        runs_succeeded=fanout_report.succeeded,
        runs_attempted=fanout_report.attempted,
        evaluations_written=er.evaluations_written,
        comparison_winner=str(cmp.overall_winner),
        scorecards_with_pick=picks,
    )


if __name__ == "__main__":
    run_demo()

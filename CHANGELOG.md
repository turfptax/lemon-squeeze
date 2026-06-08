# Changelog

All notable changes to Lemon Squeeze. Dates are local (project lives on one machine).

## [Unreleased]

### Added

- **`lemon demo` — zero-config offline showcase**. New CLI command that walks the full pipeline (fresh DB → seed → classify → register two fake models → mocked fan-out → score → compare → router recommendations → executive report) using mocked LLM responses so it runs anywhere with no API keys. Demonstrates the core value proposition: math tag picks `cheap/small-3b` ($0.00010/run, 100% pass), coding tag picks `premium/big-70b`. Logic lives in `lemon_squeeze.demo.run_demo`; `examples/library_demo.py` now thinly delegates so the script and the CLI command produce identical output. 5 tests cover the summary dataclass, the persisted DB, the CLI invocation, the scorecard outcome, and that the library script still works.

## [0.2.2] — 2026-06-08

Five small UX wins after a doc-as-workaround scan: staleness detection, auto-classify after ingest, auto-backfill after train-ml, auto-stamp on db init, and the final ingest dry-run for symmetry. Each one closes an "and then also" line from the onboarding docs.

### Added

- **Rubric-hash staleness detection.** `Evaluation.rubric_hash` (new nullable column) stores SHA-256 of `(judge_kind, judge_config, applies_to_tags)` — deliberately excluding the rubric's description so editing prose doesn't invalidate scores. When `lemon eval score` runs, existing evals whose stored hash differs from the current rubric's hash are auto-deleted and re-scored (`stale re-scored: N` in the CLI summary). Up-to-date evals are still skipped; legacy rows with NULL hash are treated as up-to-date so existing DBs aren't churned. `lemon eval replay` still works as the unconditional wipe. Migration `b2f1a7c4d5e8` adds the column via `op.batch_alter_table`; fresh installs get it via `create_all()`. 12 tests cover hash determinism, what changes invalidate it, the auto-replace flow, NULL-hash safety, and `replace_existing` precedence.

### Changed

- **Every ingest command now auto-classifies newly-inserted prompts via the heuristic classifier.** Applies to `lemon ingest seed / claude / grok / openrouter / lm-studio / ai-harness` and `lemon bench load`. The heuristic is dep-free and idempotent (uses `only_missing_classifier="heuristic"` so re-runs are no-ops). Pass `--no-classify` to opt out. Eliminates the "ingest, *then also* run `classify run`" pattern that every onboarding doc had to teach.
- **`lemon classify train-ml` now auto-backfills ML tags** on prompts that don't have them yet. Without this, the freshly-trained model sat idle until the next `classify run` reached each prompt — the previous QUICKSTART told users to remember `lemon classify run --only-missing ml` after every train. Pass `--no-backfill` to inspect the trained model before applying it. QUICKSTART/TUTORIAL updated to drop the redundant explicit step.
- **`lemon db init` now auto-stamps at the current Alembic head.** Previously users had to run `lemon db init` then `lemon db stamp head` separately to get correct migration tracking — and even QUICKSTART/TUTORIAL documented the two-step. Now it's one command. Pass `--no-stamp` to opt out if you want to manage migrations manually. Gracefully degrades when alembic.ini isn't present (wheel install scenario) — init still succeeds with a note. QUICKSTART + TUTORIAL updated to drop the redundant `db stamp head` line.

### Added

- **`lemon ingest ai-harness --dry-run`** — completes the dry-run symmetry across all 6 ingest subcommands. `AIHarnessImporter.run(dry_run=True)` does the full multi-table import work inside a session, then calls `session.rollback()` before the contextmanager's commit — so the in-memory graph stays valid for the counters but nothing persists to the 4 tables (prompts, models, runs, evaluations). Verified end-to-end against real AI Harness data: dry-run reports `would-import: 20` prompts/models/runs/evals; real run reports identical numbers. 2 new tests cover the multi-table rollback + dry-run/real-run count consistency.

## [0.2.1] — 2026-06-07

Polish + three small CLI features closing library/CLI parity gaps. No architectural changes from 0.2.0.

### Added

- **`lemon ingest <X> --dry-run`** on every ingest subcommand (tick 28). Preview what would be inserted vs deduped without writing to the DB. `Ingester.run(dry_run=True)` exposes the same at the library level. Output is prefixed with `(dry-run)` and the count column is labeled `would-insert` instead of `inserted` so it's unmistakable. 7 tests cover library + CLI + intra-batch dup counting + flag-presence on every subcommand.
- **`lemon classify ask "<prompt>"` — one-shot classification** (tick 27). Takes a prompt and prints the predicted tags + confidences from the heuristic / ML / ensemble classifier — without writing to the DB. `--classifier {heuristic,ml,ensemble}` picks which to use; `--json` produces machine-parseable output; `--top N` truncates. Same library/CLI parity argument as `lemon judge`.
- **`lemon judge <rubric>` — ad-hoc scoring** (tick 26). Takes a rubric + `--prompt` + `--response` (or `--response-file`) and prints the verdict without touching the DB. Useful for "is this response acceptable under my rubric?" without the full ingest → register → run → score pipeline. Supports per-prompt rubrics via `--metadata '{"expected_contains": ["..."]}'`. 8 tests cover happy + skip + fail + invalid-metadata + no-DB-write properties.
- **`ARCHITECTURE.md`** — module map, data flow diagram, 20 documented design decisions with reasoning, extension-points table, and known limitations. Companion to `CHANGELOG.md` for new contributors.
- **`examples/TUTORIAL.md`** — "Should I switch from Sonnet to Haiku?" narrative walkthrough using the actual workflow (load → compare → cost analysis → router → operationalize via `lemon serve`). Includes a decision template and instructions for bringing your own data.
- **`examples/scripts/refusal_analysis.py`** — second offline demo: find prompts where a strict model refuses too often. Verified end-to-end (62% vs 12% refusal; router picks the moderate model for borderline cases).
- **15 new ML classifier tests** — full train → save → load → predict cycle, insufficient-data + class-imbalance fallbacks, label-source preference (human > heuristic > unknown filtered out). `classification/ml.py` 43% → 100%.
- **8 new doctor edge-case tests** — covers OK/FAIL branches the empty-DB test couldn't reach. `doctor.py` 83.3% → 96.1%.

### Changed (cleanups from tick-19 simplify pass)

- **`utils.split_provider_family()`** replaces 3 near-identical implementations across `cli.py`, `providers.py`, and `ingestion/ai_harness.py`. Now correctly handles bare names (`llama-3.1-8b-instruct` → family `llama`) — a test caught this when the first version dropped them.
- **`/metrics` endpoint** reuses `report.headline_stats()` instead of running 4 inline `COUNT(*)` queries. Picked up `runs_with_error` and `total_cost_usd` for free.
- **`RouterWeights.from_preset_and_overrides()`** unifies CLI and HTTP server preset-merge logic — "unknown preset" validation lives in one place.
- **`eval/rubric.py:evaluate_runs`** scoped its existing-eval query to `(rubric, run_ids)`. Was loading every `Evaluation` row in the DB just to check membership; real perf win on populated databases.
- **`html.escape`** (stdlib) replaces the custom `_h` helper in `report.py`.
- **`cache.py` `after_flush`** hook collapsed from 3 identical loops over `session.new/dirty/deleted` to one `itertools.chain` + `any()`.
- **Dead `TagScorecard.cost_pick_pass_rate` field** removed — no consumer.

### Fixed

- **`providers.py`** crashed on OpenRouter records whose `pricing` field wasn't a dict (one bad upstream record killed the whole listing). Caught by reviewer agent. Defensive `context_length` parsing now accepts int, float, and numeric string.
- **`portable.py`** silently coerced `char_count=0` to `len(content)` on import. Distinct from None, now preserved correctly.
- **CLI `route pick --preset typo`** silently fell back to "size" instead of erroring. Now exits with a clear message listing known presets.

### Documentation

- **Drift checks** on `ARCHITECTURE.md`, `examples/TUTORIAL.md`, and `QUICKSTART.md` against the real CLI/code. Caught and fixed 8 small drifts total: table count (5→6), smallest-ingester citation, smallest-judge citation, API-key-required callout, model-name format, `sig` glyph meaning, doctor "8+ OK" claim, dashboard "Router playground" → "Router".
- **README** links 5 onboarding paths (QUICKSTART, TUTORIAL, library_demo, refusal_analysis, ARCHITECTURE).

### Tests

- 289/289 passing. Coverage 76.5% → 86.5% across this stretch.
- 4 test-quality fixes from the simplify pass: strict exit-code on `test_doctor_runs_all_checks`, guaranteed prediction on `test_predict_returns_confidence_scores_after_training`, patch-assertion on `test_providers_list_handles_unreachable_gracefully`, glyph-independent bar test.

### Deferred

- Live LM Studio end-to-end — local server not running for the duration of this stretch.

## [0.2.0] — 2026-06-07

The "production-grade" milestone. The project moves from a scaffold + library to **library + CLI + dashboard + HTTP API + data portability + observability + migrations + caching + provider discovery** — with 85.9% test coverage and 281 tests.

### Added (in 18 /loop iterations)

**Tick 1 — Foundation**
- SQLite + SQLAlchemy schema (`prompts`, `prompt_tags`, `models`, `runs`, `evaluations`, `tag_taxonomy`)
- Five-source ingestion (`LM Studio`, `Claude export`, `Grok export`, `OpenRouter`, `self_generated`)
- Heuristic + ML (sklearn TF-IDF + LogReg) + LLM-assist classifier ensemble
- Typer CLI entry point (`lemon`)

**Tick 2 — AI Harness importer**
- `lemon ingest ai-harness` writes prompts + models + runs + evaluations from a sibling AI Harness SQLite

**Ticks 3–5 — Eval layer, parallel fanout, bench, weighted router, dashboard, doctor, public Python API**
- `eval/` module: `ChatClient`, `Judge` ABC + 6 concrete judges (contains, exact_match, regex, json_valid, llm, expected_contains), `Rubric` system, run executor with `ThreadPoolExecutor`
- 30-prompt starter benchmark across 7 categories with per-prompt `expected_contains` ground truth
- Multi-criteria router with size/cost/latency/balanced presets and tunable weights
- Streamlit dashboard with 6 tabs (Overview, Heatmap, Runs, Router, Compare, Report)
- `lemon doctor` — 10 health checks with remediation hints
- Public Python API: `import lemon_squeeze as lemon` re-exports the full surface

**Ticks 6–10 — Compare, replay, freshness, cost-per-pass, aggregations, judge generalization, providers, sample rubrics**
- `lemon compare A B` — per-tag head-to-head with **95% Wilson confidence intervals** (winner requires non-overlapping CIs)
- `lemon eval replay` — clean re-scoring of historical runs against an updated rubric
- `lemon report` — three-section executive summary (headline + per-tag scorecard + coverage gaps) + per-rubric freshness tracking
- `bench.CategoryStat.cost_per_pass` derived property — efficiency metric
- Unified `aggregations.py` — single SQL GROUP BY backing router/compare/bench/dashboard (caught a real double-counting bug from multi-classifier tagging during the refactor)
- Per-prompt `ExpectedContainsJudge` reading ground truth from metadata; deleted bench's 50-line special case
- `UTCDateTime` SQLAlchemy TypeDecorator — fixes naive-vs-aware datetime once for every column
- `lemon providers list/sync` — auto-discover LM Studio + OpenRouter models, register locally-loaded models in one command
- 3 more starter rubrics (`no_refusal`, `concise`, `factual_quality`)

**Ticks 11–13 — Export/import, JSON+HTML reports, HTTP server**
- `lemon export <dir>` / `lemon import <dir>` — JSONL round-trip with foreign rows keyed by natural identity. Idempotent re-import. Verified disaster-recovery on real data.
- `lemon report --json` / `--html` — stable schema_version=1 JSON; self-contained inline-CSS HTML
- `lemon serve` — FastAPI HTTP API with 6 endpoints: `/healthz`, `/models`, `/route`, `/classify`, `/report`, `/compare` (behind `[server]` extra)
- `QUICKSTART.md` — 15-minute walkthrough from clone to live API

**Ticks 14–17 — Caching, /metrics, PyYAML, coverage push, Alembic**
- Process-local LRU+TTL cache for the hot aggregation path. SQLAlchemy `after_flush` invalidation. Real win: identical /route calls hit 4/5 = 80% on the cache.
- `/metrics` endpoint: DB counts, per-path request counts, cache hit/miss stats
- Swapped home-rolled YAML parser for PyYAML — block scalars and backslash escapes now work
- 38 new ingester tests + offline `examples/library_demo.py`
- 37 mocked-httpx tests for ChatClient + LLMClassifier + LLMJudge
- **Alembic scaffolding** — `alembic.ini`, `env.py` reading `settings.db_url`, initial autogenerated revision `dd2bf37a86ee` for the schema, batch-mode for SQLite. New CLI: `db upgrade / downgrade / current / stamp`

**Tick 18 — ML classifier tests + this version bump**
- 15 ML classifier tests covering full train → save → load → predict cycle, insufficient-data + class-imbalance fallbacks, label-source preference (human > heuristic > unknown filtered out)
- Coverage: `ml.py` 43% → **100%**, overall **85.9%**, 281 tests

### Performance

- Parallel `fanout` via `ThreadPoolExecutor` — configurable workers, thread-safe report aggregation
- Aggregation cache keeps `/route` zero-DB after the first call in a steady state

### Architectural decisions worth remembering

- `db/session.py` uses `expire_on_commit=False`; `check_same_thread=False` for the cross-thread fanout
- `db/types.py:UTCDateTime` re-applies UTC on read (SQLite ignores `timezone=True`)
- Foreign-row identity in exports: `Prompt.content_hash`, `Model.name`, `Run._export_id` (UUID persisted in `run_metadata`)
- Bug caught + regression-tested in tick 8: `DISTINCT (prompt_id, tag)` subquery in `aggregate_by_tag_model` to avoid double-counting when multiple classifiers tag the same prompt with the same tag

### Bugs caught by tests (post-hoc)

- `tests/conftest.py` was silently truncating the production `data/lemon.db` because `db/session.py` imported `settings` by name at module load. Fixed in tick 5 by mutating `settings.db_path` in place.
- `providers.py` crashed on OpenRouter records whose `pricing` field wasn't a dict (one bad upstream record killed the whole listing). Caught by reviewer agent in tick 12.
- `portable.py` silently coerced `char_count=0` to `len(content)` on import. Caught by reviewer agent in tick 12.
- CLI `--preset typo` silently fell back to "size" instead of erroring. Caught while writing CLI smoke tests in tick 15.

## [0.1.0] — 2026-05-23

Initial scaffolding (see project memory for details).

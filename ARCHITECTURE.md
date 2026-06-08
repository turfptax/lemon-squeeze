# Architecture

A design-decisions doc for future contributors (and future me). Captures the *why* behind the project shape, written after 21 iterations of evolution. Reading order: skim "Module map", then read "Key decisions" top-to-bottom — each decision is one paragraph and self-contained.

For the *what* (commands, library API), see [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md). For a worked example of the design in action, see [examples/TUTORIAL.md](examples/TUTORIAL.md).

---

## Module map

```
src/lemon_squeeze/
├─ db/                      data layer — SQLAlchemy ORM, UTCDateTime, session factory
│  ├─ models.py             6 tables: prompts, prompt_tags, tag_taxonomy, models, runs, evaluations
│  ├─ session.py            engine + sessionmaker + init_db
│  └─ types.py              UTCDateTime TypeDecorator
├─ ingestion/               pull prompts from external sources, all behind one ABC
│  ├─ base.py               Ingester ABC + IngestResult
│  ├─ ai_harness.py         sibling project SQLite (uniquely also imports runs + evals)
│  ├─ lm_studio.py          local conversation logs
│  ├─ claude_export.py      Claude.ai data export
│  ├─ grok_export.py        Grok export
│  ├─ openrouter.py         API or pre-downloaded JSON
│  └─ self_generated.py     seed JSONL + template expansion
├─ classification/          tag prompts (heuristic + ML + LLM ensemble)
│  ├─ heuristics.py         regex/keyword scoring; deterministic baseline
│  ├─ ml.py                 TF-IDF + LogReg, joblib persistence
│  ├─ llm.py                LLM-as-classifier, OpenAI-compatible
│  └─ ensemble.py           fans out; LLM only fires on heuristic↔ML disagreement
├─ eval/                    run prompts, score responses
│  ├─ clients.py            ChatClient — OpenAI-compatible (LM Studio + OpenRouter)
│  ├─ runner.py             execute_run + fanout (ThreadPoolExecutor)
│  ├─ judges/               Judge ABC + 6 concrete: contains, exact_match, regex,
│  │                        json_valid, llm, expected_contains
│  └─ rubric.py             Rubric = (name, judge, config); YAML-loadable
├─ aggregations.py          single GROUP BY backing router/compare/bench/dashboard
├─ router.py                weighted multi-criteria recommendation
├─ compare.py               head-to-head with Wilson 95% CIs
├─ report.py                executive summary → terminal/JSON/HTML
├─ bench.py                 packaged benchmarks (load + run + score + report)
├─ portable.py              export/import the DB as JSONL
├─ cache.py                 process-local LRU+TTL with ORM event invalidation
├─ providers.py             discover models from LM Studio + OpenRouter
├─ doctor.py                10 setup health checks
├─ server.py                FastAPI HTTP API (optional [server] extra)
├─ dashboard.py             Streamlit (optional [dashboard] extra)
├─ stats.py                 Wilson interval; no SciPy dep
├─ utils.py                 hash_prompt, count_tokens, split_provider_family
└─ cli.py                   Typer; 17 subcommands wiring all of the above
```

---

## Data flow

```
                                ┌─────────────────────┐
                                │   ingestion/*       │
                                │   (LM Studio, etc.) │
                                └──────────┬──────────┘
                                           ▼
┌──────────────────┐              ┌────────────────┐              ┌──────────────────┐
│  benchmarks/*    │ ────────────►│   prompts      │              │   models         │
│  prompts/*.jsonl │              │   (deduped)    │◄──┐          │   (registered    │
└──────────────────┘              └────────┬───────┘   │          │    or sync'd)    │
                                           │           │          └────────┬─────────┘
                                           │ classify  │ ingest             │
                                           ▼           │                    ▼
                                  ┌────────────────┐   │           ┌─────────────────┐
                                  │  prompt_tags   │   │           │   runs          │
                                  │  (3 sources)   │   │           │   (executor)    │
                                  └────────┬───────┘   │           └────────┬────────┘
                                           │           │                    │ score
                                           │           │                    ▼
                                           │           │           ┌─────────────────┐
                                           │           │           │  evaluations    │
                                           │           └───────────│  (rubrics +     │
                                           │                       │   judges)       │
                                           ▼                       └────────┬────────┘
                                  ┌──────────────────────────────────────────┘
                                  ▼
                          ┌──────────────────┐
                          │  aggregations    │  ◄── one GROUP BY, all consumers
                          └────────┬─────────┘
                                   │
            ┌──────────────────┬───┴──────────────┬─────────────────┐
            ▼                  ▼                  ▼                 ▼
       ┌─────────┐       ┌──────────┐      ┌──────────┐      ┌──────────┐
       │ router  │       │ compare  │      │  bench   │      │  report  │
       └─────────┘       └──────────┘      └──────────┘      └──────────┘
            │                  │                  │                 │
            ▼                  ▼                  ▼                 ▼
       lemon route        lemon compare      lemon bench       lemon report
       lemon serve        HTTP /compare      output            HTTP /report
       HTTP /route                                             --json / --html
```

---

## Key decisions

### 1. Five tables, deduped by content_hash

`prompts` is the canonical record. `Prompt.content_hash` is SHA-256 of the normalized content; ingestion dedupes against it across all sources. The same prompt ingested from Claude export AND AI Harness collapses to one row, with `runs` accumulating against it from both contexts. Without this, the router's pass-rate math would double-count.

### 2. `prompt_tags` is many-classifiers, not one

Multiple classifiers (heuristic, ml, llm) can independently tag the same prompt. We store all three votes — the schema is `(prompt_id, tag, classifier)` unique. This lets us A/B taggers later and answer questions like "what does ML disagree with heuristic about". The downside: a naive JOIN inflates counts. We fixed this in tick 8 with `DISTINCT (prompt_id, tag)` subqueries in `aggregations.aggregate_by_tag_model`. Regression test is in `tests/test_aggregations.py::test_aggregate_by_tag_model_dedupes_multi_classifier_tags`.

### 3. Runs and evaluations are separate tables

A run is one (prompt, model) call. An evaluation is one (run, rubric) score. This separation means:
- The same run can be scored by many rubrics (including future ones via `lemon eval replay`).
- The same prompt can be re-run under different sampling params (`temperature`, etc.) without losing history.
- Cost/latency live on the run; quality lives on the eval — so cost-per-pass math is a join, not a single row's columns.

### 4. SQLite with SQLAlchemy 2.x — explicitly future-proofed

SQLite is the right default: zero setup, single file, easy backup. SQLAlchemy 2.x gives a clean migration path to Postgres if the DB outgrows itself. We use `expire_on_commit=False` and `check_same_thread=False` so the ThreadPoolExecutor in `fanout` can share sessions cheaply; each thread still uses its own pooled connection. Alembic migrations are scaffolded for future schema changes.

### 5. `UTCDateTime` in the data layer, not in callers

SQLite returns naive datetimes even when the column is declared `DateTime(timezone=True)`. We have a `db/types.py:UTCDateTime` TypeDecorator that coerces naive→UTC on write and re-applies `tzinfo=utc` on read. Lift the fix into the data layer so callers (report, dashboard) don't redo it. The simplify reviewer flagged this exact altitude issue and we moved on it in tick 9.

### 6. Ingestion behind one `Ingester` ABC, but AI Harness is special

The `Ingester` ABC emits `RawPrompt` items and base-class persistence handles the rest. This works for 5 of 6 sources. **AI Harness is different**: it emits prompts AND runs AND evaluations all at once (it's a sibling project with the same shape we want). So `AIHarnessImporter` is its own class, not an `Ingester`. Forcing it into the ABC would distort both.

### 7. Classifier ensemble with LLM as tiebreaker

The classifier interface is `predict(prompt) -> list[TagPrediction]`. The ensemble calls heuristic + ML always, then only fires the LLM-based classifier on disagreement (default; configurable). This saves cost. Tags go into `prompt_tags` with the classifier name preserved, so we can later filter "things only ML tagged" vs "things three classifiers agreed on."

### 8. The judge interface accepts optional `metadata` — for per-prompt ground truth

Most judges only need `(prompt, response)`. But some need per-prompt ground truth — e.g., `expected_contains` lives in `Prompt.source_metadata`. We extended `Judge.evaluate(prompt, response, metadata=None)` so per-prompt judges can read it. Tick 9 generalized the bench-only special case (`_score_expected_contains`) into a standard `ExpectedContainsJudge` that any rubric (including `rubrics/per_prompt_expected.yaml`) can reference.

### 9. Aggregations module is the single source of truth for per-bucket math

Before tick 8, four call sites computed pass_rate / avg_score / avg_cost / avg_latency independently (router, compare, bench, dashboard). They drifted. We consolidated into `aggregations.py:aggregate_by_tag_model` (SQL GROUP BY backing all four). Adding a new metric (`tokens_per_pass`, p50 latency) is now a one-line change on `Aggregate`'s dataclass. Bench has a second entry point (`aggregate_by_intended_tag_model`) because it groups by a JSON field, not a tagged column — JSON GROUP BY is awkward across SQLite/Postgres, so we bucket in Python after one query.

### 10. Router scoring is weighted, normalized, presets-driven

The router doesn't pick the absolute best — it picks the **smallest qualifying** by default. Weights `(size, cost, latency)` normalize via min-max across candidates then weighted-sum. Four presets (`size`/`balanced`/`cheap`/`fast`) cover common cases; `RouterWeights.from_preset_and_overrides(preset, size=None, cost=None, latency=None)` lets CLI and server share preset validation. If no model qualifies under the threshold, we fall back to the highest-pass-rate candidate and flag `fallback=True`.

### 11. Compare uses Wilson confidence intervals

A point estimate of "100% vs 50% over 3 runs" is misleading — small samples produce wide CIs. `compare` computes a Wilson 95% CI per model per tag, and a model is only declared the winner when its CI doesn't overlap the other's. The `--no-significance` flag falls back to the old point-estimate behavior. This is the single most important guard against over-fitting decisions on noisy data.

### 12. Caching with ORM event invalidation, not TTL alone

The hot path `aggregate_by_tag_model` is called by every `/route` request. A naive LRU would serve stale data forever after a new evaluation. We attach a SQLAlchemy `after_flush` event to the sessionmaker; any insert/update/delete on a watched table bumps a version counter. Cached entries below the current version are treated as stale. The TTL is purely a safety net for code that bypasses the ORM (raw `Connection.execute`). Net effect: typical `/route` traffic gets ~80% cache hit, but writes are immediately visible.

### 13. Three render layers (rich, Streamlit, HTML) deliberately separate

Trying to share rendering across CLI tables, Streamlit dataframes, and self-contained HTML produces an abstraction that obscures more than it saves. Each layer has its own formatter. They consume the same dataclasses (`Recommendation`, `Report`, `ComparisonReport`) so the data is shared, just not the presentation.

### 14. HTTP API mirrors the CLI 1:1

`POST /route` runs the same code path as `lemon route pick`. `GET /report` returns the same `Report.to_dict()` the CLI's `--json` flag produces (`schema_version=1`). This means once you've tested one surface, you've tested the contract for the other. Adding a new endpoint is mostly wiring; the work is in the library.

### 15. Three render outputs for `report` — terminal, JSON, HTML

The same `Report` object renders to terminal (rich), JSON (stable shape, schema_version=1, machine-readable), and self-contained HTML (inline CSS, no external assets, safe to email). Adding a new sink is a render-only change. The data flow stops once `build_report()` returns.

### 16. Export/import is round-trip-safe by natural identity, not surrogate keys

Exports key foreign rows by their natural identity: `Prompt.content_hash`, `Model.name`, `Run._export_id` (UUID persisted in `run_metadata`). This means an export from machine A imports cleanly into machine B even though the integer PKs differ. The UUID for Run is generated on first export and stored, so subsequent exports recognize the same row and re-imports dedupe. Verified end-to-end: wipe DB → import → all 47/20/55/130 rows restored exactly.

### 17. Provider discovery, not just registration

`lemon providers list` hits `/v1/models` on LM Studio + OpenRouter. `lemon providers sync` auto-registers everything LM Studio reports — the user doesn't have to type `models register` 5 times to test 5 local models. OpenRouter sync is opt-in because their catalog is 200+ models.

### 18. The starter benchmark ships with per-prompt ground truth

`benchmarks/starter/prompts/*.jsonl` files have `expected_contains` per line — the substring(s) a correct response must contain. This makes scoring deterministic, doesn't need an LLM judge, and produces reproducible numbers across machines. For prompts where ground truth is harder (summarization quality, factual accuracy), shipped rubrics use the LLM judge instead.

### 19. PyYAML for rubric loading — after we hit the home-rolled parser's limits

For 15 ticks we had a 70-line hand-rolled YAML parser to avoid a dep. Two rubrics hit its limits (no block scalars, no backslash escapes). Swap to PyYAML was a 30-line diff and gained features for free. Lesson: defer dep additions until the limit is *actually* painful, not theoretically.

### 20. Tests are the safety net for refactors

281 tests at 86.5% coverage isn't decoration — it caught real regressions during the tick-19 simplify pass (the helper `split_provider_family` initially dropped family extraction for bare names, the test instantly red-flagged it). When the project grew, the test suite let us refactor confidently. The investment paid back the same iteration it was made.

---

## Extension points

If you want to extend Lemon Squeeze, here are the shapes to follow:

| To add a... | Do this | See |
|---|---|---|
| **New prompt source** | Subclass `Ingester`; yield `RawPrompt`; register CLI command. | `ingestion/claude_export.py` is the smallest example |
| **New classifier** | Subclass `Classifier`; implement `predict(prompt) -> list[TagPrediction]`. The ensemble picks it up. | `classification/heuristics.py` |
| **New judge** | Subclass `Judge`; implement `evaluate(prompt, response, metadata=None)`; add to `JUDGE_REGISTRY`. | `eval/judges/regex.py` is the smallest |
| **New aggregation metric** | Add a `@property` to `Aggregate`. Router/compare/bench/dashboard pick it up. | `aggregations.py:Aggregate.cost_per_pass` |
| **New router preset** | Add to `router.PRESETS`. | `router.py:BALANCED` |
| **New scoring rubric** | YAML file in `rubrics/`; use any registered judge. | `rubrics/contains_python_block.yaml` |
| **New CLI command** | Add to `cli.py`; reuse library functions. | `cli.py:cmd_compare` |
| **New HTTP endpoint** | Add to `server.create_app()`; reuse library functions. | `server.py` /report endpoint |
| **New report section** | Extend `Report` dataclass; populate in `build_report`; render in `report_to_html`. | `report.py:RubricFreshness` was added this way |

---

## Known limitations

- **Single-machine.** Cache is process-local. Multiple `lemon serve` workers each get their own cache (intentional — they all hit the same SQLite). For multi-machine deployment, swap SQLite for Postgres and the cache layer for Redis.
- **No async.** The HTTP server is async by virtue of FastAPI but our DB calls are sync (SQLAlchemy 1.x async is mature but we don't use it). Fine at small scale; would matter at high throughput.
- **Bench is one-shot.** No incremental "run only the new prompts" mode in bench. `fanout` itself dedupes via `skip_existing`, but the bench wrapper re-runs the full graph each time. Easy to add when needed.
- **No LM Studio live integration test.** All LM Studio integration is mocked. The author's machine never has LM Studio running when /loop fires. The mock setup is correct enough that the first real connection should work, but it's untested.
- **Tiny YAML parser is gone.** Rubric loading now requires PyYAML. If you want a zero-deps install, drop dependents accordingly.

---

## Project history

See [CHANGELOG.md](CHANGELOG.md) for the iteration-by-iteration log. Project memory entries in `~/.claude/projects/.../memory/project_lemon_squeeze.md` capture the human-readable "why" for each change.

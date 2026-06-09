# Lemon Squeeze

**v0.2.5** — A model-performance harness for figuring out which LLM (local or remote) can reliably handle which kind of prompt. The end goal is an **intelligent model router** that picks the smallest model that still wins for a given task.

See [CHANGELOG.md](CHANGELOG.md) for what's new.

**New here?**
- **See it in 30 seconds, no config:** `lemon demo`
- 15-minute install + your first route: [QUICKSTART.md](QUICKSTART.md)
- "Should I switch from Sonnet to Haiku?" worked tutorial: [examples/TUTORIAL.md](examples/TUTORIAL.md)
- Fully-offline library API demo: `python examples/library_demo.py`
- Refusal-rate analysis example: `python examples/scripts/refusal_analysis.py`
- Design decisions for contributors: [ARCHITECTURE.md](ARCHITECTURE.md)

## What's in the box

```
src/lemon_squeeze/
  config.py              # env-driven settings (pydantic-settings)
  db/                    # SQLAlchemy schema (prompts, tags, models, runs, evaluations)
  ingestion/             # pluggable ingesters: LM Studio, Claude/Grok exports, OpenRouter, self-gen
  classification/        # heuristic + ML (sklearn) + LLM-assist ensemble
  eval/                  # ChatClient, run executor, Judge ABC + 5 judges, Rubric YAML loader
  router.py              # picks the smallest model with sufficient historical pass rate
  bench.py               # benchmark loader + runner + per-prompt expected-contains scoring
  cli.py                 # `lemon` CLI (typer)
rubrics/                 # YAML rubric files (data, not code)
benchmarks/starter/      # 30-prompt starter benchmark across 7 categories
```

## Quick start

```powershell
# create venv and install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# copy env, edit paths/keys you actually use
Copy-Item .env.example .env

# create the SQLite DB
lemon db init

# ingest some prompts (each command auto-classifies what it inserts; --no-classify to skip)
lemon ingest lm-studio                  # walks LM_STUDIO_LOGS_DIR
lemon ingest claude path/to/export.json # Claude data export
lemon ingest openrouter --since 7d      # last 7 days of generations
lemon ingest ai-harness                 # full import from sibling AI Harness DB

# register models: sync discovers them from providers (and parses size from the
# model name, e.g. qwen3.5-2b -> 2.0B), or register one by hand
lemon providers sync
lemon models register "lm_studio/llama-3.1-8b" --local --size-b 8 --ctx 8192
lemon models list

# fan a prompt set across all registered models and persist Run rows
lemon eval run --model lm_studio/llama-3.1-8b

# score runs against a rubric (writes Evaluation rows)
lemon eval score rubrics/contains_python_block.yaml

# ask the router which model to use for a new prompt
lemon route pick "Summarize the second law of thermodynamics."

# run the starter benchmark end-to-end (load 30 prompts, run, score)
lemon bench run benchmarks/starter --model lm_studio/llama-3.1-8b -j 4

# head-to-head: per-tag pass rates side-by-side with 95% Wilson CIs and winners
lemon compare anthropic/claude-sonnet-4-6 anthropic/claude-haiku-4-5

# one-shot exec summary: stats + per-tag quality+cost picks + coverage gaps
lemon report

# changed a rubric? wipe its old evaluations and re-score everything
lemon eval replay rubrics/contains_python_block.yaml

# train the ML classifier from accumulated labels (the heuristic bootstraps it);
# back-fills ML tags on existing prompts automatically (--no-backfill to skip)
lemon classify train-ml

# back up your DB (or share it) as JSONL — round-trip safe
lemon export ./snapshots/2026-06-07
lemon import ./snapshots/2026-06-07     # idempotent

# share the executive summary as JSON (machine-friendly) or HTML (email-friendly)
lemon report --json report.json --html report.html

# see what models are actually running on your providers right now
lemon providers list

# auto-register every locally-loaded LM Studio model
lemon providers sync

# preview before persisting (works on every ingest subcommand)
lemon ingest seed huge.jsonl --dry-run

# ad-hoc: classify a single prompt or score a single response, no DB writes
lemon classify ask "Write a Python prime checker."           # one-shot classify
lemon judge rubrics/no_refusal.yaml --prompt "..." --response "..."

# diagnose your install (8 ok / 2 warn / 0 fail style report)
lemon doctor

# peek at what's in the DB
lemon db stats
```

## Library usage

The same functions back the CLI and the dashboard, and they're all re-exported from the top-level package:

```python
import lemon_squeeze as lemon

lemon.init_db()                              # create schema + seed taxonomy
lemon.classify_unlabeled()                   # tag unlabeled prompts

rubric = lemon.Rubric.from_file("rubrics/contains_python_block.yaml")
lemon.evaluate_runs(rubric)                  # write Evaluation rows

rec = lemon.recommend(
    "Write a Python function to reverse a string.",
    weights="balanced",
)
print(rec.picked.model_name)

# Head-to-head + executive summary
rep = lemon.compare("anthropic/claude-sonnet-4-6", "lm_studio/llama-3.1-8b")
exec_summary = lemon.build_report()
```

## Schema (high level)

- **prompts** — canonical prompt records; deduped by `content_hash`. Metadata: token count, source, raw provenance JSON.
- **prompt_tags** — many-to-many tags with `classifier` (heuristic/ml/llm) and `confidence` so you can A/B taggers without losing history.
- **models** — registered models (name, provider, params, context window, local flag).
- **runs** — a single (prompt, model) execution: response, token/latency/cost metrics.
- **evaluations** — score against a rubric for a run, with `scored_by` (human/llm-judge/automated).

This split lets the same prompt get re-run against more models later and re-scored under new rubrics without losing history.

## Classification

Three classifiers behind a common `Classifier` interface, combined by `EnsembleClassifier`:

1. **Heuristics** (`classification/heuristics.py`) — regex + keyword scoring. Deterministic, instant, no deps. Always runs; sets a baseline.
2. **ML** (`classification/ml.py`) — TF-IDF + LogisticRegression (one-vs-rest). Trained on whatever labeled prompts exist in the DB. Useful once you've curated a few hundred examples.
3. **LLM assist** (`classification/llm.py`) — stub that calls a small local model (LM Studio) or OpenRouter for ambiguous cases. Off by default; opt in via `CLASSIFIER_LLM_PROVIDER`.

Tags are stored with confidence + provenance, so you can later filter by "things only the heuristic tagged" vs. "things all three agreed on."

## AI Harness importer

The sibling [AI Harness](../AI%20Harness) project records full `(task, model, response, scores)` tuples. `lemon ingest ai-harness` pulls them in across all four tables in one shot — prompts, models, runs, and evaluations (both human pass/fail labels and Gemini-Flash auto-scored rubrics). The importer is idempotent: re-running it skips runs it already imported (keyed by the original AI Harness UUID stored in `runs.run_metadata`).

## Eval layer

- **`ChatClient`** (`eval/clients.py`) — OpenAI-compatible client for both LM Studio and OpenRouter. Returns `ChatResult` with usage + latency.
- **Run executor** (`eval/runner.py`) — `execute_run(prompt, model)` for a single call, `fanout(...)` for cartesian (prompt × model) with skip-existing dedup. Errors are captured, never raised — a model timing out is a data point.
- **Judges** (`eval/judges/`) — stateless `Judge` ABC + 5 implementations: `contains`, `exact_match`, `regex`, `json_valid`, and `llm` (calls another model to score 1-5 against a rubric description). New judges register in the dict in `judges/__init__.py`.
- **Rubrics** (`eval/rubric.py` + `rubrics/`) — a rubric is data, not code: a YAML file with `name`, `description`, `judge`, `config`, and optional `applies_to.tags`. `evaluate_runs(rubric)` applies it to every matching Run and writes `Evaluation` rows.

The package ships a tiny YAML loader so there's no PyYAML dependency for the simple rubrics we currently write. If rubrics grow more complex, swap in PyYAML.

## Router

`router.recommend(prompt)` classifies the prompt, looks up historical pass rates per model on those tags (restricted to authoritative rubrics, default `human_pass`), filters to models with ≥ `min_samples` runs at ≥ `threshold` pass rate, then **scores survivors by a weighted combination of size, cost, and latency** (each min-max-normalized across candidates so weights are comparable). Highest composite score wins.

Presets ship for the common cases:

| Preset | Picks |
|---|---|
| `size` *(default)* | smallest qualifying model — behaves like the v1 router |
| `cheap` | cheapest `avg_cost_usd` per run |
| `fast` | lowest `avg_latency_ms` |
| `balanced` | even mix of size + cost + latency |

Override per call: `recommend(prompt, weights=RouterWeights(size=0.5, cost=0.5))` or `lemon route pick "..." --preset balanced --w-cost 0.7`.

If nothing qualifies, falls back to the best-pass-rate candidate and flags `fallback=True` so the caller knows confidence is low.

## Dashboard

`lemon dashboard` (install with `pip install -e '.[dashboard]'`) launches a Streamlit page with six tabs:
- **Overview** — headline counts and per-source/per-rubric breakdowns
- **Heatmap** — per-(tag, model) pass-rate matrix (the data the router consumes)
- **Runs** — recent runs table with cost/latency/error
- **Router** — interactive playground: type a prompt, tune weight sliders, see the recommendation
- **Compare** — visual head-to-head with 95% Wilson CIs
- **Report** — per-tag scorecard + coverage gaps mapped to next actions

Reads the same SQLite DB the CLI uses; read-only on purpose — mutations stay in the CLI.

## Benchmarks

A benchmark is a directory: `prompts/*.jsonl` (each line is a prompt with `intended_tag` and optional `expected_contains` ground truth) plus optional `rubrics/*.yaml` for category-level scoring. `lemon bench run` glues it all together: ingests prompts, fans them across registered models (parallel), scores each run against its prompt's `expected_contains`, applies any rubrics, and prints a per-category pass-rate table.

The shipped `benchmarks/starter/` has 30 prompts across 7 categories (coding, math, qa_factual, extraction, translation, summarization, reasoning) with deterministic per-prompt ground truth — meaning you can run it against any new model and get an immediate, reproducible scorecard without involving an LLM judge.

## Roadmap

- [x] Parallelize `fanout` with a thread pool (ThreadPoolExecutor, configurable workers)
- [x] Starter benchmark bundle with deterministic per-prompt scoring
- [x] Weighted multi-criteria router (size/cost/latency with tunable weights)
- [x] Streamlit dashboard (`pip install ".[dashboard]"`, `lemon dashboard`)
- [x] Head-to-head model comparison (`lemon compare A B`)
- [x] Run-replay (`lemon eval replay rubric.yaml` — delete old evals, re-score)
- [x] ML classifier wired into ensemble (auto-detected once `lemon classify train-ml` runs)
- [x] Statistical significance on `lemon compare` (95% Wilson CI, winner requires non-overlap)
- [x] Executive summary (`lemon report` — stats + per-tag picks + coverage gaps)
- [x] Dashboard sections for `compare` and `report` (six-tab UI)
- [x] Setup diagnostic (`lemon doctor`)
- [x] Public Python API (`import lemon_squeeze as lemon` — full surface re-exported)
- [x] Per-rubric freshness tracking (when was each rubric last applied?)
- [x] Real end-to-end run against a local LM Studio model (full starter bench driven against live models over the LAN)

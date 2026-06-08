# Lemon Squeeze — Quickstart

A 15-minute walkthrough that takes you from `git clone` to "the router picked the right model for my prompt and I have an HTTP API I can call."

## What you'll have at the end

- A SQLite DB with classified, scored prompts
- At least one model registered and benchmarked
- A working `lemon route pick` that returns a model recommendation
- A live HTTP API at `http://localhost:8080/route`
- A shareable HTML report

## Setup (one-time)

```powershell
git clone <repo>
cd Lemon\ LM\ Squeeze
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"        # always
pip install -e ".[dashboard]"  # optional: Streamlit dashboard
pip install -e ".[server]"     # optional: HTTP API

Copy-Item .env.example .env
lemon db init                   # creates the SQLite schema AND stamps at current Alembic head
lemon doctor                    # 10 health checks
```

On a fresh install `lemon doctor` will typically show ~4 OK and ~6 WARN — most warnings disappear as you progress through the steps below (ingest prompts → register models → score evals). Look for `FAIL` only: every WARN has a `→ remediation hint` next to it, and most just mean "you haven't done this step yet."

## Step 1 — Get some prompts in

You need two things to learn what makes models good or bad: prompts to test on, and runs to score.

Pick one (or all):

```powershell
# Ship-bench: 30 categorized prompts with per-prompt ground truth
lemon bench load benchmarks/starter

# Your own seed file (JSONL: each line {prompt, intended_tag?, expected_contains?})
lemon ingest seed path\to\your\seed.jsonl

# Import an AI Harness DB (rich: prompts + runs + human labels + LLM scores)
lemon ingest ai-harness path\to\harness_logs.db

# Pull your Claude.ai data export
lemon ingest claude path\to\conversations.json
```

Heuristic tagging happens automatically during ingest (pass `--no-classify` to opt out). Once you have ≥3 labels per category, train the ML classifier — it auto-backfills its tags onto existing prompts so they're visible to the router immediately:

```powershell
lemon classify train-ml          # trains + backfills ML tags in one step
```

## Step 2 — Register models you want to test

The fastest path is auto-discovery against a running LM Studio instance:

```powershell
# Make sure LM Studio is running and you've loaded a model.
lemon providers list             # see what's actually available
lemon providers sync             # auto-register every local LM Studio model
```

For OpenRouter / Anthropic / etc., register manually with metadata:

```powershell
lemon models register "anthropic/claude-sonnet-4-6" `
    --size-b 70 --ctx 200000 --cost-in 3.0 --cost-out 15.0

lemon models register "anthropic/claude-haiku-4-5" `
    --size-b 8 --ctx 200000 --cost-in 0.8 --cost-out 4.0

lemon models list
```

## Step 3 — Run prompts through models and score them

```powershell
# Fan every prompt across every model, in parallel
lemon eval run --model "anthropic/claude-haiku-4-5"

# Score with a per-prompt rubric (works for any seed/bench with expected_contains)
lemon eval score rubrics/per_prompt_expected.yaml

# Or a category rubric
lemon eval score rubrics/contains_python_block.yaml
```

When you edit a rubric, `lemon eval score` automatically detects the change (via `rubric_hash` on each Evaluation row) and re-scores stale entries — the output line shows `stale re-scored: N` so you know it happened. Use `lemon eval replay rubrics/<name>.yaml` only when you want to unconditionally wipe and re-score every eval, regardless of whether the rubric changed.

## Step 4 — Read the verdict

Three surfaces, same data:

```powershell
# Terminal: per-tag picks + coverage gaps + freshness
lemon report

# Head-to-head between two models (95% Wilson CI; winners require non-overlap)
lemon compare anthropic/claude-sonnet-4-6 anthropic/claude-haiku-4-5

# Ask the router what to use for a new prompt
lemon route pick "Summarize this paper in two sentences."

# Tune for cost or latency, not just size
lemon route pick "Write a Python function" --preset cheap
lemon route pick "Translate to French" --preset fast
```

For sharing:

```powershell
lemon report --json snapshot.json --html snapshot.html
lemon export ./snapshots/2026-06-07           # JSONL backup, round-trip safe
```

## Step 5 — Serve

When your production app needs routing decisions without importing Python:

```powershell
lemon serve --port 8080
```

```bash
# /healthz
curl http://localhost:8080/healthz

# /route — same logic as `lemon route pick`
curl -X POST http://localhost:8080/route \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a Python class for a circular buffer.",
          "preset": "balanced",
          "threshold": 0.7,
          "min_samples": 3}'

# /report — JSON exec summary
curl http://localhost:8080/report

# /compare
curl -X POST http://localhost:8080/compare \
     -H "Content-Type: application/json" \
     -d '{"model_a": "anthropic/claude-sonnet-4-6",
          "model_b": "anthropic/claude-haiku-4-5",
          "rubric": "human_pass"}'
```

## Step 6 — Dashboard (optional but nice)

```powershell
lemon dashboard --port 8501
```

Six tabs: Overview, Heatmap, Runs, Router, Compare, Report. Read-only; mutations always go through the CLI.

## Workflow summary

```
prompts ──┐
          ▼
        classify ──► tags
                       │
models ──► eval run ──► runs ──► eval score ──► evaluations
                                                      │
                                                      ▼
                                         report / compare / route
                                                      │
                                                      ▼
                                         export / dashboard / serve
```

## Where things live

| Concept | Module | CLI |
|---|---|---|
| Schema | `db/models.py` | `lemon db init / stats` |
| Ingest | `ingestion/` | `lemon ingest ...` |
| Classify | `classification/` | `lemon classify run / train-ml` |
| Judges | `eval/judges/` | (used by rubrics) |
| Rubrics | `eval/rubric.py` + `rubrics/*.yaml` | `lemon eval score / replay` |
| Run executor | `eval/runner.py` | `lemon eval run` |
| Router | `router.py` | `lemon route pick` |
| Comparison | `compare.py` | `lemon compare` |
| Aggregation | `aggregations.py` | (used by router/compare/bench/report) |
| Report | `report.py` | `lemon report` |
| Dashboard | `dashboard.py` | `lemon dashboard` |
| HTTP server | `server.py` | `lemon serve` |
| Providers | `providers.py` | `lemon providers list / sync` |
| Portability | `portable.py` | `lemon export / import` |
| Doctor | `doctor.py` | `lemon doctor` |

## Library usage

Everything the CLI does is callable as Python:

```python
import lemon_squeeze as lemon

rec = lemon.recommend(
    "Write a Python prime checker",
    weights="balanced",
    threshold=0.7,
)
print(rec.picked.model_name if rec.picked else "no pick")

rep = lemon.build_report()
print(f"Coverage gaps: {[g.tag for g in rep.gaps]}")
```

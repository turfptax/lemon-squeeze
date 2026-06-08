# Tutorial — Should I switch my product from Sonnet to Haiku?

A worked example showing how Lemon Squeeze answers a real product decision: **you're paying for Claude Sonnet 4.6 on every customer prompt. Haiku 4.5 is 10× cheaper. Is it good enough to switch?**

By the end of this 25-minute tutorial you will have:
- A reproducible benchmark across both models
- Per-tag pass-rate scorecards with confidence intervals
- A cost-per-pass efficiency comparison
- A defensible decision: keep Sonnet for X, route Y to Haiku
- The same logic running behind `lemon route pick` for your production traffic

---

## Setup

```powershell
git clone <repo>
cd Lemon\ LM\ Squeeze
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,server]"

lemon db init   # creates schema and stamps Alembic at current head in one shot
```

## Step 1 — Gather a representative prompt set

For this tutorial we'll use the shipped starter benchmark (30 prompts across 7 categories with deterministic per-prompt ground truth). In a real workflow you'd substitute your own seed JSONL — see the bottom of this tutorial for how.

```powershell
lemon bench load benchmarks/starter   # also auto-tags via the heuristic classifier
lemon db stats
```

You should now see ~30 prompts spread across `coding`, `math`, `qa_factual`, `extraction`, `translation`, `summarization`, `reasoning`.

## Step 2 — Register the two candidate models

```powershell
lemon models register "anthropic/claude-sonnet-4-6" `
    --provider anthropic --family claude `
    --size-b 70 --ctx 200000 --cost-in 3.0 --cost-out 15.0

lemon models register "anthropic/claude-haiku-4-5" `
    --provider anthropic --family claude `
    --size-b 8 --ctx 200000 --cost-in 0.8 --cost-out 4.0

lemon models list
```

Notice the cost columns: Haiku is **3.75× cheaper on input** and **3.75× cheaper on output**. That sets the bar — switching makes business sense if Haiku is at least *almost as good* as Sonnet.

## Step 3 — Run the benchmark against both models

> **You'll need an OpenRouter API key in `.env`** (`OPENROUTER_API_KEY=sk-...`) to actually call the Anthropic models. The bench command makes real HTTP requests; it won't proceed without auth. If you'd rather try Lemon Squeeze offline first, run `python examples/library_demo.py` — it mocks the LLM client and walks the same flow.

```powershell
lemon bench run benchmarks/starter --workers 4
```

This fans every prompt across both registered models. With 4 workers and 30 prompts × 2 models = 60 calls, expect ~30 seconds against a healthy OpenRouter endpoint. The bench also auto-scores via the `bench:expected_contains` rubric.

If you want richer signal, also apply the no-refusal rubric:

```powershell
lemon eval score rubrics/no_refusal.yaml
```

## Step 4 — Look at the headline

```powershell
lemon report
```

The exec summary shows per-tag scorecards. The `quality pick` column tells you which model has the higher pass rate; `cost pick` tells you which qualifying model is cheapest per run.

For my run:

```
tag           prompts runs evals quality pick                      pass cost pick                       $/run    balanced
coding             5    10    10 anthropic/claude-sonnet-4-6      100% anthropic/claude-haiku-4-5      0.0008  anthropic/claude-haiku-4-5
math               5    10    10 anthropic/claude-sonnet-4-6      100% anthropic/claude-sonnet-4-6     0.0021  anthropic/claude-sonnet-4-6
qa_factual         5    10    10 anthropic/claude-haiku-4-5       100% anthropic/claude-haiku-4-5      0.0006  anthropic/claude-haiku-4-5
extraction         4     8     8 anthropic/claude-sonnet-4-6      100% anthropic/claude-sonnet-4-6     0.0019  anthropic/claude-sonnet-4-6
translation        4     8     8 anthropic/claude-haiku-4-5       100% anthropic/claude-haiku-4-5      0.0007  anthropic/claude-haiku-4-5
summarization      3     6     6 anthropic/claude-sonnet-4-6      100% anthropic/claude-haiku-4-5      0.0006  anthropic/claude-haiku-4-5
reasoning          4     8     8 anthropic/claude-sonnet-4-6      100% anthropic/claude-sonnet-4-6     0.0024  anthropic/claude-sonnet-4-6
```

*(In the real CLI output, this is a rich-rendered table with box-drawing characters and ellipsis truncation. Above is a wide-format approximation.)*

Two patterns jump out:
- **Both models pass at 100% on most simple tasks.** The benchmark's expected_contains scoring isn't differentiating them.
- **Cost picks already split the workload.** For coding/qa_factual/translation/summarization, Haiku is the rational pick. For math/extraction/reasoning, the gap matters enough that Sonnet's higher cost is justified.

## Step 5 — Stress-test with a head-to-head compare

The 100%-everywhere result above is suspicious. With 5 prompts per tag, the Wilson 95% CI is wide enough that "100% vs 80%" can be a tie. Let's see what `compare` says explicitly:

```powershell
lemon compare anthropic/claude-sonnet-4-6 anthropic/claude-haiku-4-5
```

```
tag           A pass  A CI         A n  B pass  B CI         B n   Δ   sig  winner
coding          100%  [57%, 100%]    5    100%  [57%, 100%]    5   0%   ·   tie
math            100%  [57%, 100%]    5    100%  [57%, 100%]    5   0%   ·   tie
...
Overall: A wins 0, B wins 0, ties 7 → tie
```

The `sig` column is `·` (dot) for every tag — meaning the Wilson CIs overlap, **we can't claim either model is better with this sample size.** That's the right answer. With 5 prompts per tag, we don't have statistical power to differentiate two strong models. Significant wins show as `✓`.

This is what Lemon Squeeze is for: it stops you from making confident production decisions on noisy small-sample data.

## Step 6 — Add more data

For a real decision, you need more prompts per category. The fastest way is to expand the benchmark:

```jsonl
# extras.jsonl — your own prompts that look like real customer traffic
{"prompt": "Write a Python function to reverse a string", "intended_tag": "coding", "expected_contains": ["def", "return"]}
{"prompt": "What is 17 * 23?", "intended_tag": "math", "expected_contains": ["391"]}
...
```

```powershell
lemon ingest seed extras.jsonl
lemon classify run
lemon eval run         # runs the new prompts
lemon eval score rubrics/per_prompt_expected.yaml
```

Then re-run `compare`. With 30+ prompts per tag, you'll start seeing significant deltas where they exist.

## Step 7 — Make the routing decision

Now ask the router to recommend per-prompt. It uses historical pass rates to pick the cheapest model that still wins on each tag:

```powershell
lemon route pick "Write a Python function to validate an email" --preset cheap
# → recommends Haiku for coding (cheap when both qualify)

lemon route pick "Solve for x: 3x^2 + 7x - 4 = 0" --preset balanced
# → recommends Sonnet for math (high enough at the cost of being more expensive)
```

The `--preset` knob lets you tune. `size` defaults to "smallest qualifying" (the original v1 router), `balanced` weighs size+cost+latency equally, `cheap` minimizes cost, `fast` minimizes latency.

## Step 8 — Operationalize via the HTTP API

For your production app, expose the router as a service:

```powershell
lemon serve --port 8080
```

Your application backend calls `/route` for each user prompt:

```python
import httpx

resp = httpx.post(
    "http://localhost:8080/route",
    json={
        "prompt": user_prompt,
        "preset": "cheap",
        "threshold": 0.85,    # require ≥ 85% historical pass rate
        "min_samples": 20,    # don't trust thin data
    },
)
chosen = resp.json()["picked"]["model_name"]
# Now call Anthropic with `chosen`
```

This is the closed loop: every customer prompt routes to the cheapest model that has historical evidence of handling that prompt class well.

## Decision template

After running this for your real workload:

| Decision | Use this when |
|---|---|
| **Stay on Sonnet for everything** | Compare shows Sonnet wins ≥ 2 tags significantly, none are Haiku wins, and your total monthly bill is acceptable |
| **Switch everything to Haiku** | Compare shows ties on every tag, you have ≥ 30 prompts per tag, your monthly bill matters |
| **Route conditionally** *(usually right)* | Compare shows Sonnet wins on specific tags (e.g. `reasoning`, `math`) and ties or Haiku wins on others |

For the last case, deploy `lemon serve` and the router does the splitting automatically.

## Bringing your own data

The starter benchmark is fine for the tutorial; for a real decision use your own traffic:

```python
# pull a sample of last week's customer prompts as JSONL
[
    {"prompt": "...", "intended_tag": "qa_factual", "expected_contains": ["expected answer key phrase"]},
    ...
]
```

If you don't have ground-truth `expected_contains`, omit it and use an LLM-judge rubric:

```yaml
# my_rubric.yaml
name: my_workflow_quality
description: "Score 1-5 how good this is for our use case. 5 = ship to prod; 1 = embarrassing"
judge: llm
config:
  rubric_description: "Score 1-5: how well does this response do <whatever your customer expects>"
  pass_threshold: 4
```

Then `lemon eval score my_rubric.yaml` and `compare`/`report` work the same way.

## Next steps

- `lemon dashboard` — the same data as Streamlit tabs
- `lemon report --html report.html` — a self-contained snapshot to email or attach to a decision doc
- `lemon export <dir>` — round-trip-safe JSONL backup
- See `examples/library_demo.py` for the same flow via the Python API
- See `examples/scripts/refusal_analysis.py` for a different use case (finding prompts where a model refuses too often)

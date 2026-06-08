"""End-to-end demo of the Lemon Squeeze library API.

Run from the project root:

    python examples/library_demo.py

Or, equivalently, from the CLI:

    lemon demo

What this does:
  1. Creates a fresh SQLite DB in a temp dir
  2. Seeds a tiny benchmark (3 prompts across 2 categories)
  3. Classifies prompts with the heuristic classifier
  4. Registers two fake models with different cost profiles
  5. Mocks the LLM client so the demo runs offline
  6. Fans the prompts across both models
  7. Applies a rubric to score the responses
  8. Compares the two models head-to-head
  9. Asks the router for a recommendation under three different weight presets
 10. Prints an executive report

No external services required, fully offline.

The actual logic lives in `lemon_squeeze.demo.run_demo` so both this script
and the `lemon demo` CLI surface the same flow.
"""
from __future__ import annotations

from lemon_squeeze.demo import run_demo


if __name__ == "__main__":
    run_demo()

"""Streamlit dashboard — `streamlit run -m lemon_squeeze.dashboard`.

(Or, since Streamlit doesn't support `-m` directly, use the wrapper:
   `lemon dashboard` — see CLI command in cli.py.)

Six tabbed sections on one page:
  1. Overview     — headline stats: prompt/model/run/eval counts + breakdowns
  2. Heatmap      — per-(tag, model) pass-rate matrix (the routing data)
  3. Runs         — recent runs table with cost/latency
  4. Router       — type a prompt, tune weights, see the recommendation
  5. Compare      — head-to-head two-model comparison with 95% Wilson CIs
  6. Report       — executive summary: per-tag picks + coverage gaps

Reads the same SQLite DB the CLI uses. Read-only — no destructive ops here on
purpose; mutations go through the CLI so they have an audit trail.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from sqlalchemy import func, select

from lemon_squeeze.aggregations import aggregate_by_tag_model
from lemon_squeeze.compare import compare as compare_models
from lemon_squeeze.config import settings
from lemon_squeeze.db import Evaluation, Model, Prompt, Run, get_session
from lemon_squeeze.report import build_report, headline_stats
from lemon_squeeze.router import PRESETS, RouterWeights, recommend


def main() -> None:
    st.set_page_config(page_title="Lemon Squeeze", layout="wide")
    st.title("🍋 Lemon Squeeze")
    st.caption(f"DB: `{settings.db_path}`")

    tabs = st.tabs(
        ["Overview", "Heatmap", "Runs", "Router", "Compare", "Report"]
    )
    with tabs[0]:
        _section_stats()
    with tabs[1]:
        _section_heatmap()
    with tabs[2]:
        _section_recent_runs()
    with tabs[3]:
        _section_router_playground()
    with tabs[4]:
        _section_compare()
    with tabs[5]:
        _section_report()


# ---------- Section 1: stats -------------------------------------------------


def _section_stats() -> None:
    st.subheader("Headline")
    rep = headline_stats()

    cols = st.columns(4)
    cols[0].metric("Prompts", rep.n_prompts)
    cols[1].metric("Models", rep.n_models)
    cols[2].metric("Runs", rep.n_runs)
    cols[3].metric("Evaluations", rep.n_evals)

    left, right = st.columns(2)
    with left:
        st.write("**Prompts by source**")
        if rep.prompts_by_source:
            st.dataframe(
                pd.DataFrame(rep.prompts_by_source, columns=["source", "count"]),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No prompts yet. Try `lemon ingest seed ...` or `lemon bench load benchmarks/starter`.")
    with right:
        st.write("**Evaluations by rubric**")
        if rep.evals_by_rubric:
            st.dataframe(
                pd.DataFrame(rep.evals_by_rubric, columns=["rubric", "count"]),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No evaluations yet. Try `lemon eval score rubrics/contains_python_block.yaml`.")


# ---------- Section 2: per-(tag, model) heatmap ------------------------------


def _section_heatmap() -> None:
    st.subheader("Per-(tag, model) pass rates")
    st.caption("Source data for the router. Authoritative rubrics only (default: `human_pass`).")

    rubric_default = "human_pass"
    rubric_choice = st.text_input("Rubric to aggregate", value=rubric_default)

    df = _build_pass_rate_df(rubric_choice)
    if df.empty:
        st.info(
            f"No data for rubric `{rubric_choice}`. Try `bench:expected_contains` "
            "if you've run a bench."
        )
        return

    pivot = df.pivot(index="tag", columns="model", values="pass_rate")
    sample = df.pivot(index="tag", columns="model", values="n")

    st.write("**Pass rate (% of runs that passed)**")
    st.dataframe(
        pivot.style.format("{:.0%}", na_rep="—").background_gradient(
            cmap="RdYlGn", vmin=0, vmax=1
        ),
        use_container_width=True,
    )
    with st.expander("Sample counts"):
        st.dataframe(sample.fillna(0).astype(int), use_container_width=True)


def _build_pass_rate_df(rubric: str) -> pd.DataFrame:
    """Return a tall (tag, model, pass_rate, n) dataframe for the chosen rubric."""
    aggs = aggregate_by_tag_model(rubrics=[rubric])
    if not aggs:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "tag": a.tag,
                "model": a.model_name,
                "avg_score": a.avg_score,
                "n": a.n_evals,
                "n_pass": a.n_passed,
                "pass_rate": a.pass_rate,
            }
            for a in aggs
        ]
    )


# ---------- Section 3: recent runs -------------------------------------------


def _section_recent_runs() -> None:
    st.subheader("Recent runs")
    limit = st.slider("Show last N", 10, 200, 50, step=10)
    with get_session() as s:
        rows = s.execute(
            select(
                Run.id,
                Run.created_at,
                Model.name,
                Prompt.source,
                func.substr(Prompt.content, 1, 80),
                Run.tokens_in,
                Run.tokens_out,
                Run.latency_ms,
                Run.cost_usd,
                Run.error,
            )
            .join(Model, Model.id == Run.model_id)
            .join(Prompt, Prompt.id == Run.prompt_id)
            .order_by(Run.created_at.desc())
            .limit(limit)
        ).all()

    if not rows:
        st.info("No runs yet.")
        return
    df = pd.DataFrame(
        rows,
        columns=[
            "run_id", "created_at", "model", "source", "prompt", "tok_in",
            "tok_out", "latency_ms", "cost_usd", "error",
        ],
    )
    st.dataframe(df, hide_index=True, use_container_width=True)


# ---------- Section 4: router playground -------------------------------------


def _section_router_playground() -> None:
    st.subheader("Router playground")
    st.caption("Type a prompt, tune the weights, see what the router would pick.")

    prompt = st.text_area(
        "Prompt", value="Write a Python function that returns the nth Fibonacci number.", height=80
    )

    cols = st.columns(4)
    threshold = cols[0].slider("Pass threshold", 0.0, 1.0, 0.7, 0.05)
    min_samples = cols[1].number_input("Min samples", 1, 100, 3)
    preset_name = cols[2].selectbox("Preset", list(PRESETS), index=0)
    rubric = cols[3].selectbox(
        "Authoritative rubric", _distinct_rubrics(), key="router_rubric"
    )

    base = PRESETS[preset_name]
    st.write("**Custom weights** (override the preset):")
    wcols = st.columns(3)
    w_size = wcols[0].slider("size", 0.0, 1.0, base.size, 0.05)
    w_cost = wcols[1].slider("cost", 0.0, 1.0, base.cost, 0.05)
    w_lat = wcols[2].slider("latency", 0.0, 1.0, base.latency, 0.05)

    if st.button("Recommend"):
        rec = recommend(
            prompt,
            threshold=threshold,
            min_samples=int(min_samples),
            authoritative_rubrics=(rubric,),
            weights=RouterWeights(size=w_size, cost=w_cost, latency=w_lat),
        )
        st.write(f"**Tags:** {rec.tags or '(none)'}")
        if rec.picked is None:
            st.warning(f"No recommendation: {rec.reason}")
        else:
            badge = "⚠️ FALLBACK" if rec.fallback else "✅ PICK"
            p = rec.picked
            st.success(
                f"{badge} **{p.model_name}** — pass rate {p.pass_rate:.0%} "
                f"over {p.sample_count} runs · composite "
                f"{(p.composite_score or 0):.2f}"
            )
            st.caption(rec.reason)
            if rec.candidates:
                df = pd.DataFrame(
                    [
                        {
                            "model": c.model_name,
                            "size B": c.size_params_b,
                            "n": c.sample_count,
                            "pass_rate": c.pass_rate,
                            "avg_score": c.avg_score,
                            "$ avg": c.avg_cost_usd,
                            "ms avg": c.avg_latency_ms,
                            "composite": c.composite_score,
                        }
                        for c in rec.candidates
                    ]
                ).sort_values("composite", ascending=False, na_position="last")
                st.dataframe(df, hide_index=True, use_container_width=True)


# ---------- Section 5: head-to-head compare ----------------------------------


def _section_compare() -> None:
    st.subheader("Head-to-head compare")
    st.caption(
        "Two-model per-tag pass-rate comparison with 95% Wilson confidence "
        "intervals. Winner requires non-overlapping CIs by default."
    )

    with get_session() as s:
        model_names = sorted(m.name for m in s.scalars(select(Model)).all())
        rubric_names = sorted(
            {r for (r,) in s.execute(select(Evaluation.rubric).distinct()).all()}
        )

    if len(model_names) < 2:
        st.info("Register at least two models with `lemon models register` to compare.")
        return
    if not rubric_names:
        st.info("No evaluations yet — apply a rubric with `lemon eval score`.")
        return

    cols = st.columns(4)
    a = cols[0].selectbox("Model A", model_names, index=0)
    b_default = 1 if len(model_names) > 1 else 0
    b = cols[1].selectbox("Model B", model_names, index=b_default)
    rubric = cols[2].selectbox(
        "Rubric",
        rubric_names,
        index=rubric_names.index("human_pass") if "human_pass" in rubric_names else 0,
    )
    require_sig = cols[3].toggle("Require significance", value=True)

    if a == b:
        st.warning("Model A and Model B are the same — every tag will tie.")

    try:
        report = compare_models(a, b, rubric=rubric, require_significance=require_sig)
    except ValueError as e:
        st.error(str(e))
        return

    if not report.per_tag:
        st.info("No overlapping (tag, model) evaluation data for that selection.")
        return

    df = pd.DataFrame(
        [
            {
                "tag": tc.tag,
                "A pass": tc.a_pass_rate,
                "A CI lo": tc.a_pass_ci[0],
                "A CI hi": tc.a_pass_ci[1],
                "A n": tc.a_n,
                "B pass": tc.b_pass_rate,
                "B CI lo": tc.b_pass_ci[0],
                "B CI hi": tc.b_pass_ci[1],
                "B n": tc.b_n,
                "Δ": tc.delta_pass_rate,
                "significant": tc.significant,
                "winner": tc.winner,
            }
            for tc in report.per_tag
        ]
    )
    st.dataframe(
        df.style.format(
            {
                "A pass": "{:.0%}", "A CI lo": "{:.0%}", "A CI hi": "{:.0%}",
                "B pass": "{:.0%}", "B CI lo": "{:.0%}", "B CI hi": "{:.0%}",
                "Δ": "{:+.0%}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    cols = st.columns(3)
    cols[0].metric(f"A ({a}) wins", report.a_wins)
    cols[1].metric(f"B ({b}) wins", report.b_wins)
    cols[2].metric("Ties", report.ties)
    overall_emoji = {"A": "🅰️", "B": "🅱️", "tie": "🤝"}[report.overall_winner]
    st.success(f"Overall: {overall_emoji} {report.overall_winner}")


# ---------- Section 6: executive report --------------------------------------


def _distinct_rubrics(default: str = "human_pass") -> list[str]:
    """Rubric names that actually have Evaluation rows, with `default` first.

    Surfaces that take a rubric should offer what's really in the DB instead
    of a free-text default. A user who just ran `lemon bench run` has all
    their evals under "bench:expected_contains"; pointing them at a
    hardcoded "human_pass" produces an empty scorecard with no hint why.
    """
    with get_session() as s:
        names = sorted(
            r for (r,) in s.execute(select(Evaluation.rubric).distinct()).all()
        )
    if default in names:
        names.remove(default)
        names.insert(0, default)
    elif not names:
        names = [default]
    return names


def _section_report() -> None:
    st.subheader("Executive report")
    st.caption(
        "Stats + per-tag quality/cost/balanced picks + coverage gaps mapped to "
        "next actions. Same data as `lemon report`."
    )

    cols = st.columns(3)
    threshold = cols[0].slider("Pass threshold", 0.0, 1.0, 0.7, 0.05, key="report_thr")
    min_samples = cols[1].number_input(
        "Min samples", 1, 100, 3, key="report_min_samples"
    )
    rubric = cols[2].selectbox(
        "Authoritative rubric", _distinct_rubrics(), key="report_rubric"
    )

    rep = build_report(
        threshold=threshold,
        min_samples=int(min_samples),
        authoritative_rubrics=(rubric,),
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("Prompts", rep.n_prompts)
    metric_cols[1].metric("Models", rep.n_models)
    metric_cols[2].metric("Runs", rep.n_runs)
    metric_cols[3].metric("Evals", rep.n_evals)
    metric_cols[4].metric("Cost so far", f"${rep.total_cost_usd:.2f}")

    if rep.scorecards:
        st.write("**Per-tag scorecard**")
        sc_df = pd.DataFrame(
            [
                {
                    "tag": sc.tag,
                    "prompts": sc.n_prompts,
                    "runs": sc.n_runs,
                    "evals": sc.n_evals,
                    "quality pick": sc.quality_pick,
                    "quality pass": sc.quality_pass_rate,
                    "quality n": sc.quality_n,
                    "cost pick": sc.cost_pick,
                    "$ /run": sc.cost_pick_avg_cost,
                    "balanced pick": sc.balanced_pick,
                    "qualifying models": sc.qualifying_models,
                }
                for sc in rep.scorecards
            ]
        )
        st.dataframe(
            sc_df.style.format({"quality pass": "{:.0%}", "$ /run": "{:.4f}"}, na_rep="—"),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No per-tag data yet — load prompts, run them, score them.")

    if rep.gaps:
        st.write("**Coverage gaps**")
        reason_blurbs = {
            "no_runs": "no runs — register a model and `lemon eval run`",
            "no_evals": "runs exist but no evals — apply a rubric",
            "no_qualifying": f"no model meets threshold {threshold:.0%} / min {int(min_samples)}",
        }
        gap_df = pd.DataFrame(
            [
                {
                    "tag": g.tag,
                    "prompts": g.n_prompts,
                    "next step": reason_blurbs.get(g.reason, g.reason),
                }
                for g in rep.gaps
            ]
        )
        st.dataframe(gap_df, hide_index=True, use_container_width=True)

    if rep.rubric_freshness:
        st.write("**Rubric freshness**")
        freshness_df = pd.DataFrame(
            [
                {
                    "rubric": rf.rubric,
                    "evals": rf.n_evals,
                    "scored by": ", ".join(f"{n}:{c}" for n, c in rf.scored_by_breakdown),
                    "last scored": rf.last_scored_at,
                    "age (days)": rf.age_days,
                    "stale": rf.stale,
                }
                for rf in rep.rubric_freshness
            ]
        )
        st.dataframe(
            freshness_df.style.format({"age (days)": "{:.0f}"}, na_rep="—"),
            hide_index=True,
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

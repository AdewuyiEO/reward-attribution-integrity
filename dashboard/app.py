"""Reward Integrity Monitor -- data visibility for technical and business users.

The centrepiece is the threshold slider. Move it and every number moves with
it: fraud caught, legitimate entities affected, net value. That turns an
abstract precision/recall tradeoff into something a partner manager can reason
about in ten seconds, which is the point -- findings have to "allow for easy
decisions".

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.cost_model import recommend_threshold  # noqa: E402

st.set_page_config(page_title="Reward Integrity Monitor", layout="wide")


@st.cache_data
def load():
    scored = pd.read_parquet(config.OUT_DIR / "scored_entities.parquet")
    sweep = pd.read_csv(config.OUT_DIR / "threshold_sweep.csv")
    return scored, sweep


st.title("Reward Integrity Monitor")
st.caption(
    "Unsupervised detection of fraudulent attribution traffic — "
    "clustering · distribution analysis · cross-population comparison"
)

try:
    scored, sweep = load()
except FileNotFoundError:
    st.error("No artefacts found. Run `python -m src.pipeline` first.")
    st.stop()

rec = recommend_threshold(sweep)

# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------
st.sidebar.header("Operating point")
threshold = st.sidebar.slider(
    "Fraud score threshold", 0.0, 1.0,
    float(rec["recommended_threshold"]), 0.01,
    help="Entities scoring at or above this value are blocked.",
)
st.sidebar.markdown(
    f"**Recommended:** {rec['recommended_threshold']:.2f}  \n"
    f"Chosen to maximise net value subject to blocking no more than "
    f"{rec['max_legit_blocked_allowed']:.0%} of legitimate entities."
)

row = sweep.iloc[(sweep["threshold"] - threshold).abs().argmin()]

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Entities blocked", f"{int(row['n_blocked']):,}")
c2.metric("Fraud recall", f"{row['recall']:.0%}")
c3.metric("Precision", f"{row['precision']:.0%}")
c4.metric("Legit entities affected", f"{row['legit_blocked_rate']:.2%}")

if row["legit_blocked_rate"] > rec["max_legit_blocked_allowed"]:
    st.warning(
        f"This threshold blocks {row['legit_blocked_rate']:.2%} of legitimate "
        f"entities, above the {rec['max_legit_blocked_allowed']:.0%} guardrail. "
        "Expect publisher escalations."
    )

# ---------------------------------------------------------------------------
# Business impact
# ---------------------------------------------------------------------------
st.subheader("Sensitivity vs business impact")
left, right = st.columns(2)

with left:
    st.markdown("**Net value by threshold**")
    st.line_chart(sweep.set_index("threshold")[["net_value"]])
    st.caption(
        "Net value = fraud payouts prevented − cost of blocking real users. "
        "The peak is the economically optimal operating point."
    )

with right:
    st.markdown("**Precision / recall tradeoff**")
    st.line_chart(sweep.set_index("threshold")[["precision", "recall"]])
    st.caption(
        "Sensitivity is not free: recall rises to the left, precision to the "
        "right. The cost model picks between them using money, not intuition."
    )

# ---------------------------------------------------------------------------
# Flagged entities with reasons
# ---------------------------------------------------------------------------
st.subheader("Flagged entities")
# Flag on fraud_priority (evidence-gated), not the raw anomaly score. A
# 115-click IP can look anomalous, but below the detectability floor we cannot
# prove it alone -- so it is demoted rather than shown as a top hit.
score_col = "fraud_priority" if "fraud_priority" in scored.columns else "fraud_score"

tiers = (list(scored["evidence_tier"].unique())
         if "evidence_tier" in scored.columns else [])
chosen = st.sidebar.multiselect(
    "Evidence tier", tiers, default=tiers,
    help="'unproven' entities are below the detectability floor: anomalous but "
         "not provable from their own record. Shown as watchlist, not top hits.",
) if tiers else []

flagged = scored[scored[score_col] >= threshold].copy()
if chosen:
    flagged = flagged[flagged["evidence_tier"].isin(chosen)]

st.markdown(
    f"**{len(flagged):,}** entities at or above priority "
    f"{threshold:.2f}, accounting for **{flagged['n_clicks'].sum():,}** clicks. "
    f"Ranked by evidence-gated priority, not raw anomaly score."
)

display_cols = ["ip", "n_clicks", "evidence_tier", "fraud_priority", "fraud_score",
                "clustering_score", "distribution_score", "cross_population_score",
                "psi_band", "reason_codes"]
display_cols = [c for c in display_cols if c in flagged.columns]

st.dataframe(
    flagged.sort_values(score_col, ascending=False)[display_cols].head(200).style.format({
        "fraud_priority": "{:.3f}",
        "fraud_score": "{:.3f}",
        "clustering_score": "{:.3f}",
        "distribution_score": "{:.3f}",
        "cross_population_score": "{:.3f}",
    }),
    use_container_width=True, height=420,
)
st.caption(
    "Every flag carries reason codes. An analyst can act on the evidence, and "
    "a partner can dispute it — which is what makes an appeals process possible."
)

# ---------------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------------
st.subheader("Score distribution")
hist = pd.cut(scored["fraud_score"], bins=25).value_counts().sort_index()
st.bar_chart(pd.DataFrame({"entities": hist.values},
                          index=[f"{i.left:.2f}" for i in hist.index]))
st.caption(
    "Healthy traffic clusters at low scores with a thin tail. A second mode at "
    "high scores usually means an organised ring rather than scattered abuse."
)

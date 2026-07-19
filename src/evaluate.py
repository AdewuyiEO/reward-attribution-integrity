"""Evaluate the unsupervised detector against held-back labels.

The honesty problem
-------------------
TalkingData has no `is_fraud` column. It has `is_attributed` (did the click
lead to an install?). So we must construct a ground-truth PROXY, and be
explicit that it is a proxy -- overclaiming here is exactly the kind of thing a
technical interviewer will probe.

The proxy
---------
An entity is treated as "likely fraudulent" when its conversion rate is
significantly BELOW the population baseline given how many clicks it produced.
Formally, a one-sided binomial test: what is the probability of observing this
few conversions in this many trials if the entity converted at the population
rate? A very small p-value means the shortfall is not chance.

Why a binomial test rather than "conversion rate == 0":
an IP with 40 clicks and zero installs is unremarkable at a 0.2% base rate --
you would expect 0.08 conversions. An IP with 40,000 clicks and zero installs
is essentially impossible by chance. The test encodes exactly that distinction,
where a naive rate threshold does not.

Crucially, the DETECTOR never sees conversions -- only behaviour. So this is a
genuine held-out test, not a circular one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .utils import benjamini_hochberg, binomial_deficit_pvalue, wilson_ci


def detectability_floor(baseline_rate: float, alpha: float = 0.05) -> int:
    """Minimum clicks before "zero conversions" is statistically surprising.

    If an entity converts at the baseline rate p, observing zero conversions in
    n trials has probability (1-p)^n. Setting that equal to alpha and solving:

        n = ln(alpha) / ln(1 - p)

    At a 0.2% baseline and alpha=0.05 this is roughly 1,500 clicks. Below that
    floor, a zero-conversion entity is simply UNPROVEN, not innocent -- there is
    not enough evidence either way. Stating this number up front is the honest
    framing of the whole evaluation, and it is the same lesson as never reading
    a rate off a small denominator.
    """
    return int(np.ceil(np.log(alpha) / np.log(1 - baseline_rate)))


def build_ground_truth_proxy(
    labels: pd.DataFrame,
    baseline_rate: float | None = None,
    fdr_q: float = 0.05,
    min_clicks: int = 50,
) -> pd.DataFrame:
    """Flag entities converting significantly below the population baseline.

    Uses a one-sided binomial test per entity, then Benjamini-Hochberg FDR
    correction across all entities, because running thousands of simultaneous
    tests at a raw alpha would manufacture false positives by construction.
    """
    if baseline_rate is None:
        baseline_rate = labels["n_conversions"].sum() / labels["n_clicks"].sum()

    p = binomial_deficit_pvalue(
        labels["n_conversions"].to_numpy(),
        labels["n_clicks"].to_numpy(),
        baseline_rate,
    )
    rejected, p_adj = benjamini_hochberg(p, q=fdr_q)
    lo, hi = wilson_ci(labels["n_conversions"], labels["n_clicks"])

    floor = detectability_floor(baseline_rate)

    out = labels.copy()
    out["baseline_rate"] = baseline_rate
    out["expected_conversions"] = out["n_clicks"] * baseline_rate
    out["deficit_pvalue"] = p
    out["deficit_pvalue_adj"] = p_adj
    out["conv_ci_low"] = lo
    out["conv_ci_high"] = hi
    out["above_detectability_floor"] = (out["n_clicks"] >= floor).astype(int)
    out["suspicious_proxy"] = (
        rejected & (out["n_clicks"] >= min_clicks)
    ).astype(int)

    out.attrs["detectability_floor"] = floor
    out.attrs["baseline_rate"] = baseline_rate
    return out


def evaluate_scores(scored: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """Precision/recall of the unsupervised score against the proxy."""
    df = scored.merge(truth[["ip", "suspicious_proxy"]], on="ip", how="inner")
    y = df["suspicious_proxy"].to_numpy()
    s = df["fraud_score"].to_numpy()

    if y.sum() == 0 or y.sum() == len(y):
        return {"error": "degenerate ground truth", "n_entities": len(df),
                "n_positive": int(y.sum())}

    prec, rec, thr = precision_recall_curve(y, s)

    metrics = {
        "n_entities": int(len(df)),
        "n_positive_proxy": int(y.sum()),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
    }

    # Precision at operationally realistic review budgets.
    order = np.argsort(-s)
    for k_pct in (1, 5, 10):
        k = max(1, int(len(df) * k_pct / 100))
        metrics[f"precision_at_top_{k_pct}pct"] = float(y[order][:k].mean())

    return metrics


def evaluate_against_synthetic_truth(scored: pd.DataFrame, truth_csv) -> dict:
    """Optional: score against the planted labels in the synthetic dataset.

    This is the cleanest possible check -- we know exactly which entities are
    fraudulent because we injected them.
    """
    truth = pd.read_csv(truth_csv)
    df = scored.merge(truth, on="ip", how="inner")
    y = (df["fraud_label"] != "legit").astype(int).to_numpy()
    s = df["fraud_score"].to_numpy()

    if y.sum() == 0:
        return {"error": "no fraud entities present after join"}

    order = np.argsort(-s)
    res = {
        "n_entities": int(len(df)),
        "n_true_fraud": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "recall_at_top_1pct": float(
            y[order][:max(1, len(df) // 100)].sum() / y.sum()),
        "recall_at_top_5pct": float(
            y[order][:max(1, len(df) // 20)].sum() / y.sum()),
    }

    # Per-archetype recall in the top 5% -- shows which fraud types we catch.
    k = max(1, len(df) // 20)
    top = df.iloc[order[:k]]
    for arch in sorted(set(df["fraud_label"]) - {"legit"}):
        total = int((df["fraud_label"] == arch).sum())
        caught = int((top["fraud_label"] == arch).sum())
        res[f"recall_{arch}"] = caught / total if total else float("nan")

    return res

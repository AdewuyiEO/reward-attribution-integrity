"""Sensitivity vs business impact -- the decision layer.

This is the part that separates an analyst from a modeller. A detector that
catches 95% of fraud while blocking 3% of real users is worse than useless: the
churn and support cost exceeds the fraud saved, and the publisher relationship
takes the damage.

The model
---------
For a candidate threshold t, every entity with fraud_score >= t is blocked.

    fraud_prevented   = payouts avoided on blocked fraudulent entities
    false_positive_cost = cost of blocking entities that were legitimate
    net_value         = fraud_prevented - false_positive_cost

Two assumptions drive everything and are exposed in config.py so a stakeholder
can challenge them directly rather than having to read the code:

    PAYOUT_PER_INSTALL   -- what we pay out per attributed install
    FALSE_POSITIVE_COST  -- cost of wrongly blocking a real user
                            (lost lifetime value + trust damage)

FALSE_POSITIVE_COST > PAYOUT_PER_INSTALL is deliberate and is the ethical core
of the model: wrongly denying a real user their reward costs more than letting
one fraudulent install through. Users churn, and publishers notice. Making that
asymmetry explicit is what stops the optimiser from recommending a
scorched-earth threshold.

The output is a curve, not a number. The recommended threshold is the argmax of
net value, but the curve lets a stakeholder choose a more conservative point if
they weight partner trust higher than the model does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def threshold_sweep(
    scored: pd.DataFrame,
    truth: pd.DataFrame,
    payout_per_install: float | None = None,
    false_positive_cost: float | None = None,
    n_steps: int = 60,
) -> pd.DataFrame:
    """Evaluate business impact across the full range of thresholds."""
    payout = payout_per_install or config.PAYOUT_PER_INSTALL
    fp_cost = false_positive_cost or config.FALSE_POSITIVE_COST

    df = scored.merge(
        truth[["ip", "n_conversions", "suspicious_proxy"]], on="ip", how="inner"
    )

    baseline_rate = truth["n_conversions"].sum() / truth["n_clicks"].sum()

    # Expected payout exposure per entity: what we would pay if we let it run.
    # For flagged-fraudulent entities the exposure is the payout they would
    # extract at the population rate had their traffic been honoured.
    df["expected_payout"] = df["n_clicks"] * baseline_rate * payout

    rows = []
    for t in np.linspace(0, 1, n_steps):
        blocked = df["fraud_score"] >= t

        tp = int((blocked & (df["suspicious_proxy"] == 1)).sum())
        fp = int((blocked & (df["suspicious_proxy"] == 0)).sum())
        fn = int((~blocked & (df["suspicious_proxy"] == 1)).sum())
        tn = int((~blocked & (df["suspicious_proxy"] == 0)).sum())

        fraud_prevented = df.loc[blocked & (df["suspicious_proxy"] == 1),
                                 "expected_payout"].sum()
        fp_penalty = fp * fp_cost
        legit_blocked_rate = fp / max(tn + fp, 1)

        rows.append({
            "threshold": t,
            "n_blocked": int(blocked.sum()),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
            "legit_blocked_rate": legit_blocked_rate,
            "fraud_payout_prevented": fraud_prevented,
            "false_positive_cost": fp_penalty,
            "net_value": fraud_prevented - fp_penalty,
        })

    return pd.DataFrame(rows)


def recommend_threshold(sweep: pd.DataFrame,
                        max_legit_blocked: float = 0.01) -> dict:
    """Pick the operating point: maximise net value under a fairness guardrail.

    The guardrail matters. Pure net-value maximisation can quietly accept a
    high false-positive rate if fraud volumes are large. Capping the share of
    legitimate entities blocked keeps the recommendation defensible to a
    publisher, and encodes the principle that real users must not be collateral
    damage. If no threshold satisfies the cap, we say so rather than silently
    relaxing it.
    """
    eligible = sweep[sweep["legit_blocked_rate"] <= max_legit_blocked]
    constrained = True
    if eligible.empty:
        eligible = sweep
        constrained = False

    best = eligible.loc[eligible["net_value"].idxmax()]
    unconstrained_best = sweep.loc[sweep["net_value"].idxmax()]

    return {
        "recommended_threshold": float(best["threshold"]),
        "guardrail_satisfied": constrained,
        "max_legit_blocked_allowed": max_legit_blocked,
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "legit_blocked_rate": float(best["legit_blocked_rate"]),
        "net_value": float(best["net_value"]),
        "entities_blocked": int(best["n_blocked"]),
        "unconstrained_threshold": float(unconstrained_best["threshold"]),
        "unconstrained_net_value": float(unconstrained_best["net_value"]),
    }


def summarise_for_stakeholder(rec: dict, sweep: pd.DataFrame) -> str:
    """One paragraph a non-technical stakeholder can act on."""
    return (
        f"Recommended operating threshold: {rec['recommended_threshold']:.2f}. "
        f"At this setting the system blocks {rec['entities_blocked']:,} entities, "
        f"catching {rec['recall']:.0%} of estimated fraudulent traffic at "
        f"{rec['precision']:.0%} precision, while affecting "
        f"{rec['legit_blocked_rate']:.2%} of legitimate entities "
        f"(guardrail: {rec['max_legit_blocked_allowed']:.0%}). "
        f"Estimated net value protected: ${rec['net_value']:,.0f} over the "
        f"analysed period. Raising the threshold further increases fraud caught "
        f"but pushes legitimate-user impact past the guardrail, which we judge "
        f"a poor trade against partner trust."
    )

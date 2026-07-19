"""Combine the three detectors into one score, with reason codes.

Why rank-normalise before blending
----------------------------------
The detectors emit incomparable quantities: a Euclidean distance, a weighted
robust z-score, a PSI. Averaging them raw would let whichever happens to have
the widest numeric range dominate the result. Converting each to a rank in
[0, 1] first makes the configured weights mean what they say.

Why reason codes
----------------
A bare score of 0.94 is unactionable. An analyst or a partner manager needs to
know *why*, both to trust the flag and to dispute it. Every flagged entity
carries plain-language evidence -- this is what turns a model output into
something a non-technical stakeholder can act on, and it is also what makes an
appeals process possible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .clustering import clustering_score
from .cross_population import cross_population_score
from .distribution import distribution_score

# Thresholds at which a component is considered to have "fired" for reasons.
REASON_RULES = [
    ("z_regularity",       2.5, "metronomic click timing (scripted, not human)"),
    ("z_burstiness",       2.5, "high share of sub-second clicks (machine speed)"),
    ("z_uniform_hours",    2.0, "activity flat around the clock (no sleep cycle)"),
    ("z_narrow_targeting", 2.5, "huge volume concentrated on very few apps"),
]


def _reason_codes(merged: pd.DataFrame) -> pd.Series:
    reasons = [[] for _ in range(len(merged))]

    for col, thresh, text in REASON_RULES:
        if col in merged.columns:
            fired = merged[col].to_numpy() >= thresh
            for i in np.flatnonzero(fired):
                reasons[i].append(text)

    psi_major = merged["psi"].to_numpy() >= 0.25
    for i in np.flatnonzero(psi_major):
        reasons[i].append("hourly profile diverges sharply from population")

    if "db_ring" in merged.columns:
        for i in np.flatnonzero(merged["db_ring"].to_numpy()):
            reasons[i].append("member of a tight behavioural cluster (possible ring)")

    if "db_noise" in merged.columns:
        for i in np.flatnonzero(merged["db_noise"].to_numpy()):
            reasons[i].append("behavioural outlier (density-isolated)")

    return pd.Series(
        ["; ".join(r) if r else "no single dominant signal" for r in reasons],
        index=merged.index,
    )


def ensemble_score(
    features: pd.DataFrame,
    feature_cols: list[str],
    weights: dict | None = None,
) -> pd.DataFrame:
    """Run all three detectors and produce the final scored entity table."""
    weights = weights or config.DETECTOR_WEIGHTS

    clus = clustering_score(features, feature_cols)
    dist = distribution_score(features)
    xpop = cross_population_score(features)

    merged = pd.concat(
        [features[["ip", "n_clicks"]].reset_index(drop=True),
         clus.reset_index(drop=True),
         dist.reset_index(drop=True),
         xpop.reset_index(drop=True)],
        axis=1,
    )

    merged["fraud_score"] = (
        weights["clustering"] * merged["clustering_score"]
        + weights["distribution"] * merged["distribution_score"]
        + weights["cross_population"] * merged["cross_population_score"]
    )

    merged["reason_codes"] = _reason_codes(merged)
    return merged.sort_values("fraud_score", ascending=False).reset_index(drop=True)

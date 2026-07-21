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

    # Honesty flag: anomalous but below the volume needed to prove it alone.
    if "evidence_tier" in merged.columns:
        low = np.isin(merged["evidence_tier"].to_numpy(), ["unproven", "suspected"])
        for i in np.flatnonzero(low):
            reasons[i].append("LOW EVIDENCE: below detectability floor, treat as watchlist")

    return pd.Series(
        ["; ".join(r) if r else "no single dominant signal" for r in reasons],
        index=merged.index,
    )


def assign_evidence(merged: pd.DataFrame, floor: int | None) -> pd.DataFrame:
    """Attach volume_confidence, evidence_tier, and the actionable fraud_priority.

    The problem this solves
    -----------------------
    fraud_score answers "how anomalous is this entity's behaviour?" -- and a
    115-click IP genuinely can look extremely anomalous. But the detectability
    floor says that below ~1,760 clicks (at a 0.2% base rate) we cannot prove a
    single entity is fraudulent from its own record. Ranking such an entity as a
    top hit contradicts our own statistics.

    The distinction that resolves it
    --------------------------------
    * INDIVIDUAL evidence (distribution, cross-population) judges one entity in
      isolation, so it is subject to the floor.
    * COLLECTIVE evidence (DBSCAN ring membership) is a claim about *many*
      entities behaving identically. That coordination does not depend on any
      single entity's volume -- a fleet of 500 low-volume bots is suspicious
      precisely because they move together. Rings are therefore NOT demoted.

    So:
        volume_confidence = min(n_clicks / floor, 1)     for isolated entities
                          = 1                             for ring members
        fraud_priority    = fraud_score * volume_confidence

    fraud_priority -- not fraud_score -- is what we rank and action on. This
    turns the detectability floor from a caveat buried in the README into a
    visible, enforced property of the output.
    """
    n_clicks = merged["n_clicks"].to_numpy(dtype=float)
    is_ring = (merged["db_ring"].to_numpy()
               if "db_ring" in merged.columns else np.zeros(len(merged), bool))

    if floor and floor > 0:
        vol_conf = np.clip(n_clicks / floor, 0.0, 1.0)
        suspected_min = floor * 0.25
    else:                       # no base rate available -> no gating
        vol_conf = np.ones(len(merged))
        suspected_min = 0.0

    # Ring members keep full confidence: collective evidence overrides volume.
    vol_conf = np.where(is_ring, 1.0, vol_conf)

    tier = np.where(
        is_ring, "ring (collective)",
        np.where(n_clicks >= (floor or 0), "proven",
        np.where(n_clicks >= suspected_min, "suspected", "unproven")))

    merged["volume_confidence"] = vol_conf
    merged["evidence_tier"] = tier
    merged["fraud_priority"] = merged["fraud_score"].to_numpy() * vol_conf
    return merged


def ensemble_score(
    features: pd.DataFrame,
    feature_cols: list[str],
    weights: dict | None = None,
    detectability_floor: int | None = None,
) -> pd.DataFrame:
    """Run all three detectors and produce the final scored entity table.

    `detectability_floor` is the minimum click count at which a single entity's
    zero-conversion record is statistically surprising (see evaluate.py). When
    supplied, low-volume isolated entities are demoted via `fraud_priority` so
    they cannot masquerade as top hits. It is a global population constant, not
    a per-entity label, so using it introduces no leakage.
    """
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

    merged = assign_evidence(merged, detectability_floor)
    merged["reason_codes"] = _reason_codes(merged)

    # Rank on fraud_priority (evidence-gated), not raw anomaly score.
    return merged.sort_values(
        ["fraud_priority", "fraud_score"], ascending=False
    ).reset_index(drop=True)

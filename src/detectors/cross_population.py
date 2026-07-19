"""Detector (c): CROSS-POPULATION COMPARISON.

The idea
--------
The previous two detectors ask "is this entity unusual in absolute terms?"
This one asks a sharper question: "does this entity behave like the population
it claims to belong to?"

That reframing matters because it is adaptive. The population baseline is
recomputed every run, so if genuine traffic shifts -- a holiday, a new region,
a big publisher onboarding -- the baseline moves with it and the detector does
not start firing on everyone. A hard-coded rule ("flag any IP over 1,000
clicks") cannot do that and will drown the team in false positives the first
time traffic patterns change.

Three measures against the global hour-of-day distribution:

  * KS statistic -- maximum gap between the entity's CDF and the population
    CDF. Scale-free and non-parametric, so it needs no distributional
    assumption about what "normal" looks like.

  * PSI (population stability index) -- the standard drift metric in credit
    risk and model monitoring. Conventional reading: <0.1 stable,
    0.1-0.25 moderate, >0.25 major divergence. Using the industry-standard
    thresholds means the output is immediately legible to a risk stakeholder.

  * Robust z-score on volume -- how extreme is this entity's click count
    relative to the population median?

Small-sample caution: an entity with 30 clicks will diverge from any baseline
by chance alone. The score is therefore shrunk toward zero for low-volume
entities (`confidence` below), which is the same lesson as putting confidence
intervals on a rate -- do not over-read a small denominator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import ks_statistic, psi, rank_normalise

HOUR_COLS = [f"h{h:02d}" for h in range(24)]

PSI_MODERATE = 0.10
PSI_MAJOR = 0.25


def cross_population_score(
    df: pd.DataFrame,
    baseline_hist: np.ndarray | None = None,
    shrink_at: int = 100,
) -> pd.DataFrame:
    """Compare each entity's temporal profile against the population baseline."""
    hours = df[HOUR_COLS].to_numpy(dtype=float)

    if baseline_hist is None:
        baseline_hist = hours.sum(axis=0)

    ks = ks_statistic(baseline_hist, hours)
    psi_vals = psi(baseline_hist, hours)

    volume = np.log1p(df["n_clicks"].to_numpy(dtype=float))
    med = np.median(volume)
    mad = np.median(np.abs(volume - med)) * 1.4826
    mad = mad if mad > 1e-9 else (volume.std() or 1.0)
    z_volume = np.clip((volume - med) / mad, 0, None)

    # Shrinkage: entities with few clicks cannot support a confident claim of
    # divergence, so their evidence is damped rather than trusted.
    n_clicks = df["n_clicks"].to_numpy(dtype=float)
    confidence = np.clip(n_clicks / shrink_at, 0, 1)

    raw = (0.40 * rank_normalise(ks)
           + 0.40 * rank_normalise(psi_vals)
           + 0.20 * rank_normalise(z_volume)) * confidence

    psi_band = np.where(psi_vals >= PSI_MAJOR, "major",
                np.where(psi_vals >= PSI_MODERATE, "moderate", "stable"))

    return pd.DataFrame({
        "ks_stat": ks,
        "psi": psi_vals,
        "psi_band": psi_band,
        "z_volume": z_volume,
        "cp_confidence": confidence,
        "cross_population_score": rank_normalise(raw),
    }, index=df.index)

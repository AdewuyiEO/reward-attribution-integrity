"""Detector (b): DISTRIBUTION ANALYSIS.

The idea
--------
Ignore *how much* an entity clicks and look at the SHAPE of its behaviour.
Three shape signals, each mapping to a documented fraud technique:

1. Inter-arrival regularity (cv_gap)
   A script firing every ~40 seconds produces a coefficient of variation near
   zero. Human browsing is bursty and irregular, CV around 1 or higher.
   *Unusually low variance is the anomaly* -- the detector therefore inverts
   this feature. This is the single most discriminative signal in the set,
   because it is hard to fake without deliberately adding jitter.

2. Diurnal entropy (hour_entropy)
   Humans sleep. Real traffic dips overnight, giving entropy below the uniform
   maximum. Server-side bots run flat around the clock, pushing entropy toward
   1.0. Entities that are *too uniform* are suspicious.

3. Burstiness (burst_rate / rapid_rate)
   Click injection fires hundreds of clicks in seconds. A high share of
   sub-second gaps is not human motor behaviour.

Scoring uses ROBUST z-scores (median and MAD) rather than mean/std. With
contaminated data the mean and standard deviation are themselves dragged by the
fraud we are hunting -- the contamination hides itself. Median and MAD have a
~50% breakdown point, so they stay stable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import rank_normalise

MAD_SCALE = 1.4826  # makes MAD a consistent estimator of sigma under normality


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * MAD_SCALE
    if mad < 1e-9:
        std = x.std()
        mad = std if std > 1e-9 else 1.0
    return (x - med) / mad


def distribution_score(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-entity distribution anomaly score in [0, 1] plus components."""
    # Regularity: invert, because LOW variability is the tell.
    z_regularity = -_robust_z(df["cv_gap"].to_numpy())

    # Uniformity across the day: high entropy = no sleep cycle.
    z_uniform = _robust_z(df["hour_entropy"].to_numpy())

    # Machine-speed clicking.
    z_burst = _robust_z(df["burst_rate"].to_numpy())
    z_rapid = _robust_z(df["rapid_rate"].to_numpy())

    # Narrow targeting: enormous volume spread over very few apps/channels.
    z_narrow = _robust_z(df["clicks_per_app"].to_numpy())

    components = pd.DataFrame({
        "z_regularity": z_regularity,
        "z_uniform_hours": z_uniform,
        "z_burstiness": z_burst,
        "z_rapid": z_rapid,
        "z_narrow_targeting": z_narrow,
    }, index=df.index)

    # Only positive deviations count -- we care about entities that are
    # anomalous in the fraudulent direction, not merely unusual.
    positive = components.clip(lower=0)

    # Weighted blend; regularity is weighted highest as the strongest signal.
    weights = np.array([0.35, 0.20, 0.20, 0.15, 0.10])
    raw = (positive.to_numpy() * weights).sum(axis=1)

    out = components.copy()
    out["distribution_raw"] = raw
    out["distribution_score"] = rank_normalise(raw)
    return out

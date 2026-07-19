"""Detector (a): CLUSTERING.

The idea
--------
Fraud is manufactured, and manufactured things are uniform. A click farm runs
the same script across its whole fleet, so its entities collapse into a tight,
dense region of behaviour space. Genuine human traffic is diffuse.

Two complementary readings:
  * KMeans  -> distance from an entity to its nearest centroid. Entities far
               from every centroid are behavioural outliers.
  * DBSCAN  -> density-based. Points DBSCAN marks as noise (-1) are isolated
               oddities; unusually tight small clusters are candidate fraud
               rings (many entities behaving identically).

Why RobustScaler rather than StandardScaler: n_clicks is heavy-tailed and a
single 200k-click farm would otherwise dominate the mean and standard
deviation, compressing everyone else into a dot. RobustScaler uses the median
and IQR, so extreme entities cannot distort the scaling of the rest.

Why log1p first: click counts span several orders of magnitude. Without a log
transform, distance in this space is essentially just "click volume", and the
detector stops being multivariate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import RobustScaler

from ..utils import rank_normalise

SKEWED = {"n_clicks", "mean_gap", "std_gap", "median_gap",
          "clicks_per_app", "clicks_per_channel", "clicks_per_hour"}


def _prepare(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X = df[feature_cols].to_numpy(dtype=float).copy()
    for i, col in enumerate(feature_cols):
        if col in SKEWED:
            X[:, i] = np.log1p(np.clip(X[:, i], 0, None))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return RobustScaler().fit_transform(X)


def clustering_score(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 8,
    dbscan_eps: float = 1.2,
    dbscan_min_samples: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return per-entity clustering anomaly score in [0, 1] plus diagnostics."""
    Xs = _prepare(df, feature_cols)
    n = len(df)

    # --- KMeans: distance to nearest centroid ------------------------------
    k = int(min(n_clusters, max(2, n // 20)))
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(Xs)
    dist = np.linalg.norm(Xs - km.cluster_centers_[labels], axis=1)

    # --- DBSCAN: density outliers and tight rings --------------------------
    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(Xs)
    db_labels = db.labels_
    is_noise = db_labels == -1

    # A "ring" = a dense cluster that is small relative to the population.
    # Many entities behaving near-identically is itself suspicious.
    ring_flag = np.zeros(n, dtype=bool)
    for lab in set(db_labels) - {-1}:
        members = db_labels == lab
        if members.sum() <= max(3, 0.02 * n):
            ring_flag |= members

    score = rank_normalise(dist)
    # Density evidence nudges the rank score upward without overwhelming it.
    score = np.clip(score + 0.15 * is_noise + 0.10 * ring_flag, 0, 1)

    return pd.DataFrame({
        "clustering_score": score,
        "km_distance": dist,
        "km_cluster": labels,
        "db_label": db_labels,
        "db_noise": is_noise,
        "db_ring": ring_flag,
    }, index=df.index)

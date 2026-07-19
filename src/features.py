"""Build the entity feature matrix by executing the SQL layer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

HOUR_COLS = [f"h{h:02d}" for h in range(24)]

# The behavioural features handed to the detectors. Deliberately excludes
# anything derived from is_attributed.
FEATURE_COLS = [
    "n_clicks",
    "n_apps",
    "n_channels",
    "n_devices",
    "n_os",
    "n_active_hours",
    "mean_gap",
    "std_gap",
    "median_gap",
    "burst_rate",
    "rapid_rate",
    "clicks_per_app",
    "clicks_per_channel",
    "clicks_per_hour",
    "cv_gap",
    "hour_entropy",
]


def build_entity_features(con, min_clicks: int | None = None) -> pd.DataFrame:
    """Run the entity SQL and attach the derived hour-entropy feature."""
    from .utils import shannon_entropy

    min_clicks = config.MIN_CLICKS if min_clicks is None else min_clicks
    sql = (config.SQL_DIR / "02_entity_features.sql").read_text()
    sql = sql.replace("{min_clicks}", str(min_clicks))

    df = con.execute(sql).df()

    hours = df[HOUR_COLS].to_numpy(dtype=float)
    df["hour_entropy"] = shannon_entropy(hours, axis=1)

    # Sparse entities produce undefined gap statistics; neutral fill.
    for col in ["mean_gap", "std_gap", "median_gap", "cv_gap",
                "burst_rate", "rapid_rate", "clicks_per_hour"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].median())

    return df


def load_label_aggregates(con) -> pd.DataFrame:
    """Evaluation-only label counts. Joined at scoring time, never at fit time."""
    sql = (config.SQL_DIR / "03_label_aggregates.sql").read_text()
    return con.execute(sql).df()


def population_hour_histogram(df: pd.DataFrame) -> np.ndarray:
    """Global hour-of-day baseline: the 'population' in cross-population."""
    return df[HOUR_COLS].to_numpy(dtype=float).sum(axis=0)

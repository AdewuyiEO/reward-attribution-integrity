"""Tests for the reward-integrity pipeline.

The leakage tests are the important ones. Everything else can be re-run; a
leaked label silently invalidates every number in the project.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import features  # noqa: E402
from src.evaluate import detectability_floor  # noqa: E402
from src.utils import (benjamini_hochberg, ks_statistic, psi,  # noqa: E402
                       rank_normalise, shannon_entropy, wilson_ci)


# ---------------------------------------------------------------------------
# Leakage guards
# ---------------------------------------------------------------------------
def test_no_label_columns_in_feature_set():
    """The detector must never see the outcome."""
    banned = {"is_attributed", "attributed_time", "n_conversions", "conv_rate"}
    assert banned.isdisjoint(set(features.FEATURE_COLS))


def test_entity_sql_does_not_reference_labels():
    sql = (Path(__file__).resolve().parents[1] / "sql" / "02_entity_features.sql").read_text()
    body = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    assert "is_attributed" not in body
    assert "attributed_time" not in body


def test_ingest_sql_excludes_attributed_time():
    sql = (Path(__file__).resolve().parents[1] / "src" / "ingest.py").read_text()
    create = sql.split("CREATE TABLE")[1].split("read_csv_auto")[0]
    assert "attributed_time" not in create


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def test_wilson_ci_contains_point_estimate():
    lo, hi = wilson_ci([3], [900])
    assert lo[0] < 3 / 900 < hi[0]


def test_wilson_ci_wider_for_small_samples():
    """The whole hour-chart lesson, encoded as a test."""
    lo_small, hi_small = wilson_ci([3], [900])
    lo_big, hi_big = wilson_ci([300], [90000])
    assert (hi_small - lo_small)[0] > (hi_big - lo_big)[0]


def test_small_and_large_sample_intervals_overlap():
    lo1, hi1 = wilson_ci([3], [900])       # 0.0033
    lo2, hi2 = wilson_ci([14], [5600])     # 0.0025
    assert lo1[0] < hi2[0] and lo2[0] < hi1[0]


def test_entropy_bounds():
    uniform = np.ones((1, 24))
    spike = np.zeros((1, 24)); spike[0, 0] = 100
    assert shannon_entropy(uniform)[0] == pytest.approx(1.0, abs=1e-9)
    assert shannon_entropy(spike)[0] == pytest.approx(0.0, abs=1e-9)


def test_psi_zero_for_identical_distributions():
    base = np.random.default_rng(0).integers(10, 100, size=24).astype(float)
    assert psi(base, base)[0] == pytest.approx(0.0, abs=1e-8)


def test_psi_positive_for_divergent_distribution():
    base = np.ones(24) * 100
    weird = np.zeros(24); weird[:3] = 100
    assert psi(base, weird)[0] > 0.25


def test_ks_statistic_range():
    base = np.ones(24)
    same = ks_statistic(base, np.ones((1, 24)))[0]
    shifted = np.zeros(24); shifted[-1] = 24
    diff = ks_statistic(base, shifted[None, :])[0]
    assert same == pytest.approx(0.0, abs=1e-9)
    assert 0 < diff <= 1


def test_rank_normalise_range():
    out = rank_normalise(np.array([5.0, 1.0, 3.0, 9.0]))
    assert out.min() == 0.0 and out.max() == 1.0


def test_benjamini_hochberg_is_conservative():
    """FDR-adjusted p-values must never be smaller than the raw ones."""
    p = np.array([0.001, 0.01, 0.03, 0.2, 0.5])
    rejected, adj = benjamini_hochberg(p, q=0.05)
    assert np.all(adj >= p - 1e-12)
    assert rejected[0]


def test_benjamini_hochberg_controls_false_discoveries():
    """Under the null, BH should reject almost nothing."""
    rng = np.random.default_rng(7)
    p = rng.uniform(size=5000)
    rejected, _ = benjamini_hochberg(p, q=0.05)
    assert rejected.sum() < 50


def test_detectability_floor_matches_closed_form():
    """At a 0.2% baseline you need ~1500 clicks to prove a zero-conversion deficit."""
    floor = detectability_floor(0.002, alpha=0.05)
    assert 1400 < floor < 1600


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------
def _toy_entities(n=300, seed=0):
    rng = np.random.default_rng(seed)
    hours = rng.multinomial(200, np.r_[np.ones(6) * 0.01, np.ones(18) * 0.0522], size=n)
    df = pd.DataFrame({
        "ip": np.arange(n),
        "n_clicks": rng.integers(50, 500, n).astype(float),
        "n_apps": rng.integers(5, 40, n).astype(float),
        "n_channels": rng.integers(5, 60, n).astype(float),
        "n_devices": rng.integers(1, 5, n).astype(float),
        "n_os": rng.integers(1, 20, n).astype(float),
        "n_active_hours": rng.integers(10, 24, n).astype(float),
        "mean_gap": rng.uniform(50, 400, n),
        "std_gap": rng.uniform(40, 400, n),
        "median_gap": rng.uniform(40, 300, n),
        "burst_rate": rng.uniform(0, 0.05, n),
        "rapid_rate": rng.uniform(0, 0.1, n),
        "clicks_per_app": rng.uniform(2, 20, n),
        "clicks_per_channel": rng.uniform(2, 20, n),
        "clicks_per_hour": rng.uniform(1, 30, n),
        "cv_gap": rng.uniform(0.8, 1.6, n),
    })
    for h in range(24):
        df[f"h{h:02d}"] = hours[:, h].astype(float)
    df["hour_entropy"] = shannon_entropy(df[[f"h{h:02d}" for h in range(24)]].to_numpy())
    return df


def test_distribution_detector_flags_metronomic_entity():
    from src.detectors import distribution_score
    df = _toy_entities()
    df.loc[0, "cv_gap"] = 0.01          # perfectly regular = scripted
    df.loc[0, "burst_rate"] = 0.9
    out = distribution_score(df)
    assert out["distribution_score"].iloc[0] > 0.98


def test_cross_population_flags_divergent_hours():
    from src.detectors import cross_population_score
    df = _toy_entities()
    for h in range(24):
        df.loc[0, f"h{h:02d}"] = 0.0
    df.loc[0, "h03"] = 5000.0            # all activity in one dead hour
    df.loc[0, "n_clicks"] = 5000.0
    out = cross_population_score(df)
    assert out["cross_population_score"].iloc[0] > 0.95
    assert out["psi_band"].iloc[0] == "major"


def test_cross_population_shrinks_low_volume_entities():
    """A 30-click entity must not be confidently condemned."""
    from src.detectors import cross_population_score
    df = _toy_entities()
    df.loc[0, "n_clicks"] = 30.0
    out = cross_population_score(df)
    assert out["cp_confidence"].iloc[0] < 0.5


def test_ensemble_produces_scores_and_reasons():
    from src.detectors import ensemble_score
    df = _toy_entities()
    df.loc[0, "cv_gap"] = 0.01
    df.loc[0, "burst_rate"] = 0.95
    out = ensemble_score(df, features.FEATURE_COLS)
    assert out["fraud_score"].between(0, 1).all()
    assert len(out) == len(df)
    assert out["reason_codes"].iloc[0] != ""


def test_evidence_gate_demotes_low_volume_anomaly():
    """The reported bug: a high-anomaly, low-volume entity must not rank top."""
    from src.detectors import ensemble_score
    df = _toy_entities()
    # Make entity 0 behaviourally extreme but tiny in volume.
    df.loc[0, "cv_gap"] = 0.01
    df.loc[0, "burst_rate"] = 0.95
    df.loc[0, "n_clicks"] = 115.0          # far below any realistic floor

    out = ensemble_score(df, features.FEATURE_COLS, detectability_floor=1760)
    row = out[out["ip"] == 0].iloc[0]

    assert row["evidence_tier"] == "unproven"
    # Its raw anomaly stays high, but its actionable priority is crushed.
    assert row["fraud_score"] > 0.5
    assert row["fraud_priority"] < 0.1
    # And it is no longer the top-ranked entity.
    assert out.iloc[0]["ip"] != 0 or out.iloc[0]["n_clicks"] >= 1760


def test_evidence_gate_protects_high_volume():
    """A genuinely high-volume anomaly must keep full priority."""
    from src.detectors import ensemble_score
    df = _toy_entities()
    df.loc[0, "cv_gap"] = 0.01
    df.loc[0, "burst_rate"] = 0.95
    df.loc[0, "n_clicks"] = 20000.0
    out = ensemble_score(df, features.FEATURE_COLS, detectability_floor=1760)
    row = out[out["ip"] == 0].iloc[0]
    assert row["evidence_tier"] == "proven"
    assert row["volume_confidence"] == 1.0
    assert abs(row["fraud_priority"] - row["fraud_score"]) < 1e-9


def test_no_gate_without_floor():
    """Without a floor, priority equals score (backward compatible)."""
    from src.detectors import ensemble_score
    df = _toy_entities()
    out = ensemble_score(df, features.FEATURE_COLS)
    assert (out["volume_confidence"] == 1.0).all()
    assert (out["fraud_priority"] - out["fraud_score"]).abs().max() < 1e-9


def test_cost_model_penalises_overblocking():
    """A threshold of 0 blocks everyone and must not be recommended."""
    from src.cost_model import recommend_threshold, threshold_sweep
    rng = np.random.default_rng(3)
    n = 500
    scored = pd.DataFrame({
        "ip": np.arange(n),
        "fraud_score": rng.uniform(0, 1, n),
        "n_clicks": rng.integers(100, 5000, n),
    })
    truth = pd.DataFrame({
        "ip": np.arange(n),
        "n_clicks": scored["n_clicks"],
        "n_conversions": rng.integers(0, 5, n),
        "suspicious_proxy": (rng.uniform(size=n) < 0.05).astype(int),
    })
    sweep = threshold_sweep(scored, truth)
    rec = recommend_threshold(sweep)
    assert rec["recommended_threshold"] > 0.0
    assert rec["legit_blocked_rate"] <= rec["max_legit_blocked_allowed"] + 1e-9

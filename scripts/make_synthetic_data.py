"""Generate a TalkingData-schema dataset with KNOWN injected fraud.

Why this exists
---------------
The real train.csv is ~7GB / 200M rows. This generator produces the same schema
with three planted fraud archetypes, so you can:
  1. run the whole pipeline today, before the download finishes
  2. verify the detectors actually catch fraud you *know* is there
  3. keep a fast test fixture for CI

Planted archetypes (each maps to a real mobile-fraud pattern):
  * click_farm     - huge click volume, near-zero installs, round-the-clock
  * bot_regular    - metronomic inter-arrival times (scripted, not human)
  * burst_injector - short violent bursts of clicks (click injection)

Usage:
    python scripts/make_synthetic_data.py --rows 400000 --out data/train_synthetic.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_CONV_RATE = 0.0023          # matches TalkingData's real ~0.23%
START = pd.Timestamp("2017-11-07 00:00:00")


def _legit_clicks(n: int, rng: np.random.Generator, ip_pool: np.ndarray) -> pd.DataFrame:
    """Normal human traffic: diurnal rhythm, irregular gaps, baseline conversion."""
    # Diurnal curve - low overnight, peaks late morning and evening.
    hours = np.arange(24)
    shape = 0.35 + 0.65 * (np.sin((hours - 8) / 24 * 2 * np.pi) + 1) / 2
    shape[0:6] *= 0.25                      # people sleep
    probs = shape / shape.sum()

    hour = rng.choice(hours, size=n, p=probs)
    day = rng.integers(0, 3, size=n)
    secs = day * 86400 + hour * 3600 + rng.integers(0, 3600, size=n)

    df = pd.DataFrame({
        "ip": rng.choice(ip_pool, size=n),
        "app": rng.integers(1, 60, size=n),
        "device": rng.choice([1, 2, 1, 1, 3, 6], size=n),
        "os": rng.integers(1, 40, size=n),
        "channel": rng.integers(1, 150, size=n),
        "click_time": START + pd.to_timedelta(secs, unit="s"),
    })
    df["is_attributed"] = (rng.random(n) < BASE_CONV_RATE).astype("uint8")
    df["fraud_label"] = "legit"
    return df


def _click_farm(rng, ip, n) -> pd.DataFrame:
    """Massive volume, flat around the clock, essentially never converts."""
    secs = rng.integers(0, 3 * 86400, size=n)          # uniform => no diurnal dip
    df = pd.DataFrame({
        "ip": ip,
        "app": rng.integers(1, 12, size=n),             # narrow app range
        "device": rng.choice([1, 2], size=n),
        "os": rng.integers(1, 8, size=n),
        "channel": rng.choice([101, 102, 103], size=n),  # few channels
        "click_time": START + pd.to_timedelta(secs, unit="s"),
    })
    df["is_attributed"] = (rng.random(n) < 0.00005).astype("uint8")
    df["fraud_label"] = "click_farm"
    return df


def _bot_regular(rng, ip, n) -> pd.DataFrame:
    """Scripted clicker: near-constant gap between clicks. Humans never do this."""
    gap = rng.integers(25, 60)
    jitter = rng.normal(0, 0.6, size=n).cumsum()
    secs = np.clip(np.arange(n) * gap + jitter, 0, 3 * 86400)
    df = pd.DataFrame({
        "ip": ip,
        "app": rng.integers(1, 20, size=n),
        "device": 1,
        "os": rng.integers(1, 10, size=n),
        "channel": rng.integers(100, 140, size=n),
        "click_time": START + pd.to_timedelta(secs.astype(int), unit="s"),
    })
    df["is_attributed"] = (rng.random(n) < 0.0001).astype("uint8")
    df["fraud_label"] = "bot_regular"
    return df


def _burst_injector(rng, ip, n) -> pd.DataFrame:
    """Click injection: silent, then hundreds of clicks in seconds, then silent."""
    n_bursts = max(2, n // 150)
    starts = rng.integers(0, 3 * 86400, size=n_bursts)
    per = max(1, n // n_bursts)
    secs = np.concatenate([s + rng.integers(0, 30, size=per) for s in starts])[:n]
    df = pd.DataFrame({
        "ip": ip,
        "app": rng.integers(1, 15, size=len(secs)),
        "device": rng.choice([1, 2, 3], size=len(secs)),
        "os": rng.integers(1, 12, size=len(secs)),
        "channel": rng.integers(100, 130, size=len(secs)),
        "click_time": START + pd.to_timedelta(secs, unit="s"),
    })
    df["is_attributed"] = (rng.random(len(secs)) < 0.0002).astype("uint8")
    df["fraud_label"] = "burst_injector"
    return df


def generate(rows: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    n_fraud_target = int(rows * 0.25)
    n_legit = rows - n_fraud_target
    ip_pool = rng.integers(1_000, 90_000, size=max(2000, n_legit // 40))

    frames = [_legit_clicks(n_legit, rng, ip_pool)]

    # Fraudulent entities get their own dedicated IP space (>= 500000) so the
    # ground truth is unambiguous when we score the detectors.
    fraud_ip = 500_000
    produced = 0
    makers = [_click_farm, _bot_regular, _burst_injector]
    i = 0
    while produced < n_fraud_target:
        maker = makers[i % 3]
        n = int(rng.integers(2500, 12000))
        n = min(n, n_fraud_target - produced)
        if n < 500:
            break
        frames.append(maker(rng, fraud_ip, n))
        produced += n
        fraud_ip += 1
        i += 1

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("click_time").reset_index(drop=True)

    # attributed_time is populated only for conversions -- replicating the exact
    # leakage trap present in the real dataset.
    df["attributed_time"] = pd.NaT
    conv = df["is_attributed"] == 1
    df.loc[conv, "attributed_time"] = df.loc[conv, "click_time"] + pd.to_timedelta(
        rng.integers(30, 4000, size=int(conv.sum())), unit="s"
    )

    return df[["ip", "app", "device", "os", "channel", "click_time",
               "attributed_time", "is_attributed", "fraud_label"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/train_synthetic.csv"))
    args = ap.parse_args()

    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # fraud_label is ground truth for *validation only*. It is written to a
    # separate file so it can never accidentally be read as a feature.
    truth = df[["ip", "fraud_label"]].drop_duplicates("ip")
    truth.to_csv(args.out.parent / "synthetic_truth.csv", index=False)

    df.drop(columns=["fraud_label"]).to_csv(args.out, index=False)

    print(f"rows written      : {len(df):,}")
    print(f"conversion rate   : {df['is_attributed'].mean():.5f}")
    print(f"fraud entities    : {(truth['fraud_label'] != 'legit').sum():,}")
    print(f"legit entities    : {(truth['fraud_label'] == 'legit').sum():,}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

"""End-to-end run: ingest -> features -> detect -> evaluate -> cost -> artefacts.

Run:
    python -m src.pipeline
    python -m src.pipeline --csv data/train.csv --min-clicks 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config, features, ingest
from .cost_model import recommend_threshold, summarise_for_stakeholder, threshold_sweep
from .detectors import ensemble_score
from .evaluate import (build_ground_truth_proxy, detectability_floor,
                       evaluate_against_synthetic_truth,
                       evaluate_scores)


def run(csv_path=None, min_clicks=None, quiet: bool = False) -> dict:
    csv_path = Path(csv_path or config.SOURCE_CSV)
    min_clicks = min_clicks or config.MIN_CLICKS
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    def log(msg):
        if not quiet:
            print(msg, flush=True)

    # -- 1. Ingest ---------------------------------------------------------
    log(f"[1/6] Loading {csv_path} into DuckDB ...")
    con = ingest.connect()
    n_rows = ingest.load_csv(con, csv_path)
    stats = ingest.summarise(con)
    log(f"      rows={stats['n_rows']:,}  ips={stats['n_ips']:,}  "
        f"conv_rate={stats['conv_rate']:.5f}  conversions={stats['n_conversions']:,}")

    # -- 2. Features -------------------------------------------------------
    log(f"[2/6] Building entity features (min_clicks={min_clicks}) ...")
    feats = features.build_entity_features(con, min_clicks=min_clicks)
    log(f"      entities={len(feats):,}  features={len(features.FEATURE_COLS)}")

    # -- 3. Detect ---------------------------------------------------------
    # Compute the detectability floor first (a global population constant, not a
    # per-entity label) so the ensemble can evidence-gate low-volume entities.
    labels = features.load_label_aggregates(con)
    base_rate = labels["n_conversions"].sum() / labels["n_clicks"].sum()
    floor = detectability_floor(base_rate)
    log(f"[3/6] Detectability floor = {floor:,} clicks (base rate {base_rate:.5f})")
    log("      Running detectors: clustering | distribution | cross-population ...")
    scored = ensemble_score(feats, features.FEATURE_COLS, detectability_floor=floor)

    tier_counts = scored["evidence_tier"].value_counts().to_dict()
    log(f"      Evidence tiers: {tier_counts}")

    # -- 4. Evaluate -------------------------------------------------------
    log("[4/6] Evaluating against held-out labels ...")
    truth = build_ground_truth_proxy(labels)
    metrics = evaluate_scores(scored, truth)

    synth_truth = csv_path.parent / "synthetic_truth.csv"
    synth_metrics = None
    if synth_truth.exists():
        synth_metrics = evaluate_against_synthetic_truth(scored, synth_truth)

    # -- 5. Cost model -----------------------------------------------------
    log("[5/6] Sweeping thresholds for business impact ...")
    sweep = threshold_sweep(scored, truth)
    rec = recommend_threshold(sweep)
    statement = summarise_for_stakeholder(rec, sweep)

    # -- 6. Persist --------------------------------------------------------
    log("[6/6] Writing artefacts ...")
    scored_out = scored.merge(
        truth[["ip", "n_conversions", "conv_rate", "suspicious_proxy"]],
        on="ip", how="left")
    scored_out.to_parquet(config.OUT_DIR / "scored_entities.parquet", index=False)
    sweep.to_csv(config.OUT_DIR / "threshold_sweep.csv", index=False)
    feats.to_parquet(config.OUT_DIR / "entity_features.parquet", index=False)

    report = {
        "source": str(csv_path),
        "data": {k: (str(v) if hasattr(v, "isoformat") else v)
                 for k, v in stats.items()},
        "n_entities": int(len(feats)),
        "evaluation_vs_proxy": metrics,
        "evaluation_vs_synthetic_truth": synth_metrics,
        "recommendation": rec,
        "stakeholder_statement": statement,
    }
    (config.OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str))

    con.close()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="path to clicks csv")
    ap.add_argument("--min-clicks", type=int, default=None)
    args = ap.parse_args()

    report = run(args.csv, args.min_clicks)

    print("\n" + "=" * 72)
    print("EVALUATION (unsupervised score vs held-out label proxy)")
    print("=" * 72)
    for k, v in report["evaluation_vs_proxy"].items():
        print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")

    if report["evaluation_vs_synthetic_truth"]:
        print("\n" + "=" * 72)
        print("EVALUATION (vs planted synthetic ground truth)")
        print("=" * 72)
        for k, v in report["evaluation_vs_synthetic_truth"].items():
            print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")

    print("\n" + "=" * 72)
    print("RECOMMENDATION")
    print("=" * 72)
    print(report["stakeholder_statement"])
    print()


if __name__ == "__main__":
    main()

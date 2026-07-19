"""Airflow DAG: daily reward-integrity scan.

Why this exists
---------------
The JD asks for people who "think beyond one-off analysis and build reusable
frameworks, automated pipelines, and data visibility solutions". A notebook is
a one-off analysis. This is the same logic on a schedule, with dependencies,
retries, and an alert when something moves.

Task graph:

    ingest -> build_features -> score_entities -> evaluate -> cost_model
                                                                  |
                                                          detect_fraud_spike
                                                                  |
                                                          publish_dashboard

Design notes worth defending:
* Tasks are thin wrappers around `src/` functions. Business logic never lives
  in the DAG file, so everything stays unit-testable outside Airflow.
* State passes through the filesystem (parquet), not XCom. XCom is for small
  metadata; shipping a scored entity table through it would be an abuse of it.
* `detect_fraud_spike` compares today's flagged rate against a trailing
  baseline. Monitoring the DETECTOR is as important as the detector itself --
  a silent drop to zero flags is usually a broken pipeline, not a quiet day.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from airflow import DAG  # noqa: E402
from airflow.operators.python import PythonOperator  # noqa: E402

DEFAULT_ARGS = {
    "owner": "anti-fraud",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

SPIKE_THRESHOLD = 1.5   # flagged rate 1.5x the trailing baseline -> alert
DROP_THRESHOLD = 0.5    # flagged rate below half baseline -> likely pipeline break


# --------------------------------------------------------------------------
# Task callables
# --------------------------------------------------------------------------
def _ingest(**context):
    from src import config, ingest
    con = ingest.connect()
    n = ingest.load_csv(con, config.SOURCE_CSV)
    stats = ingest.summarise(con)
    con.close()
    return {"rows": n, "conv_rate": float(stats["conv_rate"])}


def _build_features(**context):
    from src import config, features, ingest
    con = ingest.connect()
    feats = features.build_entity_features(con)
    feats.to_parquet(config.OUT_DIR / "entity_features.parquet", index=False)
    con.close()
    return {"entities": len(feats)}


def _score_entities(**context):
    from src import config, features
    from src.detectors import ensemble_score
    import pandas as pd

    feats = pd.read_parquet(config.OUT_DIR / "entity_features.parquet")
    scored = ensemble_score(feats, features.FEATURE_COLS)
    scored.to_parquet(config.OUT_DIR / "scored_entities.parquet", index=False)
    return {"scored": len(scored)}


def _evaluate(**context):
    from src import config, features, ingest
    from src.evaluate import build_ground_truth_proxy, evaluate_scores
    import pandas as pd

    con = ingest.connect()
    labels = features.load_label_aggregates(con)
    con.close()

    truth = build_ground_truth_proxy(labels)
    truth.to_parquet(config.OUT_DIR / "label_proxy.parquet", index=False)

    scored = pd.read_parquet(config.OUT_DIR / "scored_entities.parquet")
    metrics = evaluate_scores(scored, truth)
    (config.OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def _cost_model(**context):
    from src import config
    from src.cost_model import (recommend_threshold, summarise_for_stakeholder,
                                threshold_sweep)
    import pandas as pd

    scored = pd.read_parquet(config.OUT_DIR / "scored_entities.parquet")
    truth = pd.read_parquet(config.OUT_DIR / "label_proxy.parquet")

    sweep = threshold_sweep(scored, truth)
    sweep.to_csv(config.OUT_DIR / "threshold_sweep.csv", index=False)

    rec = recommend_threshold(sweep)
    rec["statement"] = summarise_for_stakeholder(rec, sweep)
    (config.OUT_DIR / "recommendation.json").write_text(json.dumps(rec, indent=2))
    return rec


def _detect_fraud_spike(**context):
    """Monitor the monitor. Alert on a spike OR an implausible drop."""
    from src import config
    import pandas as pd

    scored = pd.read_parquet(config.OUT_DIR / "scored_entities.parquet")
    rec = json.loads((config.OUT_DIR / "recommendation.json").read_text())
    threshold = rec["recommended_threshold"]

    flagged_rate = float((scored["fraud_score"] >= threshold).mean())

    history_path = config.OUT_DIR / "flagged_rate_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []

    baseline = (sum(h["flagged_rate"] for h in history[-7:]) / len(history[-7:])
                if history else flagged_rate)

    alert = None
    if history:
        if flagged_rate > baseline * SPIKE_THRESHOLD:
            alert = f"FRAUD SPIKE: flagged rate {flagged_rate:.4f} vs baseline {baseline:.4f}"
        elif flagged_rate < baseline * DROP_THRESHOLD:
            alert = f"SUSPICIOUS DROP: flagged rate {flagged_rate:.4f} vs baseline {baseline:.4f} (check pipeline health)"

    history.append({
        "date": str(context.get("ds", datetime.utcnow().date())),
        "flagged_rate": flagged_rate,
    })
    history_path.write_text(json.dumps(history[-90:], indent=2))

    if alert:
        print(f"[ALERT] {alert}")
    return {"flagged_rate": flagged_rate, "baseline": baseline, "alert": alert}


def _publish_dashboard(**context):
    from src import config
    marker = config.OUT_DIR / "dashboard_refreshed_at.txt"
    marker.write_text(datetime.utcnow().isoformat())
    return "ok"


# --------------------------------------------------------------------------
# DAG definition
# --------------------------------------------------------------------------
with DAG(
    dag_id="reward_integrity_daily",
    description="Daily unsupervised fraud scan over reward attribution traffic",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2024, 1, 1),
    schedule="0 3 * * *",     # 03:00 daily, after upstream logs settle
    catchup=False,
    max_active_runs=1,
    tags=["anti-fraud", "anomaly-detection"],
) as dag:

    tasks = {}
    for task_id, fn in [
        ("ingest", _ingest),
        ("build_features", _build_features),
        ("score_entities", _score_entities),
        ("evaluate", _evaluate),
        ("cost_model", _cost_model),
        ("detect_fraud_spike", _detect_fraud_spike),
        ("publish_dashboard", _publish_dashboard),
    ]:
        tasks[task_id] = PythonOperator(task_id=task_id, python_callable=fn)

    (tasks["ingest"] >> tasks["build_features"] >> tasks["score_entities"]
     >> tasks["evaluate"] >> tasks["cost_model"]
     >> tasks["detect_fraud_spike"] >> tasks["publish_dashboard"])

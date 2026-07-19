"""Central configuration.

Single place to switch between synthetic data (for development) and the real
TalkingData train.csv (~200M rows). Nothing else in the codebase needs to change.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
SQL_DIR = ROOT / "sql"

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
# Point this at data/train.csv once your download finishes.
# Keep data/train_synthetic.csv for fast iteration and CI.
SOURCE_CSV = DATA_DIR / "train_synthetic.csv"
DUCKDB_PATH = DATA_DIR / "fraud.duckdb"

# ---------------------------------------------------------------------------
# Memory optimisation
# ---------------------------------------------------------------------------
# TalkingData ships every id as int64 by default (8 bytes each). The real
# cardinalities are small, so we downcast aggressively. On the full 200M-row
# file this is the difference between "fits on a laptop" and "does not".
#   ip      max ~364k   -> uint32 (4 bytes)
#   app     max ~768    -> uint16 (2 bytes)
#   device  max ~4k     -> uint16 (2 bytes)
#   os      max ~956    -> uint16 (2 bytes)
#   channel max ~500    -> uint16 (2 bytes)
CSV_DTYPES = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
    "is_attributed": "uint8",
}
DATE_COLS = ["click_time"]

# `attributed_time` is populated ONLY when is_attributed == 1.
# It is perfect target leakage. It is never loaded as a feature.
LEAKAGE_COLS = ["attributed_time"]

# ---------------------------------------------------------------------------
# Entity definition
# ---------------------------------------------------------------------------
# The unit of analysis. Fraud lives in entities, not in individual clicks.
ENTITY_KEYS = ["ip"]          # alternatives: ["ip","device","os"], ["channel"]
MIN_CLICKS = 30               # entities below this are too sparse to judge

# ---------------------------------------------------------------------------
# Detector weights (ensemble)
# ---------------------------------------------------------------------------
DETECTOR_WEIGHTS = {
    "clustering": 0.30,
    "distribution": 0.40,
    "cross_population": 0.30,
}

# ---------------------------------------------------------------------------
# Cost model (business impact)
# ---------------------------------------------------------------------------
# These are the assumptions a stakeholder will challenge, so they live in one
# visible place rather than buried in a notebook.
PAYOUT_PER_INSTALL = 2.00     # USD paid out per attributed install
FALSE_POSITIVE_COST = 5.00    # USD cost of wrongly blocking a real user
                              # (lost lifetime value + trust damage)

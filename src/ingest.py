"""Load raw clicks into DuckDB with aggressive type optimisation.

Design decision worth defending in an interview
-----------------------------------------------
We do NOT read 200M rows into pandas. DuckDB streams the CSV from disk and
performs the aggregation out-of-core, so peak memory stays bounded regardless
of file size. pandas is used only for the small per-entity result set.

We still measure the pandas dtype saving (`benchmark_dtypes`) because that is
the concrete number to quote: "downcasting the five id columns from int64 cuts
the frame by ~70%."
"""
from __future__ import annotations

import duckdb
import pandas as pd

from . import config


def connect(db_path=None) -> duckdb.DuckDBPyConnection:
    path = str(db_path or config.DUCKDB_PATH)
    con = duckdb.connect(path)
    con.execute("PRAGMA threads=4")
    return con


def load_csv(con: duckdb.DuckDBPyConnection, csv_path=None, table: str = "clicks") -> int:
    """Stream the CSV into a typed DuckDB table.

    Note the explicit column types: DuckDB would otherwise infer BIGINT for
    every id. Note also that `attributed_time` is deliberately NOT loaded --
    it is non-null exactly when is_attributed == 1, making it perfect target
    leakage. Excluding it at load time means it cannot leak by accident later.
    """
    csv_path = str(csv_path or config.SOURCE_CSV)

    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(
        f"""
        CREATE TABLE {table} AS
        SELECT
            CAST(ip      AS UINTEGER)  AS ip,
            CAST(app     AS USMALLINT) AS app,
            CAST(device  AS USMALLINT) AS device,
            CAST(os      AS USMALLINT) AS os,
            CAST(channel AS USMALLINT) AS channel,
            CAST(click_time AS TIMESTAMP) AS click_time,
            CAST(is_attributed AS UTINYINT) AS is_attributed
        FROM read_csv_auto('{csv_path}', header=true)
        """
    )
    n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]

    con.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_ip ON {table}(ip)")
    return n


def summarise(con: duckdb.DuckDBPyConnection, table: str = "clicks") -> dict:
    """Headline numbers you must know before modelling anything."""
    row = con.execute(
        f"""
        SELECT count(*)                       AS n_rows,
               sum(is_attributed)             AS n_conversions,
               avg(is_attributed)             AS conv_rate,
               count(DISTINCT ip)             AS n_ips,
               min(click_time)                AS t_min,
               max(click_time)                AS t_max
        FROM {table}
        """
    ).fetchone()
    return {
        "n_rows": row[0],
        "n_conversions": row[1],
        "conv_rate": row[2],
        "n_ips": row[3],
        "t_min": row[4],
        "t_max": row[5],
    }


def benchmark_dtypes(csv_path=None, nrows: int = 200_000) -> dict:
    """Quantify the memory saved by downcasting. This is the number to quote."""
    csv_path = str(csv_path or config.SOURCE_CSV)
    cols = ["ip", "app", "device", "os", "channel", "click_time", "is_attributed"]

    naive = pd.read_csv(csv_path, nrows=nrows, usecols=cols)
    naive_mb = naive.memory_usage(deep=True).sum() / 1024**2

    tuned = pd.read_csv(
        csv_path, nrows=nrows, usecols=cols,
        dtype=config.CSV_DTYPES, parse_dates=config.DATE_COLS,
    )
    tuned_mb = tuned.memory_usage(deep=True).sum() / 1024**2

    return {
        "rows": len(naive),
        "naive_mb": naive_mb,
        "tuned_mb": tuned_mb,
        "reduction_pct": 100 * (1 - tuned_mb / naive_mb),
        "projected_naive_gb_at_200m": naive_mb / len(naive) * 200e6 / 1024,
        "projected_tuned_gb_at_200m": tuned_mb / len(naive) * 200e6 / 1024,
    }

-- =============================================================================
-- Typed load. See src/ingest.py for the executed version.
-- =============================================================================
-- Every id column is cast down from the default BIGINT:
--   ip -> UINTEGER (4B), app/device/os/channel -> USMALLINT (2B)
-- On 200M rows that is roughly a 70% reduction in the scanned footprint.
--
-- attributed_time is intentionally absent. It is non-null exactly when
-- is_attributed = 1, so any model given it scores ~1.00 AUC and has learned
-- nothing. Excluding it at load time makes the mistake structurally impossible.
-- =============================================================================
CREATE OR REPLACE TABLE clicks AS
SELECT
    CAST(ip      AS UINTEGER)  AS ip,
    CAST(app     AS USMALLINT) AS app,
    CAST(device  AS USMALLINT) AS device,
    CAST(os      AS USMALLINT) AS os,
    CAST(channel AS USMALLINT) AS channel,
    CAST(click_time AS TIMESTAMP) AS click_time,
    CAST(is_attributed AS UTINYINT) AS is_attributed
FROM read_csv_auto('data/train.csv', header=true);

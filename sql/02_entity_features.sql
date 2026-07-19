-- =============================================================================
-- Entity-level behavioural features
-- =============================================================================
-- Unit of analysis: the ENTITY (here: ip), not the click. Fraud is a property
-- of an actor, not of a single event. A single click is almost never
-- identifiable as fraudulent; an actor producing 4,000 clicks at 38-second
-- intervals with zero installs is unmistakable.
--
-- CRITICAL: every feature below is computed from BEHAVIOUR ONLY.
-- No feature touches is_attributed or attributed_time. The labels are held
-- back entirely for evaluation, which is what makes the evaluation honest --
-- and what mirrors production, where labels arrive late or never.
-- =============================================================================

WITH ordered AS (
    -- Window function over each entity's click sequence to derive the gap
    -- between consecutive clicks. This is the backbone of timing analysis.
    SELECT
        ip,
        click_time,
        EXTRACT(hour FROM click_time) AS hr,
        app,
        channel,
        device,
        os,
        epoch(click_time) - LAG(epoch(click_time))
            OVER (PARTITION BY ip ORDER BY click_time) AS gap_s
    FROM clicks
),

agg AS (
    SELECT
        ip,
        count(*)                                   AS n_clicks,
        count(DISTINCT app)                        AS n_apps,
        count(DISTINCT channel)                    AS n_channels,
        count(DISTINCT device)                     AS n_devices,
        count(DISTINCT os)                         AS n_os,
        count(DISTINCT hr)                         AS n_active_hours,

        -- Timing signature -------------------------------------------------
        avg(gap_s)                                 AS mean_gap,
        stddev_samp(gap_s)                         AS std_gap,
        median(gap_s)                              AS median_gap,
        -- Bot tell #1: a large share of sub-second gaps (machine-speed).
        sum(CASE WHEN gap_s < 1 THEN 1 ELSE 0 END)::DOUBLE
            / nullif(count(gap_s), 0)              AS burst_rate,
        -- Bot tell #2: clicks packed into a tiny window (click injection).
        sum(CASE WHEN gap_s < 10 THEN 1 ELSE 0 END)::DOUBLE
            / nullif(count(gap_s), 0)              AS rapid_rate,

        (max(epoch(click_time)) - min(epoch(click_time))) AS span_s,

        -- Hour-of-day histogram: 24 bins, consumed by the entropy /
        -- cross-population detectors in Python.
        count(*) FILTER (WHERE hr = 0)  AS h00, count(*) FILTER (WHERE hr = 1)  AS h01,
        count(*) FILTER (WHERE hr = 2)  AS h02, count(*) FILTER (WHERE hr = 3)  AS h03,
        count(*) FILTER (WHERE hr = 4)  AS h04, count(*) FILTER (WHERE hr = 5)  AS h05,
        count(*) FILTER (WHERE hr = 6)  AS h06, count(*) FILTER (WHERE hr = 7)  AS h07,
        count(*) FILTER (WHERE hr = 8)  AS h08, count(*) FILTER (WHERE hr = 9)  AS h09,
        count(*) FILTER (WHERE hr = 10) AS h10, count(*) FILTER (WHERE hr = 11) AS h11,
        count(*) FILTER (WHERE hr = 12) AS h12, count(*) FILTER (WHERE hr = 13) AS h13,
        count(*) FILTER (WHERE hr = 14) AS h14, count(*) FILTER (WHERE hr = 15) AS h15,
        count(*) FILTER (WHERE hr = 16) AS h16, count(*) FILTER (WHERE hr = 17) AS h17,
        count(*) FILTER (WHERE hr = 18) AS h18, count(*) FILTER (WHERE hr = 19) AS h19,
        count(*) FILTER (WHERE hr = 20) AS h20, count(*) FILTER (WHERE hr = 21) AS h21,
        count(*) FILTER (WHERE hr = 22) AS h22, count(*) FILTER (WHERE hr = 23) AS h23
    FROM ordered
    GROUP BY ip
)

SELECT
    *,
    n_clicks::DOUBLE / nullif(n_apps, 0)          AS clicks_per_app,
    n_clicks::DOUBLE / nullif(n_channels, 0)      AS clicks_per_channel,
    n_clicks::DOUBLE / nullif(span_s / 3600.0, 0) AS clicks_per_hour,
    -- Coefficient of variation of the inter-arrival gap.
    -- Human browsing is bursty and irregular  -> CV around 1 or above.
    -- A scripted clicker firing every ~40s    -> CV near 0.
    -- LOW values here are the anomaly, which is why the detector inverts it.
    std_gap / nullif(mean_gap, 0)                 AS cv_gap
FROM agg
WHERE n_clicks >= {min_clicks}

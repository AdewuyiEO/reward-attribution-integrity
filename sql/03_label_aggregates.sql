-- =============================================================================
-- Label aggregates -- EVALUATION ONLY. Never joined into the feature matrix.
-- =============================================================================
-- Kept in a separate file and a separate table on purpose. The physical
-- separation is the safeguard: to leak a label into training you would have to
-- deliberately join this in, rather than do it by accident.
-- =============================================================================
SELECT
    ip,
    count(*)           AS n_clicks,
    sum(is_attributed) AS n_conversions,
    avg(is_attributed) AS conv_rate
FROM clicks
GROUP BY ip

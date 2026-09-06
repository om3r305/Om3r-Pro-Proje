-- Brian 2026 ALPHA read-path optimization.
-- Production evidence on 2026-09-06:
--   MICRO_1_5M: ~150k rows
--   FAST_5_30M: ~52k rows
--   DAILY: 8 rows
-- The causal DAILY ALPHA query currently walks tens of thousands of unrelated
-- asset/time index entries to return only DAILY evidence. Keep write overhead
-- minimal by indexing only the tiny DAILY + available + non-GDELT subset.
-- SHADOW ONLY. No Phase 3.7, scoring, freshness, or execution semantics change.

CREATE INDEX IF NOT EXISTS brian_sensor_obs_alpha_daily_asset_time_idx
ON public.brian_sensor_observations (asset_id, observed_at DESC)
WHERE horizon = 'DAILY'
  AND available = true
  AND independent_group <> 'news_gdelt';

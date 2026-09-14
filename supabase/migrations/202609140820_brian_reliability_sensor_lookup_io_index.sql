-- Preserve the full sensor-learning window while avoiding repeated heap-wide scans.
-- This is a derived lookup acceleration only; no evidence, cadence, or DIP behavior changes.
create index if not exists brian_sensor_obs_reliability_lookup_v2_idx
on public.brian_sensor_observations (observation_id)
include (independent_group, sensor_family, horizon, direction, observed_at)
where available = true and direction <> 0 and independent_group <> 'news_gdelt';

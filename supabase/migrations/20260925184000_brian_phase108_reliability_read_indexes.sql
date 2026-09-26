-- Phase108 readiness reads are point-in-time and read-only.
-- These indexes accelerate the exact horizon/window selectors used by the
-- strict edge reader without changing evidence, thresholds, or execution logic.

create index if not exists brian_sensor_reliability_shadow_phase108_window_idx
on public.brian_sensor_reliability_shadow_snapshots
  (outcome_horizon_seconds, window_end desc, generated_at desc);

create index if not exists brian_sensor_reliability_shadow_phase108_group_window_idx
on public.brian_sensor_reliability_shadow_snapshots
  (outcome_horizon_seconds, window_end, generated_at, independent_group);

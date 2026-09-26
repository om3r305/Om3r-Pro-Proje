-- Accelerate the prospective reliability feature-freeze PIT lookups.
-- No evidence, thresholds, weights, or execution behavior changes.
create index if not exists brian_sensor_reliability_shadow_pit_window_generated_idx
on public.brian_sensor_reliability_shadow_snapshots
  (window_end desc, generated_at desc);

create index if not exists brian_sensor_reliability_shadow_feature_lookup_idx
on public.brian_sensor_reliability_shadow_snapshots
  (window_end, independent_group, sensor_family, sensor_horizon, generated_at desc);

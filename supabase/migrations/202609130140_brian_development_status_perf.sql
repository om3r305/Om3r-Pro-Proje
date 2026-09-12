-- Speed up Brian Development/Anatomy evidence counts without changing semantics.
-- The dashboard asks for exact counts of resolved prospective calibration rows and hits.
-- A partial realized_hit index keeps those reads index-friendly as the evidence table grows.
create index if not exists brian_sensor_calibration_realized_hit_idx
  on public.brian_sensor_reliability_prospective_calibration (realized_hit)
  where realized_hit is not null;

-- Global latest-source assessment reads order by assessed_at across all sources.
create index if not exists brian_world_source_assessments_assessed_at_idx
  on public.brian_world_source_assessments (assessed_at desc);

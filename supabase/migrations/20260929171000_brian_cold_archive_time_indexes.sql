-- Target: brian-market-intelligence. Cursor indexes for the cold archive scan
-- (order by observed_at, pk) on world tables that only had (key, observed_at).
-- Built CONCURRENTLY in production on 2026-09-29; IF NOT EXISTS keeps replays idempotent.
create index if not exists brian_world_asset_impact_archive_idx on public.brian_world_asset_impact_candidates(observed_at, impact_id);
create index if not exists brian_world_entity_obs_archive_idx on public.brian_world_entity_observations(observed_at, observation_id);
create index if not exists brian_world_narrative_archive_idx on public.brian_world_narrative_snapshots(observed_at, snapshot_id);

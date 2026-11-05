-- REALTIME project (dliediwlldojkfjzlznm). brian-realtime-multiasset-export reads the last 15 minutes of
-- brian_crowd_behavior_frames ordered by observed_at; the only index led with asset_id, so every call
-- scanned the whole index (1.8 s cold on 388k rows) and hit the 8 s PostgREST statement timeout under
-- load, failing the multiasset engine from ~16:00 UTC on 2026-11-05. Applied live with
-- CREATE INDEX CONCURRENTLY on 2026-11-05 22:47 UTC (query now 0.2 ms); this file records it.
create index if not exists brian_crowd_behavior_frames_observed_at_idx
  on public.brian_crowd_behavior_frames (observed_at desc);

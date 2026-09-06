-- Brian 2026 ALPHA frozen Phase 3.7 evidence read optimization.
-- Production query shape filters one immutable experiment plus NATIVE/PROFIT,
-- then orders globally by observed_at DESC and keeps 20 rows. Existing indexes
-- put policy_kind before observed_at, so PostgreSQL may choose seq-scan + sort.
-- This tiny partial index contains only the frozen experiment's NATIVE/PROFIT
-- ticks and directly satisfies the required recency ordering.
-- SHADOW ONLY. Phase 3.7 behavior remains frozen and untouched.

CREATE INDEX IF NOT EXISTS brian_live_shadow_phase37_recent_idx
ON public.brian_live_shadow_ticks (observed_at DESC)
WHERE experiment_id = 'phase37-prospective-live-20260903'
  AND policy_kind IN ('NATIVE', 'PROFIT');

-- DIP V8.3 resume fix.
-- Preserve append-only PAUSE/START history while allowing a paused session
-- to resume under the same session_id and retain its runtime/account state.
-- Session-control functions are serialized by brian-aggressive-dip-control,
-- so uniqueness is enforced by the control lock + latest-event state machine,
-- not by a one-START-per-session partial index.

drop index if exists public.brian_dip_session_start_unique;

create index if not exists brian_dip_session_start_lookup_idx
  on public.brian_dip_session_events(session_id, requested_at, event_id)
  where event_kind='START';

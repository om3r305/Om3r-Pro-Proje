-- Close direct PostgREST exposure on internal scheduler/capability state.
-- Existing SECURITY DEFINER control/status functions and service-role workers retain
-- their owner/service-role access; no public RLS policy is intentionally added.
alter table public.brian_scheduler_state enable row level security;
alter table public.brian_external_capability_heartbeats enable row level security;

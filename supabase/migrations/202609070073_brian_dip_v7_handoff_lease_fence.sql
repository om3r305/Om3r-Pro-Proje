-- Brian DIP V7 handoff lease fence.
-- SHADOW ONLY. V7 browser sessions explicitly declare server_authoritative=true in START config.
-- For those sessions the legacy V4/V5 heartbeat timer must not keep the browser engine lease fresh,
-- otherwise the 1-minute server worker would remain in WAIT_BROWSER_HANDOFF forever.

create or replace function public.brian_v7_neuter_browser_engine_heartbeat()
returns trigger
language plpgsql
security definer
set search_path = pg_catalog, public
as $function$
declare
  v_server_authoritative boolean := false;
begin
  select coalesce((e.config ->> 'server_authoritative')::boolean, false)
    into v_server_authoritative
  from public.brian_dip_session_events e
  where e.session_id = new.session_id
    and e.event_kind = 'START'
  order by e.requested_at asc, e.event_id asc
  limit 1;

  if coalesce(v_server_authoritative, false)
     and coalesce(new.claimed_by, '') = 'monster-coins-pro-dip-v4' then
    -- Worker handoff grace is 90 seconds. Keep browser lease telemetry append-safe but stale.
    new.heartbeat_at := least(
      coalesce(new.heartbeat_at, now() - interval '2 minutes'),
      now() - interval '2 minutes'
    );
  end if;

  return new;
end;
$function$;

revoke all on function public.brian_v7_neuter_browser_engine_heartbeat() from public, anon, authenticated;
grant execute on function public.brian_v7_neuter_browser_engine_heartbeat() to service_role;

drop trigger if exists brian_dip_v7_browser_heartbeat_fence on public.brian_dip_engine_leases;
create trigger brian_dip_v7_browser_heartbeat_fence
  before insert or update of heartbeat_at, claimed_by
  on public.brian_dip_engine_leases
  for each row execute function public.brian_v7_neuter_browser_engine_heartbeat();

comment on function public.brian_v7_neuter_browser_engine_heartbeat() is
  'For explicit DIP V7 server-authoritative sessions, prevents legacy browser engine_check timers from refreshing the browser lease and blocking server takeover. Legacy browser-owned V4/V5 sessions are unchanged.';

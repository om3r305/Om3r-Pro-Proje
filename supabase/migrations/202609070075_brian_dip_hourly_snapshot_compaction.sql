-- Brian DIP V8 bounded runtime persistence.
--
-- Problem: the server-authoritative DIP worker runs every minute and historically appended a
-- complete JSONB runtime snapshot on every pass. That is useful for live state, but unbounded as
-- historical evidence and can consume the Free-plan database quota quickly.
--
-- Contract after this migration:
--   * worker cadence stays 1 minute (decision quality / live dashboard are NOT slowed down)
--   * V8 keeps exactly one mutable snapshot row per session per UTC hour
--   * every minute refreshes that hour row with the newest runtime state
--   * once the UTC hour rolls, the prior hour row is no longer touched and becomes immutable history
--   * trade/thesis lifecycle evidence continues to live in brian_dip_events / brian_dip_theses
--   * V7/legacy snapshots are not rewritten by this trigger
--   * SHADOW ONLY; Phase 3.7 and the main cashbox are untouched

create or replace function public.brian_dip_compact_v8_snapshot_hourly()
returns trigger
language plpgsql
security definer
set search_path = pg_catalog, public
as $function$
declare
  v_hour_start timestamptz;
  v_existing_id text;
begin
  -- Fail open to the existing append-only behavior for anything that is not the V8 worker.
  if coalesce(new.state #>> '{serverRuntime,worker_version}', '') <> 'brian-dip-chart-reader-v8' then
    return new;
  end if;

  v_hour_start := (
    date_trunc('hour', new.observed_at at time zone 'UTC') at time zone 'UTC'
  );

  select s.snapshot_id
    into v_existing_id
    from public.brian_dip_snapshots s
   where s.session_id = new.session_id
     and s.observed_at >= v_hour_start
     and s.observed_at < v_hour_start + interval '1 hour'
   order by s.observed_at desc, s.snapshot_id desc
   limit 1
   for update;

  if v_existing_id is null then
    -- First observation of the UTC hour: keep it as the hour's bounded runtime row.
    return new;
  end if;

  -- Refresh the hour row instead of inserting another large JSONB snapshot. This function is
  -- SECURITY DEFINER owned by postgres, so the existing append-only guard permits this explicit
  -- maintenance mutation while ordinary service_role/browser mutations remain forbidden.
  update public.brian_dip_snapshots
     set observed_at     = new.observed_at,
         cash            = new.cash,
         equity          = new.equity,
         realized_pnl    = new.realized_pnl,
         unrealized_pnl  = new.unrealized_pnl,
         trade_count     = new.trade_count,
         win_count       = new.win_count,
         loss_count      = new.loss_count,
         state           = new.state,
         evidence_class  = new.evidence_class,
         shadow_only     = true,
         live_execution  = false
   where snapshot_id = v_existing_id;

  -- Suppress the redundant minute-row insert. The worker sees no insert error, and on its next
  -- pass latestRuntime() reads the refreshed hour row as the latest authoritative state.
  return null;
end;
$function$;

revoke all on function public.brian_dip_compact_v8_snapshot_hourly() from public, anon, authenticated, service_role;

drop trigger if exists brian_dip_v8_snapshot_hourly_compact on public.brian_dip_snapshots;
create trigger brian_dip_v8_snapshot_hourly_compact
  before insert on public.brian_dip_snapshots
  for each row execute function public.brian_dip_compact_v8_snapshot_hourly();

-- The current-hour row is updated repeatedly, so vacuum aggressively enough that dead JSONB/TOAST
-- versions are recycled instead of becoming a new form of storage leak.
alter table public.brian_dip_snapshots set (
  autovacuum_vacuum_scale_factor = 0.02,
  autovacuum_vacuum_threshold = 20,
  autovacuum_analyze_scale_factor = 0.05,
  toast.autovacuum_vacuum_scale_factor = 0.02,
  toast.autovacuum_vacuum_threshold = 20
);

comment on function public.brian_dip_compact_v8_snapshot_hourly() is
  'Compacts Brian DIP V8 minute runtime writes into one live/final snapshot per session per UTC hour. SHADOW ONLY.';

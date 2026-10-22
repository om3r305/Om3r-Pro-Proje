-- Unified treasury (operator request 2026-10-22: one decision chain, cost gate, anti-churn,
-- evidence-based sizing). This migration adds the common expected-edge EVIDENCE: for every
-- directional multiasset signal, the realised 60-minute forward return of its own reference price.
-- Outcomes are written once, only after the horizon has passed (point in time: a signal is never
-- scored with information it did not have). The treasury reads small segment aggregates from it.
-- IO budget: hourly, incremental, index-backed; no repeated scans of the 200 MB decision table.

create index if not exists brian_multiasset_alpha_observed_idx
  on public.brian_multiasset_alpha_decisions (observed_at);

create table if not exists public.brian_treasury_signal_outcomes (
  decision_id text primary key,
  lane text not null check (lane in ('MULTIASSET_EVENT_REACTION')),
  asset_id text not null,
  asset_class text not null,
  direction smallint not null check (direction in (-1, 1)),
  observed_at timestamptz not null,
  evidence_score double precision,
  cost_bps double precision not null check (cost_bps >= 0),
  entry_price double precision not null check (entry_price > 0),
  exit_price double precision not null check (exit_price > 0),
  exit_observed_at timestamptz not null,
  horizon_minutes integer not null check (horizon_minutes > 0),
  gross_bps double precision not null,
  created_at timestamptz not null default now()
);
create index if not exists brian_treasury_signal_outcomes_seg_idx
  on public.brian_treasury_signal_outcomes (lane, asset_class, direction, observed_at desc);

drop trigger if exists brian_treasury_signal_outcomes_append_only on public.brian_treasury_signal_outcomes;
create trigger brian_treasury_signal_outcomes_append_only before update or delete on public.brian_treasury_signal_outcomes
  for each row execute function public.brian_reject_mutation();

alter table public.brian_treasury_signal_outcomes enable row level security;
revoke all on public.brian_treasury_signal_outcomes from public, anon, authenticated;
grant select on public.brian_treasury_signal_outcomes to service_role;

-- Record matured outcomes for signals observed in [now - p_lookback, now - horizon - 15 min].
-- Exit price = first decision for the same asset observed 55..75 minutes after the signal.
create or replace function public.brian_record_treasury_signal_outcomes_v1(p_lookback interval default interval '3 hours')
returns integer
language plpgsql
security definer
set search_path = pg_catalog, public
set statement_timeout = '90s'
as $fn$
declare n integer;
begin
  if p_lookback > interval '8 days' then raise exception 'lookback too long'; end if;
  insert into public.brian_treasury_signal_outcomes
    (decision_id, lane, asset_id, asset_class, direction, observed_at, evidence_score, cost_bps,
     entry_price, exit_price, exit_observed_at, horizon_minutes, gross_bps)
  select d.decision_id, 'MULTIASSET_EVENT_REACTION', d.asset_id, coalesce(d.asset_class, 'unknown'), d.direction::smallint,
         d.observed_at, d.evidence_score, d.estimated_round_trip_cost_bps,
         d.observed_reference_price, x.observed_reference_price, x.observed_at, 60,
         d.direction * (x.observed_reference_price / d.observed_reference_price - 1) * 10000
  from public.brian_multiasset_alpha_decisions d
  cross join lateral (
    select y.observed_reference_price, y.observed_at
    from public.brian_multiasset_alpha_decisions y
    where y.asset_id = d.asset_id
      and y.observed_at between d.observed_at + interval '55 minutes' and d.observed_at + interval '75 minutes'
      and y.observed_reference_price > 0
    order by y.observed_at
    limit 1) x
  where d.observed_at >= now() - p_lookback
    and d.observed_at < now() - interval '75 minutes'
    and d.action in ('OPEN_LONG', 'OPEN_SHORT')
    and d.direction in (-1, 1)
    and d.session_state = 'REGULAR'
    and d.observed_reference_price > 0
    and d.estimated_round_trip_cost_bps >= 0
  on conflict (decision_id) do nothing;
  get diagnostics n = row_count;
  return n;
end $fn$;
revoke all on function public.brian_record_treasury_signal_outcomes_v1(interval) from public, anon, authenticated;

-- Segment evidence for the treasury (small aggregate; read once per treasury cycle).
create or replace function public.brian_treasury_segment_edges_v1(p_days integer default 7)
returns table (lane text, asset_class text, direction smallint, n bigint, gross_bps double precision,
               se_bps double precision, hit_rate double precision, cost_bps double precision, last_observed_at timestamptz)
language sql
stable
security definer
set search_path = pg_catalog, public
as $fn$
  select o.lane, o.asset_class, o.direction, count(*), avg(o.gross_bps),
         coalesce(stddev_samp(o.gross_bps) / sqrt(count(*)), 1e6),
         avg((o.gross_bps > 0)::int::double precision), avg(o.cost_bps), max(o.observed_at)
  from public.brian_treasury_signal_outcomes o
  where o.observed_at >= now() - make_interval(days => greatest(1, least(coalesce(p_days, 7), 30)))
  group by o.lane, o.asset_class, o.direction
$fn$;
revoke all on function public.brian_treasury_segment_edges_v1(integer) from public, anon, authenticated;
grant execute on function public.brian_treasury_segment_edges_v1(integer) to service_role;

insert into public.brian_system_job_registry (job_name, component, operator_managed)
values ('brian-treasury-signal-outcomes-hourly', 'TREASURY', true)
on conflict do nothing;

select cron.schedule('brian-treasury-signal-outcomes-hourly', '23 * * * *',
  $cron$select public.brian_record_treasury_signal_outcomes_v1(interval '3 hours')$cron$);

-- Kill switch for the unified lane (read by brian-evolution-treasury every cycle). Starts OFF; the
-- operator-approved switch-on is a separate, logged step.
insert into public.brian_evolution_runtime_config (config_key, config_value)
values ('brian_treasury_unified_enabled', 'false')
on conflict (config_key) do nothing;

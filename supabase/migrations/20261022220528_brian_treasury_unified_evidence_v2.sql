-- Unified treasury evidence v2 (operator 2026-10-23: "bulut hariç bütün kopuklukları gider").
--  * Crypto: canonical ALPHA directional decisions join the evidence ledger as lane CANONICAL_ALPHA,
--    measured on their own 3600 s outcomes (brian_alpha_decision_outcomes.direction_adjusted_return).
--  * News: every multiasset signal is news-triggered; its evidence is now split by the best trust
--    class of the linked news (OFFICIAL / PROFESSIONAL / UNVERIFIED), so the treasury measures
--    which kind of news actually pays after costs. Crypto signals carry news_class NONE.
--  * IO: segment aggregates are rebuilt once per hour into a tiny cache table; the treasury reads
--    only the cache (no per-cycle aggregation over the ledger).
-- Dormant by contract: the hourly job is re-pointed through an explicit brian_private function.

alter table public.brian_treasury_signal_outcomes drop constraint if exists brian_treasury_signal_outcomes_lane_check;
alter table public.brian_treasury_signal_outcomes
  add constraint brian_treasury_signal_outcomes_lane_check check (lane in ('MULTIASSET_EVENT_REACTION', 'CANONICAL_ALPHA'));

create or replace function public.brian_news_class_v1(p_event_ids text[])
returns text
language sql
stable
security definer
set search_path = pg_catalog, public
as $fn$
  select case
    when coalesce(cardinality(p_event_ids), 0) = 0 then 'NONE'
    when bool_or(e.trust_class = 'OFFICIAL_PRIMARY') then 'OFFICIAL'
    when bool_or(e.trust_class = 'INDEPENDENT_PROFESSIONAL') then 'PROFESSIONAL'
    when count(e.event_id) > 0 then 'UNVERIFIED'
    else 'UNKNOWN'
  end
  from public.brian_intel_events e
  where e.event_id = any(p_event_ids)
$fn$;
revoke all on function public.brian_news_class_v1(text[]) from public, anon, authenticated;
grant execute on function public.brian_news_class_v1(text[]) to service_role;

-- Crypto outcomes resolved in [now - p_lookback, now]: one row per directional decision, 3600 s horizon.
create or replace function public.brian_record_treasury_crypto_outcomes_v1(p_lookback interval default interval '3 hours')
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
  select d.decision_id, 'CANONICAL_ALPHA', d.asset_id, 'crypto', d.direction::smallint, d.observed_at, d.evidence_score,
         d.estimated_round_trip_cost_bps, o.reference_price, o.resolved_price, o.resolved_at, 60,
         o.direction_adjusted_return * 10000
  from public.brian_alpha_decision_outcomes o
  join public.brian_alpha_decisions d on d.decision_id = o.decision_id
  where o.resolved_at >= now() - p_lookback
    and o.horizon_seconds = 3600
    and d.action in ('OPEN_LONG', 'OPEN_SHORT')
    and d.direction in (-1, 1)
    and d.veto_reason is null
    and d.estimated_round_trip_cost_bps >= 0
    and o.reference_price > 0 and o.resolved_price > 0
    and o.direction_adjusted_return is not null
  on conflict (decision_id) do nothing;
  get diagnostics n = row_count;
  return n;
end $fn$;
revoke all on function public.brian_record_treasury_crypto_outcomes_v1(interval) from public, anon, authenticated;

create table if not exists public.brian_treasury_segment_edge_cache (
  lane text not null,
  asset_class text not null,
  direction smallint not null check (direction in (-1, 1)),
  news_class text not null,
  n bigint not null,
  gross_bps double precision not null,
  se_bps double precision not null,
  hit_rate double precision not null,
  cost_bps double precision not null,
  last_observed_at timestamptz,
  computed_at timestamptz not null default now(),
  primary key (lane, asset_class, direction, news_class)
);
alter table public.brian_treasury_segment_edge_cache enable row level security;
revoke all on public.brian_treasury_segment_edge_cache from public, anon, authenticated;
grant select on public.brian_treasury_segment_edge_cache to service_role;

-- Rebuild the 7-day segment cache (small: tens of rows). News class for multiasset outcomes comes
-- from the linked events of the original decision.
create or replace function public.brian_refresh_treasury_segment_cache_v1(p_days integer default 7)
returns integer
language plpgsql
security definer
set search_path = pg_catalog, public
set statement_timeout = '90s'
as $fn$
declare n integer;
begin
  delete from public.brian_treasury_segment_edge_cache;
  insert into public.brian_treasury_segment_edge_cache
    (lane, asset_class, direction, news_class, n, gross_bps, se_bps, hit_rate, cost_bps, last_observed_at, computed_at)
  with seg_rows as (
    select o.lane, o.asset_class, o.direction,
           case when o.lane = 'MULTIASSET_EVENT_REACTION'
                then coalesce(public.brian_news_class_v1(m.linked_event_ids), 'UNKNOWN') else 'NONE' end as news_class,
           o.gross_bps, o.cost_bps, o.observed_at
    from public.brian_treasury_signal_outcomes o
    left join public.brian_multiasset_alpha_decisions m on m.decision_id = o.decision_id and o.lane = 'MULTIASSET_EVENT_REACTION'
    where o.observed_at >= now() - make_interval(days => greatest(1, least(coalesce(p_days, 7), 30))))
  select lane, asset_class, direction, news_class, count(*), avg(gross_bps),
         coalesce(stddev_samp(gross_bps) / sqrt(count(*)), 1e6), avg((gross_bps > 0)::int::double precision),
         avg(cost_bps), max(observed_at), now()
  from seg_rows group by 1, 2, 3, 4;
  get diagnostics n = row_count;
  return n;
end $fn$;
revoke all on function public.brian_refresh_treasury_segment_cache_v1(integer) from public, anon, authenticated;

create or replace function public.brian_treasury_segment_edges_v2()
returns table (lane text, asset_class text, direction smallint, news_class text, n bigint, gross_bps double precision,
               se_bps double precision, hit_rate double precision, cost_bps double precision, computed_at timestamptz)
language sql
stable
security definer
set search_path = pg_catalog, public
as $fn$
  select lane, asset_class, direction, news_class, n, gross_bps, se_bps, hit_rate, cost_bps, computed_at
  from public.brian_treasury_segment_edge_cache
$fn$;
revoke all on function public.brian_treasury_segment_edges_v2() from public, anon, authenticated;
grant execute on function public.brian_treasury_segment_edges_v2() to service_role;

-- One hourly pipeline: multiasset outcomes, crypto outcomes, cache rebuild.
create or replace function public.brian_treasury_evidence_hourly_v1()
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $fn$
declare a integer; c integer; s integer;
begin
  a := public.brian_record_treasury_signal_outcomes_v1(interval '3 hours');
  c := public.brian_record_treasury_crypto_outcomes_v1(interval '3 hours');
  s := public.brian_refresh_treasury_segment_cache_v1(7);
  return jsonb_build_object('multiasset', a, 'crypto', c, 'segments', s, 'at', now());
end $fn$;
revoke all on function public.brian_treasury_evidence_hourly_v1() from public, anon, authenticated;

create or replace function brian_private.schedule_treasury_evidence_hourly_v1()
returns bigint
language plpgsql
security definer
set search_path = pg_catalog, public
as $fn$
declare j bigint;
begin
  perform cron.unschedule(jobid) from cron.job where jobname = 'brian-treasury-signal-outcomes-hourly';
  j := cron.schedule('brian-treasury-signal-outcomes-hourly', '23 * * * *',
    $cron$select public.brian_treasury_evidence_hourly_v1()$cron$);
  return j;
end $fn$;
revoke all on function brian_private.schedule_treasury_evidence_hourly_v1() from public, anon, authenticated, service_role;

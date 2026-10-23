-- Futures-roll guard for the treasury evidence ledger (found 2026-10-23 in the day-end audit).
-- The multiasset price feed switched NATGAS and BRENT to the next contract at 07:53 UTC: one
-- 10-minute step of +13.7% / -4.3% that then stayed flat, while the real front month did not move
-- (Brent traded ~114-115 all day; NATGAS ~2.66). Four NATGAS shorts opened just before the switch were
-- recorded as ~-1,500 bps each and distorted the commodity/short segment.
-- Rule: a commodity outcome whose 60-min window contains a single step > 4% between consecutive
-- observations at most 20 min apart is a contract-roll artefact, not a market outcome. In 8 days of
-- data only these two steps exceed 3% for commodities. Rows are flagged, never deleted, and the
-- segment cache ignores flagged rows. No new job: the existing hourly evidence job runs the marker.

alter table public.brian_treasury_signal_outcomes add column if not exists excluded_reason text;

create or replace function public.brian_mark_treasury_roll_outcomes_v1(p_lookback interval default interval '8 days')
returns integer
language plpgsql
security definer
set search_path = pg_catalog, public
set statement_timeout = '60s'
as $fn$
declare n integer;
begin
  if p_lookback > interval '30 days' then raise exception 'lookback too long'; end if;
  with steps as (
    select asset_id, observed_at t, observed_reference_price p,
           lag(observed_reference_price) over (partition by asset_id order by observed_at) pp,
           lag(observed_at) over (partition by asset_id order by observed_at) pt
    from public.brian_multiasset_alpha_decisions
    where asset_class = 'commodity' and observed_at >= now() - p_lookback - interval '2 hours'
      and observed_reference_price > 0),
  jumps as (
    select asset_id, t from steps
    where pp > 0 and t - pt <= interval '20 minutes' and abs(ln(p / pp)) > 0.04)
  update public.brian_treasury_signal_outcomes o
     set excluded_reason = 'FUTURES_ROLL_DISCONTINUITY'
   where o.excluded_reason is null
     and o.lane = 'MULTIASSET_EVENT_REACTION'
     and o.asset_class = 'commodity'
     and o.observed_at >= now() - p_lookback
     and exists (select 1 from jumps j where j.asset_id = o.asset_id and j.t > o.observed_at and j.t <= o.exit_observed_at);
  get diagnostics n = row_count;
  return n;
end $fn$;
revoke all on function public.brian_mark_treasury_roll_outcomes_v1(interval) from public, anon, authenticated, service_role;

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
    where o.observed_at >= now() - make_interval(days => greatest(1, least(coalesce(p_days, 7), 30)))
      and o.excluded_reason is null)
  select lane, asset_class, direction, news_class, count(*), avg(gross_bps),
         coalesce(stddev_samp(gross_bps) / sqrt(count(*)), 1e6), avg((gross_bps > 0)::int::double precision),
         avg(cost_bps), max(observed_at), now()
  from seg_rows group by 1, 2, 3, 4;
  get diagnostics n = row_count;
  return n;
end $fn$;
revoke all on function public.brian_refresh_treasury_segment_cache_v1(integer) from public, anon, authenticated;

create or replace function public.brian_treasury_evidence_hourly_v1()
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $fn$
declare a integer; c integer; r integer; s integer;
begin
  a := public.brian_record_treasury_signal_outcomes_v1(interval '3 hours');
  c := public.brian_record_treasury_crypto_outcomes_v1(interval '3 hours');
  r := public.brian_mark_treasury_roll_outcomes_v1(interval '1 day');
  s := public.brian_refresh_treasury_segment_cache_v1(7);
  return jsonb_build_object('multiasset', a, 'crypto', c, 'roll_excluded', r, 'segments', s, 'at', now());
end $fn$;
revoke all on function public.brian_treasury_evidence_hourly_v1() from public, anon, authenticated;

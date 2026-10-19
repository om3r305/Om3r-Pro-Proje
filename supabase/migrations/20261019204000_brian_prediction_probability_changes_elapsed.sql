-- Brian Prediction Intelligence -- Phase 2a.1: report real elapsed time and relative change.
--
-- Additive revision of brian_prediction_probability_changes_v1 (same signature, extra fields):
--   * baseline_*_at / elapsed_*_seconds: when the comparison value was actually received and the
--     real time between it and the latest observation (collection gaps make this differ from the
--     nominal horizon).
--   * relative_change_*: (p_now - p_then) / p_then, kept separate from the point change so that
--     +10 percentage points and +25% relative are never confused. NULL when p_then = 0.
-- Selection is unchanged: only snapshots with received_at <= as_of are used.

create or replace function public.brian_prediction_probability_changes_v1(
  p_as_of timestamptz default now(),
  p_limit integer default 60,
  p_stale_after interval default interval '45 minutes'
) returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  with params as (
    select p_as_of as as_of, greatest(1, least(coalesce(p_limit, 60), 300)) as lim,
           coalesce(p_stale_after, interval '45 minutes') as stale_after
  ),
  known as (
    select s.market_key, s.received_at, s.source_updated_at, s.spread, s.liquidity, s.volume_24h,
           s.closed,
           coalesce(
             (select (o->>'implied_probability')::numeric from jsonb_array_elements(s.outcomes) o
               where o->>'label' = 'Yes' limit 1),
             (s.outcomes->0->>'implied_probability')::numeric
           ) as p
    from public.brian_prediction_market_snapshots s, params
    where s.received_at <= params.as_of
  ),
  latest as (
    select distinct on (market_key) * from known order by market_key, received_at desc
  ),
  horizons as (
    -- For each horizon: the value Brian knew at (as_of - horizon) and WHEN it was received, so the
    -- real elapsed time between the two observations is reported, not assumed to be the horizon.
    select l.market_key,
      b1.p as p_1h, b1.received_at as baseline_1h_at,
      b6.p as p_6h, b6.received_at as baseline_6h_at,
      b24.p as p_24h, b24.received_at as baseline_24h_at,
      (select count(*) from known k where k.market_key = l.market_key) as snapshot_count
    from latest l
    cross join params
    left join lateral (select k.p, k.received_at from known k where k.market_key = l.market_key
      and k.received_at <= params.as_of - interval '1 hour' order by k.received_at desc limit 1) b1 on true
    left join lateral (select k.p, k.received_at from known k where k.market_key = l.market_key
      and k.received_at <= params.as_of - interval '6 hours' order by k.received_at desc limit 1) b6 on true
    left join lateral (select k.p, k.received_at from known k where k.market_key = l.market_key
      and k.received_at <= params.as_of - interval '24 hours' order by k.received_at desc limit 1) b24 on true
  ),
  mrows as (
    select l.market_key, m.question, m.event_title, m.end_date,
           l.p as probability, l.received_at as latest_received_at, l.source_updated_at,
           extract(epoch from (params.as_of - l.received_at))::integer as age_seconds,
           l.spread, l.liquidity, l.volume_24h, l.closed,
           h.p_1h, h.p_6h, h.p_24h, h.snapshot_count,
           h.baseline_1h_at, h.baseline_6h_at, h.baseline_24h_at,
           extract(epoch from (l.received_at - h.baseline_1h_at))::integer as elapsed_1h_seconds,
           extract(epoch from (l.received_at - h.baseline_6h_at))::integer as elapsed_6h_seconds,
           extract(epoch from (l.received_at - h.baseline_24h_at))::integer as elapsed_24h_seconds,
           -- Relative change is a different quantity from the point change (0.40 -> 0.50 is +0.10
           -- probability points but +25% relative); undefined (NULL) when the baseline is 0.
           case when l.p is null or h.p_1h is null or h.p_1h = 0 then null else round((l.p - h.p_1h) / h.p_1h, 6) end as relative_change_1h,
           case when l.p is null or h.p_6h is null or h.p_6h = 0 then null else round((l.p - h.p_6h) / h.p_6h, 6) end as relative_change_6h,
           case when l.p is null or h.p_24h is null or h.p_24h = 0 then null else round((l.p - h.p_24h) / h.p_24h, 6) end as relative_change_24h,
           case when l.p is null or h.p_1h is null then null else round(l.p - h.p_1h, 6) end as change_1h,
           case when l.p is null or h.p_6h is null then null else round(l.p - h.p_6h, 6) end as change_6h,
           case when l.p is null or h.p_24h is null then null else round(l.p - h.p_24h, 6) end as change_24h,
           case
             when l.p is null then 'NO_PROBABILITY'
             when params.as_of - l.received_at > params.stale_after then 'STALE'
             when h.p_1h is null then 'INSUFFICIENT_HISTORY'
             else 'FRESH'
           end as data_status
    from latest l
    join public.brian_prediction_markets m on m.market_key = l.market_key
    join horizons h on h.market_key = l.market_key
    cross join params
  )
  select jsonb_build_object(
    'as_of', (select as_of from params),
    'method', 'point_in_time_received_at_only',
    'evidence_class', 'OBSERVED_VERIFIED',
    'trade_signal', false,
    'change_unit', 'probability_points: change_* = p_now - p_then in [0,1] (0.10 = 10 percentage points); relative_change_* = change / p_then',
    'markets', coalesce((
      select jsonb_agg(to_jsonb(r) order by abs(coalesce(r.change_1h, 0)) desc, r.volume_24h desc nulls last, r.market_key)
      from (select * from mrows order by abs(coalesce(change_1h, 0)) desc, volume_24h desc nulls last, market_key
            limit (select lim from params)) r
    ), '[]'::jsonb),
    'counts', jsonb_build_object(
      'markets', (select count(*) from mrows),
      'fresh', (select count(*) from mrows where data_status = 'FRESH'),
      'insufficient_history', (select count(*) from mrows where data_status = 'INSUFFICIENT_HISTORY'),
      'stale', (select count(*) from mrows where data_status = 'STALE'),
      'no_probability', (select count(*) from mrows where data_status = 'NO_PROBABILITY')
    )
  );
$$;

revoke all on function public.brian_prediction_probability_changes_v1(timestamptz, integer, interval) from public, anon, authenticated;
grant execute on function public.brian_prediction_probability_changes_v1(timestamptz, integer, interval) to service_role;

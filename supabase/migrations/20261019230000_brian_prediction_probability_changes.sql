-- Brian Prediction Intelligence -- Phase 2a: point-in-time probability changes + data health.
--
-- Two read-only functions over the evidence collected since Phase 1d. Neither writes anything,
-- produces a trade signal, or changes a decision.
--
-- NO FUTURE INFORMATION. Every number is computed "as of" p_as_of using only snapshots whose
-- received_at <= p_as_of, i.e. what Brian had actually received by then. The source's own
-- update time is reported but never used to select history, so a late-arriving row can never
-- leak into an earlier as-of view. A change over a horizon is NULL (not 0) when no snapshot that
-- old exists yet: INSUFFICIENT_HISTORY is a state, not a zero move.
--
-- Probability = implied probability of the "Yes" outcome (first outcome if no "Yes" label).
-- A market whose latest snapshot is older than p_stale_after is STALE and its changes are still
-- shown but flagged; a NULL probability stays NULL.

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
    select l.market_key,
      (select k.p from known k where k.market_key = l.market_key and k.received_at <= params.as_of - interval '1 hour'
        order by k.received_at desc limit 1) as p_1h,
      (select k.p from known k where k.market_key = l.market_key and k.received_at <= params.as_of - interval '6 hours'
        order by k.received_at desc limit 1) as p_6h,
      (select k.p from known k where k.market_key = l.market_key and k.received_at <= params.as_of - interval '24 hours'
        order by k.received_at desc limit 1) as p_24h,
      (select count(*) from known k where k.market_key = l.market_key) as snapshot_count
    from latest l, params
  ),
  mrows as (
    select l.market_key, m.question, m.event_title, m.end_date,
           l.p as probability, l.received_at as latest_received_at, l.source_updated_at,
           extract(epoch from (params.as_of - l.received_at))::integer as age_seconds,
           l.spread, l.liquidity, l.volume_24h, l.closed,
           h.p_1h, h.p_6h, h.p_24h, h.snapshot_count,
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

-- Data health over a window ending at p_as_of: per-source status mix and last observation,
-- collection-cycle coverage against the 15-minute schedule, snapshot freshness, storage vs cap.
create or replace function public.brian_prediction_health_summary_v1(
  p_as_of timestamptz default now(),
  p_window interval default interval '24 hours'
) returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  with params as (
    select p_as_of as as_of, coalesce(p_window, interval '24 hours') as win
  ),
  h as (
    select hh.* from public.brian_prediction_source_health hh, params
    where hh.observed_at <= params.as_of and hh.observed_at > params.as_of - params.win
  ),
  per_source as (
    select source,
           count(*) as rows,
           count(*) filter (where status = 'OK') as ok,
           count(*) filter (where status <> 'OK') as not_ok,
           max(observed_at) as last_observed_at,
           (array_agg(status order by observed_at desc))[1] as last_status,
           round(avg(latency_ms))::integer as avg_latency_ms,
           max(latency_ms) as max_latency_ms
    from h group by source
  ),
  cycles as (
    -- One discovery request per tag per cycle; a cycle is identified by its minute.
    select count(distinct date_trunc('minute', observed_at)) as observed
    from h where source like 'polymarket-gamma-events:%'
  ),
  snaps as (
    select s.* from public.brian_prediction_market_snapshots s, params
    where s.received_at <= params.as_of and s.received_at > params.as_of - params.win
  )
  select jsonb_build_object(
    'as_of', (select as_of from params),
    'window_seconds', (select extract(epoch from win)::integer from params),
    'sources', coalesce((select jsonb_agg(to_jsonb(p) order by p.source) from per_source p), '[]'::jsonb),
    'cycles', jsonb_build_object(
      'observed', (select observed from cycles),
      -- Expected is bounded by when collection actually started, so a young module is not
      -- reported as missing cycles it could never have run.
      'expected', (
        select greatest(1, floor(extract(epoch from (params.as_of - greatest(
          params.as_of - params.win,
          coalesce((select min(observed_at) from public.brian_prediction_source_health
                    where source like 'polymarket-gamma-events:%'), params.as_of)
        ))) / 900)::integer + 1)
        from params
      )
    ),
    'snapshots', jsonb_build_object(
      'rows', (select count(*) from snaps),
      'markets', (select count(distinct market_key) from snaps),
      'median_source_freshness_seconds', (select percentile_cont(0.5) within group (order by source_freshness_seconds) from snaps),
      'null_probability_rows', (select count(*) from snaps where not exists (
        select 1 from jsonb_array_elements(snaps.outcomes) o where o->>'implied_probability' is not null))
    ),
    'storage', jsonb_build_object(
      'bytes', public.brian_prediction_storage_bytes(),
      'cap_bytes', (select max_storage_mb::bigint * 1024 * 1024 from public.brian_prediction_control),
      'cap_reached_rows', (select count(*) from h where metadata->>'storage_cap_reached' = 'true')
    ),
    'control', (select jsonb_build_object('module_enabled', module_enabled, 'wallet_tracking_enabled', wallet_tracking_enabled)
                from public.brian_prediction_control)
  );
$$;

revoke all on function public.brian_prediction_probability_changes_v1(timestamptz, integer, interval) from public, anon, authenticated;
revoke all on function public.brian_prediction_health_summary_v1(timestamptz, interval) from public, anon, authenticated;
grant execute on function public.brian_prediction_probability_changes_v1(timestamptz, integer, interval) to service_role;
grant execute on function public.brian_prediction_health_summary_v1(timestamptz, interval) to service_role;

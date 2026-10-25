-- Longer-horizon evidence for the unified treasury (operator 2026-10-24: "important news is not
-- abandoned after one hour"). Measure-only: the treasury keeps deciding on the 60-min ledger
-- (brian_treasury_signal_outcomes); nothing here changes its behaviour.
--  * horizons 240M and 1440M for crypto (canonical alpha) and multiasset signals, SESSION_CLOSE for
--    multiasset (last REGULAR observation of the same UTC day);
--  * per multiasset signal: news trust class, theme_sector_match (a linked theme is the asset's primary
--    theme), breadth (distinct assets of the same sector with a same-direction OPEN signal in the
--    30 minutes up to and including the signal; past-only, no look-ahead);
--  * futures-roll artefacts are flagged (same >4% single-step rule as 20261023200659), never deleted;
--  * one hourly pipeline (existing job 106): no new job.

create table if not exists public.brian_treasury_horizon_outcomes (
  decision_id text not null,
  horizon text not null check (horizon in ('240M', '1440M', 'SESSION_CLOSE')),
  lane text not null check (lane in ('MULTIASSET_EVENT_REACTION', 'CANONICAL_ALPHA')),
  asset_id text not null,
  asset_class text not null,
  sector text not null,
  direction smallint not null check (direction in (-1, 1)),
  observed_at timestamptz not null,
  cost_bps double precision not null,
  entry_price double precision not null,
  exit_price double precision not null,
  exit_observed_at timestamptz not null,
  gross_bps double precision not null,
  news_class text not null,
  linked_themes text[] not null default '{}',
  theme_sector_match boolean,
  breadth integer not null,
  excluded_reason text,
  created_at timestamptz not null default now(),
  primary key (decision_id, horizon)
);
create index if not exists brian_treasury_horizon_outcomes_observed_idx on public.brian_treasury_horizon_outcomes (observed_at);
alter table public.brian_treasury_horizon_outcomes enable row level security;
revoke all on public.brian_treasury_horizon_outcomes from public, anon, authenticated;
grant select on public.brian_treasury_horizon_outcomes to service_role;

-- Sector and primary themes of a tracked asset (crypto = one sector).
create or replace function public.brian_asset_sector_v1(p_asset_id text)
returns text language sql immutable set search_path = pg_catalog as $fn$
  select case
    when p_asset_id like 'crypto:%' then 'CRYPTO'
    when p_asset_id in ('commodity:BRENT', 'commodity:WTI', 'commodity:NATGAS', 'equity:CVX', 'equity:XOM', 'etf:USO', 'etf:XLE') then 'ENERGY'
    when p_asset_id in ('equity:AAPL', 'equity:AMZN', 'equity:GOOGL', 'equity:META', 'equity:MSFT', 'equity:NVDA', 'equity:TSLA',
                        'etf:SMH', 'etf:QQQ', 'index:NASDAQ100') then 'TECH'
    when p_asset_id in ('commodity:GOLD', 'commodity:SILVER', 'etf:GLD') then 'METALS'
    when p_asset_id in ('equity:JPM', 'etf:XLF', 'etf:IWM') then 'FINANCIALS'
    when p_asset_id in ('etf:TLT', 'fx:EURUSD', 'fx:GBPUSD', 'fx:USDCHF', 'fx:USDJPY') then 'RATES_FX'
    when p_asset_id in ('etf:SPY', 'index:SP500', 'index:DAX', 'etf:EWG') then 'BROAD_EQUITY'
    else 'OTHER' end
$fn$;

create or replace function public.brian_sector_primary_themes_v1(p_sector text)
returns text[] language sql immutable set search_path = pg_catalog as $fn$
  select case p_sector
    when 'ENERGY' then array['ENERGY']
    when 'TECH' then array['AI_TECH']
    when 'METALS' then array['MONETARY', 'GEOPOLITICAL']
    when 'FINANCIALS' then array['FINANCIAL', 'MONETARY']
    when 'RATES_FX' then array['MONETARY', 'FX']
    when 'BROAD_EQUITY' then array['MONETARY', 'FINANCIAL']
    else array[]::text[] end
$fn$;
revoke all on function public.brian_asset_sector_v1(text), public.brian_sector_primary_themes_v1(text) from public, anon, authenticated;

-- Record horizon outcomes for signals whose horizon matured within the last p_window (backfill: larger window).
create or replace function public.brian_record_treasury_horizon_outcomes_v1(p_window interval default interval '3 hours')
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
set statement_timeout = '120s'
as $fn$
declare ma integer; cr integer; sc integer;
begin
  if p_window > interval '9 days' then raise exception 'window too long'; end if;

  insert into public.brian_treasury_horizon_outcomes
    (decision_id, horizon, lane, asset_id, asset_class, sector, direction, observed_at, cost_bps, entry_price, exit_price,
     exit_observed_at, gross_bps, news_class, linked_themes, theme_sector_match, breadth)
  select d.decision_id, h.label, 'MULTIASSET_EVENT_REACTION', d.asset_id, coalesce(d.asset_class, 'unknown'),
         public.brian_asset_sector_v1(d.asset_id), d.direction::smallint, d.observed_at, d.estimated_round_trip_cost_bps,
         d.observed_reference_price, x.p, x.t, d.direction * (x.p / d.observed_reference_price - 1) * 10000,
         coalesce(public.brian_news_class_v1(d.linked_event_ids), 'UNKNOWN'), th.themes,
         (th.themes && public.brian_sector_primary_themes_v1(public.brian_asset_sector_v1(d.asset_id))) or ('DIRECT_ASSET' = any(th.themes)),
         b.n
  from public.brian_multiasset_alpha_decisions d
  cross join (values ('240M', 240), ('1440M', 1440)) h(label, mins)
  cross join lateral (
    select array(select jsonb_array_elements_text(coalesce(d.metadata->'linked_themes', '[]'::jsonb))) themes) th
  cross join lateral (
    select y.observed_reference_price p, y.observed_at t
    from public.brian_multiasset_alpha_decisions y
    where y.asset_id = d.asset_id
      and y.observed_at between d.observed_at + make_interval(mins => h.mins - 20) and d.observed_at + make_interval(mins => h.mins + 20)
      and y.observed_reference_price > 0
    order by abs(extract(epoch from (y.observed_at - d.observed_at)) - h.mins * 60)
    limit 1) x
  cross join lateral (
    select count(distinct z.asset_id)::int n
    from public.brian_multiasset_alpha_decisions z
    where z.observed_at between d.observed_at - interval '30 minutes' and d.observed_at
      and z.action = d.action and z.veto_reason is null
      and public.brian_asset_sector_v1(z.asset_id) = public.brian_asset_sector_v1(d.asset_id)) b
  where d.observed_at >= now() - make_interval(mins => h.mins) - p_window
    and d.observed_at < now() - make_interval(mins => h.mins + 20)
    and d.action in ('OPEN_LONG', 'OPEN_SHORT') and d.direction in (-1, 1) and d.veto_reason is null
    and d.session_state = 'REGULAR' and d.observed_reference_price > 0 and d.estimated_round_trip_cost_bps >= 0
  on conflict (decision_id, horizon) do nothing;
  get diagnostics ma = row_count;

  insert into public.brian_treasury_horizon_outcomes
    (decision_id, horizon, lane, asset_id, asset_class, sector, direction, observed_at, cost_bps, entry_price, exit_price,
     exit_observed_at, gross_bps, news_class, linked_themes, theme_sector_match, breadth)
  select d.decision_id, 'SESSION_CLOSE', 'MULTIASSET_EVENT_REACTION', d.asset_id, coalesce(d.asset_class, 'unknown'),
         public.brian_asset_sector_v1(d.asset_id), d.direction::smallint, d.observed_at, d.estimated_round_trip_cost_bps,
         d.observed_reference_price, x.p, x.t, d.direction * (x.p / d.observed_reference_price - 1) * 10000,
         coalesce(public.brian_news_class_v1(d.linked_event_ids), 'UNKNOWN'), th.themes,
         (th.themes && public.brian_sector_primary_themes_v1(public.brian_asset_sector_v1(d.asset_id))) or ('DIRECT_ASSET' = any(th.themes)),
         b.n
  from public.brian_multiasset_alpha_decisions d
  cross join lateral (
    select array(select jsonb_array_elements_text(coalesce(d.metadata->'linked_themes', '[]'::jsonb))) themes) th
  cross join lateral (
    select y.observed_reference_price p, y.observed_at t
    from public.brian_multiasset_alpha_decisions y
    where y.asset_id = d.asset_id and y.observed_at > d.observed_at
      and y.observed_at < date_trunc('day', d.observed_at at time zone 'UTC') at time zone 'UTC' + interval '1 day'
      and y.session_state = 'REGULAR' and y.observed_reference_price > 0
    order by y.observed_at desc
    limit 1) x
  cross join lateral (
    select count(distinct z.asset_id)::int n
    from public.brian_multiasset_alpha_decisions z
    where z.observed_at between d.observed_at - interval '30 minutes' and d.observed_at
      and z.action = d.action and z.veto_reason is null
      and public.brian_asset_sector_v1(z.asset_id) = public.brian_asset_sector_v1(d.asset_id)) b
  where d.observed_at >= now() - interval '1 day' - p_window
    and (d.observed_at at time zone 'UTC')::date < (now() at time zone 'UTC')::date
    and d.action in ('OPEN_LONG', 'OPEN_SHORT') and d.direction in (-1, 1) and d.veto_reason is null
    and d.session_state = 'REGULAR' and d.observed_reference_price > 0 and d.estimated_round_trip_cost_bps >= 0
  on conflict (decision_id, horizon) do nothing;
  get diagnostics sc = row_count;

  insert into public.brian_treasury_horizon_outcomes
    (decision_id, horizon, lane, asset_id, asset_class, sector, direction, observed_at, cost_bps, entry_price, exit_price,
     exit_observed_at, gross_bps, news_class, linked_themes, theme_sector_match, breadth)
  select d.decision_id, h.label, 'CANONICAL_ALPHA', d.asset_id, 'crypto', 'CRYPTO', d.direction::smallint, d.observed_at,
         d.estimated_round_trip_cost_bps, d.observed_reference_price, x.p, x.t,
         d.direction * (x.p / d.observed_reference_price - 1) * 10000, 'NONE', '{}', null, b.n
  from public.brian_alpha_decisions d
  cross join (values ('240M', 240), ('1440M', 1440)) h(label, mins)
  cross join lateral (
    select y.observed_reference_price p, y.observed_at t
    from public.brian_alpha_decisions y
    where y.asset_id = d.asset_id
      and y.observed_at between d.observed_at + make_interval(mins => h.mins - 20) and d.observed_at + make_interval(mins => h.mins + 20)
      and y.observed_reference_price > 0
    order by abs(extract(epoch from (y.observed_at - d.observed_at)) - h.mins * 60)
    limit 1) x
  cross join lateral (
    select count(distinct z.asset_id)::int n
    from public.brian_alpha_decisions z
    where z.action = d.action and z.observed_at between d.observed_at - interval '30 minutes' and d.observed_at
      and z.veto_reason is null) b
  where d.observed_at >= now() - make_interval(mins => h.mins) - p_window
    and d.observed_at < now() - make_interval(mins => h.mins + 20)
    and d.action in ('OPEN_LONG', 'OPEN_SHORT') and d.direction in (-1, 1) and d.veto_reason is null
    and d.observed_reference_price > 0 and d.estimated_round_trip_cost_bps >= 0
  on conflict (decision_id, horizon) do nothing;
  get diagnostics cr = row_count;

  -- Futures-roll artefacts (commodities): a single >4% step between consecutive observations inside the window.
  with steps as (
    select asset_id, observed_at t, observed_reference_price p,
           lag(observed_reference_price) over (partition by asset_id order by observed_at) pp,
           lag(observed_at) over (partition by asset_id order by observed_at) pt
    from public.brian_multiasset_alpha_decisions
    where asset_class = 'commodity' and observed_at >= now() - interval '2 days' - p_window and observed_reference_price > 0),
  jumps as (select asset_id, t from steps where pp > 0 and t - pt <= interval '20 minutes' and abs(ln(p / pp)) > 0.04)
  update public.brian_treasury_horizon_outcomes o
     set excluded_reason = 'FUTURES_ROLL_DISCONTINUITY'
   where o.excluded_reason is null and o.asset_class = 'commodity'
     and o.observed_at >= now() - interval '2 days' - p_window
     and exists (select 1 from jumps j where j.asset_id = o.asset_id and j.t > o.observed_at and j.t <= o.exit_observed_at);

  return jsonb_build_object('multiasset', ma, 'session_close', sc, 'crypto', cr);
end $fn$;
revoke all on function public.brian_record_treasury_horizon_outcomes_v1(interval) from public, anon, authenticated, service_role;

create table if not exists public.brian_treasury_horizon_segment_cache (
  horizon text not null,
  lane text not null,
  sector text not null,
  direction smallint not null,
  news_class text not null,
  theme_sector_match text not null,
  breadth_bucket text not null,
  n bigint not null,
  gross_bps double precision not null,
  se_bps double precision not null,
  hit_rate double precision not null,
  cost_bps double precision not null,
  last_observed_at timestamptz,
  computed_at timestamptz not null default now(),
  primary key (horizon, lane, sector, direction, news_class, theme_sector_match, breadth_bucket)
);
alter table public.brian_treasury_horizon_segment_cache enable row level security;
revoke all on public.brian_treasury_horizon_segment_cache from public, anon, authenticated;
grant select on public.brian_treasury_horizon_segment_cache to service_role;

create or replace function public.brian_refresh_treasury_horizon_cache_v1(p_days integer default 14)
returns integer
language plpgsql
security definer
set search_path = pg_catalog, public
set statement_timeout = '60s'
as $fn$
declare n integer;
begin
  delete from public.brian_treasury_horizon_segment_cache;
  insert into public.brian_treasury_horizon_segment_cache
    (horizon, lane, sector, direction, news_class, theme_sector_match, breadth_bucket, n, gross_bps, se_bps, hit_rate, cost_bps, last_observed_at, computed_at)
  select horizon, lane, sector, direction, news_class,
         case when theme_sector_match is null then 'NA' when theme_sector_match then 'MATCH' else 'NO_MATCH' end,
         case when breadth >= 3 then '3+' when breadth = 2 then '2' else '1' end,
         count(*), avg(gross_bps), coalesce(stddev_samp(gross_bps) / sqrt(count(*)), 1e6),
         avg((gross_bps > cost_bps)::int::double precision), avg(cost_bps), max(observed_at), now()
  from public.brian_treasury_horizon_outcomes
  where excluded_reason is null
    and observed_at >= now() - make_interval(days => greatest(1, least(coalesce(p_days, 14), 30)))
  group by 1, 2, 3, 4, 5, 6, 7;
  get diagnostics n = row_count;
  return n;
end $fn$;
revoke all on function public.brian_refresh_treasury_horizon_cache_v1(integer) from public, anon, authenticated;

create or replace function public.brian_treasury_horizon_segments_v1()
returns setof public.brian_treasury_horizon_segment_cache
language sql stable security definer set search_path = pg_catalog, public as $fn$
  select * from public.brian_treasury_horizon_segment_cache
$fn$;
revoke all on function public.brian_treasury_horizon_segments_v1() from public, anon, authenticated;
grant execute on function public.brian_treasury_horizon_segments_v1() to service_role;

-- The existing hourly evidence pipeline (job 106) also records and summarises the longer horizons.
create or replace function public.brian_treasury_evidence_hourly_v1()
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $fn$
declare a integer; c integer; r integer; s integer; hz jsonb; hs integer;
begin
  a := public.brian_record_treasury_signal_outcomes_v1(interval '3 hours');
  c := public.brian_record_treasury_crypto_outcomes_v1(interval '3 hours');
  r := public.brian_mark_treasury_roll_outcomes_v1(interval '1 day');
  s := public.brian_refresh_treasury_segment_cache_v1(7);
  hz := public.brian_record_treasury_horizon_outcomes_v1(interval '3 hours');
  hs := public.brian_refresh_treasury_horizon_cache_v1(14);
  return jsonb_build_object('multiasset', a, 'crypto', c, 'roll_excluded', r, 'segments', s,
                            'horizons', hz, 'horizon_segments', hs, 'at', now());
end $fn$;
revoke all on function public.brian_treasury_evidence_hourly_v1() from public, anon, authenticated;

-- Integration status: horizons component (informational; not part of feeds_treasury).
create or replace function public.brian_treasury_integration_status_v1(p_now timestamptz default now())
returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $fn$
  with
  t as (select observed_at, equity_usd, cash_usd, deployment_pct, treasury_version, gate_version, blocked_reasons
        from public.brian_treasury_shadow_latest order by observed_at desc limit 1),
  cfg as (select config_value from public.brian_evolution_runtime_config where config_key = 'brian_treasury_unified_enabled'),
  cache as (select count(*) segments, max(computed_at) computed_at,
                   count(*) filter (where n >= 100 and gross_bps * n / (n + 100.0) - se_bps - cost_bps >= 3) clearing
            from public.brian_treasury_segment_edge_cache),
  lanes as (select coalesce(jsonb_object_agg(lane, jsonb_build_object('n7d', n, 'last_exit_at', last_at)), '{}'::jsonb) v
            from (select lane, count(*) n, max(exit_observed_at) last_at from public.brian_treasury_signal_outcomes
                  where observed_at >= p_now - interval '7 days' group by lane) x),
  eye as (select max(observed_at) at from public.brian_prediction_source_health where observed_at >= p_now - interval '1 day'),
  pilot as (select state, last_heartbeat_at from public.brian_niulai4_pilot_runs order by started_at desc limit 1),
  audit as (select max(created_at) at from public.brian_missed_opportunity_receipts where created_at >= p_now - interval '7 days'),
  hz as (select max(computed_at) computed_at, count(*) segments,
                (select coalesce(jsonb_object_agg(horizon, n), '{}'::jsonb) from (select horizon, count(*) n from public.brian_treasury_horizon_outcomes
                  where observed_at >= p_now - interval '14 days' and excluded_reason is null group by 1) x) rows_14d
         from public.brian_treasury_horizon_segment_cache),
  jobs as (select coalesce(jsonb_object_agg(jobname, active), '{}'::jsonb) v from cron.job
           where jobname in ('brian-evolution-treasury-1m', 'brian-treasury-signal-outcomes-hourly', 'brian-prediction-eye-15m',
                             'brian-pm-calibration-shadow-hourly', 'brian-cold-archive-daily')),
  c as (
    select
      jsonb_build_object('status', case when t.observed_at >= p_now - interval '15 minutes' then 'OK' else 'STALE' end,
        'observed_at', t.observed_at, 'equity_usd', t.equity_usd, 'cash_usd', t.cash_usd, 'deployment_pct', t.deployment_pct,
        'treasury_version', t.treasury_version, 'gate_version', t.gate_version, 'blocked_reasons', t.blocked_reasons) treasury,
      jsonb_build_object('status', case when (select config_value from cfg) = 'true' then 'OK' else 'OFF' end,
        'lanes', jsonb_build_array('MULTIASSET_EVENT_REACTION', 'CANONICAL_ALPHA')) unified,
      jsonb_build_object('status', case when cache.computed_at >= p_now - interval '2 hours' then 'OK' else 'STALE' end,
        'computed_at', cache.computed_at, 'segments', cache.segments, 'segments_clearing_cost', cache.clearing,
        'lanes', (select v from lanes)) evidence,
      jsonb_build_object('status', case when eye.at >= p_now - interval '1 hour' then 'OK' else 'STALE' end,
        'observed_at', eye.at, 'role', 'news and prediction-market sensors; news trust class feeds the evidence split') prediction_eye,
      (select public.brian_pm_calibration_status_v1()) || jsonb_build_object('status', 'OK',
        'role', 'pre-registered Polymarket study; joins the treasury only if its pre-registered rule passes') pm_calibration,
      jsonb_build_object('status', case when pilot.last_heartbeat_at >= p_now - interval '30 minutes' then 'OK' else 'STALE' end,
        'state', pilot.state, 'last_heartbeat_at', pilot.last_heartbeat_at, 'host', 'LOCAL_PC',
        'role', 'shadow pilot; never copied, joins only after its own evaluation') niulai4,
      jsonb_build_object('status', 'LOCAL_ONLY', 'host', 'LOCAL_PC',
        'role', 'order-flow collection; gaps while the PC is off, cloud host pending (operator)') ofi,
      jsonb_build_object('status', case when audit.at is not null then 'OK' else 'STALE' end, 'last_receipt_at', audit.at,
        'role', 'hindsight report only; the treasury learns from the point-in-time evidence ledger instead') opportunity_audit,
      jsonb_build_object('status', case when hz.computed_at >= p_now - interval '2 hours' then 'OK' else 'STALE' end,
        'computed_at', hz.computed_at, 'segments', hz.segments, 'rows_14d', hz.rows_14d,
        'role', 'measure-only 4h / 1d / session-close outcomes with theme-sector match and breadth; not used for treasury decisions yet') horizons,
      (select v from jobs) jobs
    from (select 1) one
    left join t on true left join cache on true left join eye on true left join pilot on true left join audit on true left join hz on true
  )
  select jsonb_build_object(
    'version', 'brian.integration-status.v1',
    'observed_at', p_now,
    'integrated', (treasury->>'status') = 'OK' and (unified->>'status') = 'OK' and (evidence->>'status') = 'OK'
                  and (prediction_eye->>'status') = 'OK' and not exists (select 1 from jsonb_each(jobs) j where j.value <> 'true'::jsonb),
    'feeds_treasury', jsonb_build_array('treasury', 'unified', 'evidence', 'prediction_eye'),
    'components', jsonb_build_object('treasury', treasury, 'unified', unified, 'evidence', evidence, 'prediction_eye', prediction_eye,
      'pm_calibration', pm_calibration, 'niulai4', niulai4, 'ofi', ofi, 'opportunity_audit', opportunity_audit, 'horizons', horizons),
    'jobs', jobs,
    'shadow_only', true, 'live_execution', false, 'trade_signal', false)
  from c
$fn$;
revoke all on function public.brian_treasury_integration_status_v1(timestamptz) from public, anon, authenticated;
grant execute on function public.brian_treasury_integration_status_v1(timestamptz) to service_role;

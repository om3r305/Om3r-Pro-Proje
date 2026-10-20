-- Brian Prediction Intelligence -- Phase 2c: resolutions and point-in-time calibration inputs.
--
-- Adds an append-only record of GENUINE market resolutions (closed, UMA "resolved", final prices
-- exactly one 1 and the rest 0 -- enforced by the collector and re-checked here), the bounded list
-- of markets that are due a resolution lookup, and the calibration input view.
--
-- Calibration inputs are point-in-time: for each resolved Yes/No market the forecast is the latest
-- snapshot RECEIVED at or before (end_date - lead). A market with no snapshot that early has no
-- forecast and is excluded (counted), never back-filled with a later price. Resolutions observed
-- after p_as_of are invisible to an as-of query. The cluster key is event_id (market_key when the
-- source gave no event), so dependent markets of one event are resampled together downstream.
--
-- Nothing here enables anything new by itself: resolutions are only written by the eye while
-- module_enabled is on, and calibration is a read-only query. No trade signal.

-- CHECK constraints cannot hold subqueries, so the "exactly one 1, rest 0" rule lives in an
-- immutable helper.
create or replace function public.brian_prediction_prices_are_final(p jsonb)
returns boolean
language sql
immutable
set search_path = pg_catalog, public
as $$
  select jsonb_typeof(p) = 'array' and jsonb_array_length(p) > 0
     and (select count(*) from jsonb_array_elements(p) e where (e->>'price')::numeric = 1) = 1
     and (select count(*) from jsonb_array_elements(p) e where (e->>'price') is null or (e->>'price')::numeric not in (0, 1)) = 0;
$$;

create table if not exists public.brian_prediction_resolutions (
  resolution_id text primary key,
  market_key text not null unique references public.brian_prediction_markets(market_key),
  winning_label text not null check (length(trim(winning_label)) > 0),
  yes_outcome smallint check (yes_outcome is null or yes_outcome in (0, 1)),
  final_prices jsonb not null check (jsonb_typeof(final_prices) = 'array' and jsonb_array_length(final_prices) > 0),
  uma_resolution_status text not null check (uma_resolution_status = 'resolved'),
  closed_time timestamptz,
  received_at timestamptz not null,
  ingested_at timestamptz not null default now(),
  model_version text not null,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  -- Exactly one winning price of 1, every other price 0.
  constraint brian_prediction_resolution_prices_final check (public.brian_prediction_prices_are_final(final_prices))
);

alter table public.brian_prediction_resolutions enable row level security;
revoke all on public.brian_prediction_resolutions from anon, authenticated;
revoke update, delete, truncate, references, trigger on public.brian_prediction_resolutions from service_role;
grant select, insert on public.brian_prediction_resolutions to service_role;
drop trigger if exists brian_prediction_resolutions_append_only on public.brian_prediction_resolutions;
create trigger brian_prediction_resolutions_append_only
  before update or delete on public.brian_prediction_resolutions
  for each row execute function public.brian_reject_mutation();

-- The storage cap now also counts resolutions.
create or replace function public.brian_prediction_storage_bytes()
returns bigint
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  select coalesce(sum(pg_total_relation_size(c.oid)), 0)::bigint
  from pg_class c
  where c.relnamespace = 'public'::regnamespace
    and c.relkind = 'r'
    and c.relname in (
      'brian_prediction_markets', 'brian_prediction_market_snapshots',
      'brian_prediction_wallet_trades', 'brian_prediction_source_health', 'brian_prediction_wallets',
      'brian_prediction_resolutions'
    );
$$;

-- Markets whose end date has passed (plus a grace period for the oracle) and that have no
-- recorded resolution, oldest first, bounded.
create or replace function public.brian_prediction_resolution_candidates_v1(
  p_as_of timestamptz default now(),
  p_limit integer default 10,
  p_grace interval default interval '1 hour'
) returns table(market_id text, market_key text, end_date timestamptz)
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  select m.market_id, m.market_key, m.end_date
  from public.brian_prediction_markets m
  where m.end_date is not null
    and m.end_date <= p_as_of - coalesce(p_grace, interval '1 hour')
    and not exists (select 1 from public.brian_prediction_resolutions r where r.market_key = m.market_key)
  order by m.end_date asc, m.market_key
  limit greatest(1, least(coalesce(p_limit, 10), 10));
$$;

create or replace function public.brian_record_prediction_resolutions_v1(p_rows jsonb)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_rows jsonb := coalesce(p_rows, '[]'::jsonb);
  v_module boolean;
  v_cap_mb integer;
  v_stored integer := 0;
begin
  if jsonb_typeof(v_rows) <> 'array' then raise exception 'BRIAN_PREDICTION: resolutions must be a JSON array'; end if;
  if jsonb_array_length(v_rows) > 10 then raise exception 'BRIAN_PREDICTION: resolution batch exceeds the per-call bound'; end if;
  select module_enabled, max_storage_mb into v_module, v_cap_mb
    from public.brian_prediction_control where control_id = 'prediction-intelligence';
  if v_module is distinct from true then return jsonb_build_object('status', 'OPERATOR_STOPPED'); end if;
  if public.brian_prediction_storage_bytes() >= v_cap_mb::bigint * 1024 * 1024 then
    return jsonb_build_object('status', 'STORAGE_CAP_REACHED');
  end if;
  with ins as (
    insert into public.brian_prediction_resolutions (
      resolution_id, market_key, winning_label, yes_outcome, final_prices, uma_resolution_status,
      closed_time, received_at, model_version
    )
    select r.resolution_id, r.market_key, r.winning_label, r.yes_outcome, r.final_prices,
           r.uma_resolution_status, r.closed_time, r.received_at, r.model_version
    from jsonb_to_recordset(v_rows) as r(
      resolution_id text, market_key text, winning_label text, yes_outcome smallint, final_prices jsonb,
      uma_resolution_status text, closed_time timestamptz, received_at timestamptz, model_version text
    )
    where exists (select 1 from public.brian_prediction_markets m where m.market_key = r.market_key)
    on conflict do nothing
    returning 1
  ) select count(*) into v_stored from ins;
  return jsonb_build_object('status', 'SUCCESS', 'received', jsonb_array_length(v_rows), 'stored', v_stored);
end;
$$;

-- One point-in-time forecast per resolved Yes/No market. Rows without an early-enough forecast are
-- counted in 'excluded', not silently dropped.
create or replace function public.brian_prediction_calibration_inputs_v1(
  p_as_of timestamptz default now(),
  p_lead interval default interval '24 hours'
) returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $$
  with params as (select p_as_of as as_of, coalesce(p_lead, interval '24 hours') as lead),
  resolved as (
    select r.market_key, r.yes_outcome, r.received_at as resolution_received_at, m.end_date,
           coalesce(m.event_id, m.market_key) as cluster_key
    from public.brian_prediction_resolutions r
    join public.brian_prediction_markets m on m.market_key = r.market_key
    cross join params
    where r.received_at <= params.as_of and r.yes_outcome is not null and m.end_date is not null
  ),
  forecasts as (
    select rs.*, f.p as forecast, f.received_at as forecast_received_at
    from resolved rs
    cross join params
    left join lateral (
      select coalesce(
               (select (o->>'implied_probability')::numeric from jsonb_array_elements(s.outcomes) o where o->>'label' = 'Yes' limit 1),
               (s.outcomes->0->>'implied_probability')::numeric) as p,
             s.received_at
      from public.brian_prediction_market_snapshots s
      where s.market_key = rs.market_key
        and s.received_at <= rs.end_date - params.lead
        and s.received_at <= params.as_of
      order by s.received_at desc
      limit 1
    ) f on true
  )
  select jsonb_build_object(
    'as_of', (select as_of from params),
    'lead_seconds', (select extract(epoch from lead)::integer from params),
    'method', 'latest_snapshot_received_at_or_before_end_date_minus_lead',
    'trade_signal', false,
    'rows', coalesce((select jsonb_agg(jsonb_build_object(
        'eventId', cluster_key, 'marketKey', market_key, 'forecast', forecast, 'outcome', yes_outcome,
        'forecastReceivedAt', forecast_received_at, 'endDate', end_date) order by end_date, market_key)
      from forecasts where forecast is not null), '[]'::jsonb),
    'excluded', jsonb_build_object(
      'no_forecast_before_lead', (select count(*) from forecasts where forecast_received_at is null),
      'null_probability', (select count(*) from forecasts where forecast_received_at is not null and forecast is null),
      'non_binary_resolutions', (select count(*) from public.brian_prediction_resolutions r, params
                                 where r.received_at <= params.as_of and r.yes_outcome is null)
    )
  );
$$;

revoke all on function public.brian_prediction_resolution_candidates_v1(timestamptz, integer, interval) from public, anon, authenticated;
revoke all on function public.brian_record_prediction_resolutions_v1(jsonb) from public, anon, authenticated;
revoke all on function public.brian_prediction_calibration_inputs_v1(timestamptz, interval) from public, anon, authenticated;
revoke all on function public.brian_prediction_storage_bytes() from public, anon, authenticated;
grant execute on function public.brian_prediction_resolution_candidates_v1(timestamptz, integer, interval) to service_role;
grant execute on function public.brian_record_prediction_resolutions_v1(jsonb) to service_role;
grant execute on function public.brian_prediction_calibration_inputs_v1(timestamptz, interval) to service_role;
grant execute on function public.brian_prediction_storage_bytes() to service_role;

do $$
begin
  if to_regclass('public.brian_archive_policies') is null then return; end if;
  insert into public.brian_archive_policies
    (table_name, time_column, pk_column, hot_retention_days, batch_size, archive_enabled, purge_enabled, priority, notes, updated_at)
  values ('brian_prediction_resolutions', 'received_at', 'resolution_id', 3650, 500, true, false, 45,
          'Prediction resolutions: small, needed for calibration; archived, never purged.', now())
  on conflict (table_name) do nothing;
end
$$;

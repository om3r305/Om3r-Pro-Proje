-- Brian Prediction Intelligence -- Phase 1c: bounded storage.
--
-- Two independent bounds, because the existing cold archive (brian-cold-archive) has been idle
-- since 2026-10-03 and must not be the only thing standing between this module and the disk:
--
--   1. HARD CAP. brian_prediction_control.max_storage_mb (default 256 MB). The batch recorder
--      measures the module's own tables on every call; at or above the cap it stores no new
--      markets/snapshots/trades and returns STORAGE_CAP_REACHED (health rows are still kept, so
--      the reason is visible). Nothing is deleted to make room.
--   2. ARCHIVE POLICIES. The evidence tables are registered in brian_archive_policies so the
--      existing archive -> verify -> 24h grace -> purge path applies to them whenever an operator
--      runs it. Rows are only ever deleted after a verified cold-archive copy exists; this
--      migration deletes nothing and does not (re)enable any schedule.
--
-- brian_prediction_markets is archive-only (never purged): it is small and snapshots reference it.
-- The policy insert is skipped when brian_archive_policies does not exist (e.g. a bare CI database).

alter table public.brian_prediction_control
  add column if not exists max_storage_mb integer not null default 256
  check (max_storage_mb between 1 and 2048);

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
      'brian_prediction_wallet_trades', 'brian_prediction_source_health', 'brian_prediction_wallets'
    );
$$;

revoke all on function public.brian_prediction_storage_bytes() from public, anon, authenticated;
grant execute on function public.brian_prediction_storage_bytes() to service_role;

create or replace function public.brian_record_prediction_batch_v1(
  p_markets jsonb,
  p_snapshots jsonb,
  p_trades jsonb,
  p_health jsonb
) returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_module boolean;
  v_wallets boolean;
  v_cap_mb integer;
  v_bytes bigint;
  v_cap_bytes bigint;
  v_markets jsonb := coalesce(p_markets, '[]'::jsonb);
  v_snapshots jsonb := coalesce(p_snapshots, '[]'::jsonb);
  v_trades jsonb := coalesce(p_trades, '[]'::jsonb);
  v_health jsonb := coalesce(p_health, '[]'::jsonb);
  v_markets_stored integer := 0;
  v_snapshots_stored integer := 0;
  v_trades_stored integer := 0;
  v_health_stored integer := 0;
begin
  if jsonb_typeof(v_markets) <> 'array' or jsonb_typeof(v_snapshots) <> 'array'
     or jsonb_typeof(v_trades) <> 'array' or jsonb_typeof(v_health) <> 'array' then
    raise exception 'BRIAN_PREDICTION: every batch section must be a JSON array';
  end if;
  if jsonb_array_length(v_markets) > 300 or jsonb_array_length(v_snapshots) > 300
     or jsonb_array_length(v_trades) > 1000 or jsonb_array_length(v_health) > 50 then
    raise exception 'BRIAN_PREDICTION: batch exceeds the per-call size bound';
  end if;

  select module_enabled, wallet_tracking_enabled, max_storage_mb into v_module, v_wallets, v_cap_mb
    from public.brian_prediction_control where control_id = 'prediction-intelligence';
  if v_module is distinct from true then
    return jsonb_build_object('status', 'OPERATOR_STOPPED');
  end if;

  -- Hard storage cap, independent of whether the cold archive is running. At or above the cap no
  -- new evidence is stored; only the (tiny, bounded) health rows are kept so the reason is visible.
  v_bytes := public.brian_prediction_storage_bytes();
  v_cap_bytes := v_cap_mb::bigint * 1024 * 1024;
  if v_bytes >= v_cap_bytes then
    with ins as (
      insert into public.brian_prediction_source_health (
        health_id, source, status, http_status, attempts, latency_ms, records_seen, records_stored,
        detail, observed_at, metadata
      )
      select r.health_id, r.source, r.status, r.http_status, r.attempts, r.latency_ms,
             coalesce(r.records_seen, 0), 0, r.detail, r.observed_at,
             coalesce(r.metadata, '{}'::jsonb) || jsonb_build_object('storage_cap_reached', true)
      from jsonb_to_recordset(v_health) as r(
        health_id text, source text, status text, http_status integer, attempts integer, latency_ms integer,
        records_seen integer, records_stored integer, detail text, observed_at timestamptz, metadata jsonb
      )
      on conflict (health_id) do nothing
      returning 1
    ) select count(*) into v_health_stored from ins;
    return jsonb_build_object(
      'status', 'STORAGE_CAP_REACHED',
      'storage_bytes', v_bytes,
      'cap_bytes', v_cap_bytes,
      'health', jsonb_build_object('received', jsonb_array_length(v_health), 'stored', v_health_stored)
    );
  end if;

  with ins as (
    insert into public.brian_prediction_markets (
      market_key, source, market_id, condition_id, event_id, event_title, question, slug,
      end_date, outcomes, first_seen_at
    )
    select r.market_key, r.source, r.market_id, r.condition_id, r.event_id, r.event_title, r.question,
           r.slug, r.end_date, r.outcomes, r.first_seen_at
    from jsonb_to_recordset(v_markets) as r(
      market_key text, source text, market_id text, condition_id text, event_id text, event_title text,
      question text, slug text, end_date timestamptz, outcomes jsonb, first_seen_at timestamptz
    )
    on conflict (market_key) do nothing
    returning 1
  ) select count(*) into v_markets_stored from ins;

  with ins as (
    insert into public.brian_prediction_market_snapshots (
      snapshot_id, market_key, source_updated_at, received_at, outcomes, best_bid, best_ask, spread,
      liquidity, volume_24h, active, closed, source_freshness_seconds, model_version
    )
    select r.snapshot_id, r.market_key, r.source_updated_at, r.received_at, r.outcomes, r.best_bid,
           r.best_ask, r.spread, r.liquidity, r.volume_24h, r.active, r.closed,
           r.source_freshness_seconds, r.model_version
    from jsonb_to_recordset(v_snapshots) as r(
      snapshot_id text, market_key text, source_updated_at timestamptz, received_at timestamptz,
      outcomes jsonb, best_bid numeric, best_ask numeric, spread numeric, liquidity numeric,
      volume_24h numeric, active boolean, closed boolean, source_freshness_seconds numeric,
      model_version text
    )
    on conflict (snapshot_id) do nothing
    returning 1
  ) select count(*) into v_snapshots_stored from ins;

  if v_wallets is true then
    with ins as (
      insert into public.brian_prediction_wallet_trades (
        idempotency_key, wallet, condition_id, asset_id, market_slug, title, outcome, side, price,
        size, usdc_size, transaction_hash, traded_at, received_at, model_version
      )
      select r.idempotency_key, r.wallet, r.condition_id, r.asset_id, r.market_slug, r.title, r.outcome,
             r.side, r.price, r.size, r.usdc_size, r.transaction_hash, r.traded_at, r.received_at,
             r.model_version
      from jsonb_to_recordset(v_trades) as r(
        idempotency_key text, wallet text, condition_id text, asset_id text, market_slug text, title text,
        outcome text, side text, price numeric, size numeric, usdc_size numeric, transaction_hash text,
        traded_at timestamptz, received_at timestamptz, model_version text
      )
      where exists (
        select 1 from public.brian_prediction_wallets w
        where w.wallet = lower(r.wallet) and w.verification_status = 'VERIFIED'
      )
      on conflict (idempotency_key) do nothing
      returning 1
    ) select count(*) into v_trades_stored from ins;
  end if;

  with ins as (
    insert into public.brian_prediction_source_health (
      health_id, source, status, http_status, attempts, latency_ms, records_seen, records_stored,
      detail, observed_at, metadata
    )
    select r.health_id, r.source, r.status, r.http_status, r.attempts, r.latency_ms,
           coalesce(r.records_seen, 0), coalesce(r.records_stored, 0), r.detail, r.observed_at,
           coalesce(r.metadata, '{}'::jsonb)
    from jsonb_to_recordset(v_health) as r(
      health_id text, source text, status text, http_status integer, attempts integer, latency_ms integer,
      records_seen integer, records_stored integer, detail text, observed_at timestamptz, metadata jsonb
    )
    on conflict (health_id) do nothing
    returning 1
  ) select count(*) into v_health_stored from ins;

  return jsonb_build_object(
    'status', 'SUCCESS',
    'storage_bytes', v_bytes,
    'cap_bytes', v_cap_bytes,
    'markets', jsonb_build_object('received', jsonb_array_length(v_markets), 'stored', v_markets_stored),
    'snapshots', jsonb_build_object('received', jsonb_array_length(v_snapshots), 'stored', v_snapshots_stored),
    'trades', jsonb_build_object(
      'received', jsonb_array_length(v_trades), 'stored', v_trades_stored, 'tracking_enabled', v_wallets
    ),
    'health', jsonb_build_object('received', jsonb_array_length(v_health), 'stored', v_health_stored)
  );
end;
$$;

revoke all on function public.brian_record_prediction_batch_v1(jsonb, jsonb, jsonb, jsonb) from public, anon, authenticated;
grant execute on function public.brian_record_prediction_batch_v1(jsonb, jsonb, jsonb, jsonb) to service_role;

do $$
begin
  if to_regclass('public.brian_archive_policies') is null then
    raise notice 'brian_archive_policies not present; prediction archive policies not registered';
    return;
  end if;
  insert into public.brian_archive_policies
    (table_name, time_column, pk_column, hot_retention_days, batch_size, archive_enabled, purge_enabled, priority, notes, updated_at)
  values
    ('brian_prediction_market_snapshots', 'received_at', 'snapshot_id', 90, 1000, true, true, 41,
     'Prediction snapshots: 90 days hot, then verified cold archive.', now()),
    ('brian_prediction_wallet_trades', 'traded_at', 'idempotency_key', 30, 1000, true, true, 42,
     'Public wallet trades: 30 days hot, then verified cold archive.', now()),
    ('brian_prediction_source_health', 'observed_at', 'health_id', 14, 1000, true, true, 43,
     'Prediction source health: 14 days hot, then verified cold archive.', now()),
    ('brian_prediction_markets', 'first_seen_at', 'market_key', 365, 500, true, false, 44,
     'Prediction market definitions: archived, never purged (snapshots reference them).', now())
  on conflict (table_name) do nothing;
end
$$;

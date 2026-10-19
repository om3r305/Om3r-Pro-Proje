-- Brian Prediction Intelligence -- Phase 1b storage (observation only).
--
-- Stores what Polymarket's PUBLIC read endpoints reported: market definitions, point-in-time
-- probability snapshots, public wallet trades of a verified wallet, and per-run source health.
-- It adds NO signal, NO decision threshold, NO order placement and NO live-decision authority.
-- Nothing in brian-2026 reads these tables yet; wiring them into decisions is later, separately
-- gated work. Every table follows the PROSPECTIVE_DEVELOPMENT_SHADOW convention
-- (evidence_class / shadow_only / live_execution) used by the other Phase 3.8+ tables.
--
-- Doctrine:
--   * Evidence tables are append-only: original observations are never rewritten. Re-ingesting the
--     same source row is a no-op because every row has a deterministic idempotency key.
--   * Missing values stay NULL. A missing price is never stored as 0.
--   * Two small operational-state tables are deliberately mutable (control switches and the
--     wallet registry's ingest watermark); they hold no evidence.
--   * Control defaults are all OFF. Applying this migration collects nothing by itself.
--   * RLS is on, anon/authenticated have no access, service_role gets only what it needs, and the
--     writer RPCs are SECURITY DEFINER with a pinned search_path and service_role-only EXECUTE.
--
-- Evaluations/signals tables are intentionally NOT created here: they belong with the Phase 2
-- analysis engine and creating unused tables now would only add surface area.
--
-- Like 202609030009_brian_collector_lease.sql this file needs no vault/pg_net/pg_cron, but it does
-- reference the Supabase roles anon/authenticated/service_role (bootstrapped in the CI fixture).

create or replace function public.brian_reject_mutation()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  if current_user = 'postgres' then
    if tg_op = 'DELETE' then return old; else return new; end if;
  end if;
  raise exception 'BRIAN_APPEND_ONLY: % on %.% is forbidden', tg_op, tg_table_schema, tg_table_name;
end;
$$;

-- ---------------------------------------------------------------------------------------------
-- Operational state 1/2: control switches (single row, all OFF by default).
-- ---------------------------------------------------------------------------------------------
create table if not exists public.brian_prediction_control (
  control_id text primary key default 'prediction-intelligence' check (control_id = 'prediction-intelligence'),
  module_enabled boolean not null default false,
  wallet_tracking_enabled boolean not null default false,
  max_markets integer not null default 60 check (max_markets between 1 and 200),
  min_liquidity numeric not null default 10000 check (min_liquidity >= 0),
  tag_slugs text[] not null default array['fed', 'fomc', 'cpi', 'inflation', 'economy', 'bitcoin'],
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  updated_at timestamptz not null default now()
);
insert into public.brian_prediction_control (control_id) values ('prediction-intelligence')
  on conflict (control_id) do nothing;

alter table public.brian_prediction_control enable row level security;
revoke all on public.brian_prediction_control from anon, authenticated;
revoke insert, delete, truncate, references, trigger on public.brian_prediction_control from service_role;
grant select, update on public.brian_prediction_control to service_role;

-- ---------------------------------------------------------------------------------------------
-- Operational state 2/2: wallet registry. A wallet is trackable only when VERIFIED, and a
-- VERIFIED row must carry a full 42-character address and a verification time. A shortened or
-- guessed address cannot be stored.
-- ---------------------------------------------------------------------------------------------
create table if not exists public.brian_prediction_wallets (
  wallet text primary key check (wallet ~ '^0x[0-9a-f]{40}$'),
  display_name text not null check (length(trim(display_name)) > 0),
  verification_status text not null check (verification_status in ('VERIFIED', 'UNVERIFIED', 'AMBIGUOUS')),
  verification_evidence jsonb not null default '[]'::jsonb,
  verified_at timestamptz,
  -- Highest trade time fully ingested; the collector resumes from here (Data API offsets are
  -- capped at 5000, so time-based resumption is required for an active trader).
  trade_watermark timestamptz,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  updated_at timestamptz not null default now(),
  constraint brian_prediction_wallet_verified_needs_time
    check (verification_status <> 'VERIFIED' or verified_at is not null)
);

alter table public.brian_prediction_wallets enable row level security;
revoke all on public.brian_prediction_wallets from anon, authenticated;
revoke delete, truncate, references, trigger on public.brian_prediction_wallets from service_role;
grant select, insert, update on public.brian_prediction_wallets to service_role;

-- ---------------------------------------------------------------------------------------------
-- Evidence (append-only).
-- ---------------------------------------------------------------------------------------------
create table if not exists public.brian_prediction_markets (
  market_key text primary key,
  source text not null,
  market_id text not null,
  condition_id text,
  event_id text,
  event_title text,
  question text not null check (length(trim(question)) > 0),
  slug text,
  end_date timestamptz,
  outcomes jsonb not null check (jsonb_typeof(outcomes) = 'array' and jsonb_array_length(outcomes) > 0),
  first_seen_at timestamptz not null,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);
create index if not exists brian_prediction_markets_condition_idx on public.brian_prediction_markets(condition_id);

create table if not exists public.brian_prediction_market_snapshots (
  snapshot_id text primary key,
  market_key text not null references public.brian_prediction_markets(market_key),
  -- Source-reported update time when the API provides one; NULL when it does not.
  source_updated_at timestamptz,
  received_at timestamptz not null,
  ingested_at timestamptz not null default now(),
  -- [{label, implied_probability|null, clob_token_id|null}] -- probabilities are never coerced to 0.
  outcomes jsonb not null check (jsonb_typeof(outcomes) = 'array'),
  best_bid numeric check (best_bid is null or (best_bid >= 0 and best_bid <= 1)),
  best_ask numeric check (best_ask is null or (best_ask >= 0 and best_ask <= 1)),
  spread numeric check (spread is null or spread >= 0),
  liquidity numeric check (liquidity is null or liquidity >= 0),
  volume_24h numeric check (volume_24h is null or volume_24h >= 0),
  active boolean not null,
  closed boolean not null,
  -- received_at - source_updated_at in seconds, when both are known; NULL otherwise.
  source_freshness_seconds numeric,
  model_version text not null,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);
create index if not exists brian_prediction_snapshots_market_time_idx
  on public.brian_prediction_market_snapshots(market_key, received_at desc);

create table if not exists public.brian_prediction_wallet_trades (
  idempotency_key text primary key,
  wallet text not null references public.brian_prediction_wallets(wallet),
  condition_id text not null,
  asset_id text not null,
  market_slug text,
  title text,
  outcome text,
  side text not null check (side in ('BUY', 'SELL')),
  price numeric not null check (price >= 0 and price <= 1),
  size numeric not null check (size > 0),
  usdc_size numeric check (usdc_size is null or usdc_size >= 0),
  transaction_hash text not null check (transaction_hash ~ '^0x[0-9a-f]{64}$'),
  traded_at timestamptz not null,
  received_at timestamptz not null,
  ingested_at timestamptz not null default now(),
  model_version text not null,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);
create index if not exists brian_prediction_trades_wallet_time_idx
  on public.brian_prediction_wallet_trades(wallet, traded_at desc);
create index if not exists brian_prediction_trades_condition_idx
  on public.brian_prediction_wallet_trades(condition_id, traded_at desc);

create table if not exists public.brian_prediction_source_health (
  health_id text primary key,
  source text not null,
  status text not null check (status in ('OK', 'DEGRADED', 'BLOCKED', 'RATE_LIMITED', 'ERROR', 'SKIPPED')),
  http_status integer,
  attempts integer check (attempts is null or attempts >= 0),
  latency_ms integer check (latency_ms is null or latency_ms >= 0),
  records_seen integer not null default 0 check (records_seen >= 0),
  records_stored integer not null default 0 check (records_stored >= 0),
  detail text,
  observed_at timestamptz not null,
  ingested_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);
create index if not exists brian_prediction_health_source_time_idx
  on public.brian_prediction_source_health(source, observed_at desc);

do $$
declare
  t text;
begin
  foreach t in array array[
    'brian_prediction_markets', 'brian_prediction_market_snapshots',
    'brian_prediction_wallet_trades', 'brian_prediction_source_health'
  ] loop
    execute format('alter table public.%I enable row level security', t);
    execute format('revoke all on public.%I from anon, authenticated', t);
    execute format('revoke update, delete, truncate, references, trigger on public.%I from service_role', t);
    execute format('grant select, insert on public.%I to service_role', t);
    execute format('drop trigger if exists %I on public.%I', t || '_append_only', t);
    execute format(
      'create trigger %I before update or delete on public.%I for each row execute function public.brian_reject_mutation()',
      t || '_append_only', t
    );
  end loop;
end
$$;

-- ---------------------------------------------------------------------------------------------
-- Writer RPCs.
-- ---------------------------------------------------------------------------------------------

-- Verified-wallet registry upsert. The trade watermark only ever moves forward. A failed
-- re-verification may downgrade a wallet (fail safe); a wallet is never promoted without evidence.
create or replace function public.brian_upsert_prediction_wallet_v1(
  p_wallet text,
  p_display_name text,
  p_status text,
  p_evidence jsonb,
  p_watermark timestamptz default null
) returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_wallet text := lower(coalesce(p_wallet, ''));
begin
  if v_wallet !~ '^0x[0-9a-f]{40}$' then
    raise exception 'BRIAN_PREDICTION: a full 42-character wallet address is required';
  end if;
  if p_status not in ('VERIFIED', 'UNVERIFIED', 'AMBIGUOUS') then
    raise exception 'BRIAN_PREDICTION: invalid verification status';
  end if;
  insert into public.brian_prediction_wallets as w (
    wallet, display_name, verification_status, verification_evidence, verified_at, trade_watermark, updated_at
  ) values (
    v_wallet, p_display_name, p_status, coalesce(p_evidence, '[]'::jsonb),
    case when p_status = 'VERIFIED' then now() else null end, p_watermark, now()
  )
  on conflict (wallet) do update set
    display_name = excluded.display_name,
    verification_status = excluded.verification_status,
    verification_evidence = excluded.verification_evidence,
    verified_at = case when excluded.verification_status = 'VERIFIED' then now() else null end,
    trade_watermark = case
      when excluded.trade_watermark is null then w.trade_watermark
      when w.trade_watermark is null then excluded.trade_watermark
      else greatest(w.trade_watermark, excluded.trade_watermark)
    end,
    updated_at = now();
  return jsonb_build_object('status', 'SUCCESS', 'wallet', v_wallet);
end;
$$;

-- Batch recorder. Returns OPERATOR_STOPPED (writing nothing) unless the module is enabled.
-- Trades are stored only for a wallet that is VERIFIED in the registry AND only while wallet
-- tracking is enabled; anything else is counted as received but not stored.
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

  select module_enabled, wallet_tracking_enabled into v_module, v_wallets
    from public.brian_prediction_control where control_id = 'prediction-intelligence';
  if v_module is distinct from true then
    return jsonb_build_object('status', 'OPERATOR_STOPPED');
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
    'markets', jsonb_build_object('received', jsonb_array_length(v_markets), 'stored', v_markets_stored),
    'snapshots', jsonb_build_object('received', jsonb_array_length(v_snapshots), 'stored', v_snapshots_stored),
    'trades', jsonb_build_object(
      'received', jsonb_array_length(v_trades), 'stored', v_trades_stored, 'tracking_enabled', v_wallets
    ),
    'health', jsonb_build_object('received', jsonb_array_length(v_health), 'stored', v_health_stored)
  );
end;
$$;

revoke all on function public.brian_upsert_prediction_wallet_v1(text, text, text, jsonb, timestamptz) from public, anon, authenticated;
revoke all on function public.brian_record_prediction_batch_v1(jsonb, jsonb, jsonb, jsonb) from public, anon, authenticated;
grant execute on function public.brian_upsert_prediction_wallet_v1(text, text, text, jsonb, timestamptz) to service_role;
grant execute on function public.brian_record_prediction_batch_v1(jsonb, jsonb, jsonb, jsonb) to service_role;

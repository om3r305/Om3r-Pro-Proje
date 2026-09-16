-- Catalyst Sentinel v1: persistent event watch, alerts, and fast official-source polling.
-- SHADOW only. No live execution surface is created by this migration.

create table if not exists public.brian_catalyst_sentinel_feed_state (
  endpoint_id text primary key,
  last_checked_at timestamptz,
  last_changed_at timestamptz,
  last_content_hash text,
  consecutive_failures integer not null default 0 check (consecutive_failures >= 0),
  last_error text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.brian_catalyst_sentinel_watches (
  watch_id text primary key,
  event_id text not null,
  asset_id text not null,
  source_id text not null,
  event_kind text not null,
  event_title text not null,
  event_published_at timestamptz,
  started_at timestamptz not null,
  expires_at timestamptz not null,
  status text not null default 'WATCHING'
    check (status in ('WATCHING','BUILDING','BREAKOUT_CANDIDATE','CONFIRMED','INVALIDATED','EXPIRED')),
  direction smallint not null default 0 check (direction in (-1,0,1)),
  reference_price double precision,
  last_price double precision,
  last_return double precision,
  reaction_score double precision not null default 0 check (reaction_score >= 0 and reaction_score <= 1),
  mfe double precision not null default 0,
  mae double precision not null default 0,
  spread_bps double precision,
  orderbook_imbalance double precision,
  realized_range_bps double precision,
  cross_market_confirmation double precision,
  last_evaluated_at timestamptz,
  last_state_change_at timestamptz,
  last_alpha_trigger_at timestamptz,
  recheck_count integer not null default 0 check (recheck_count >= 0),
  next_recheck_at timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_CATALYST_SENTINEL_SHADOW',
  shadow_only boolean not null default true check (shadow_only = true),
  live_execution boolean not null default false check (live_execution = false),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique(event_id, asset_id)
);

create index if not exists brian_catalyst_sentinel_watches_active_idx
  on public.brian_catalyst_sentinel_watches(status, expires_at desc);
create index if not exists brian_catalyst_sentinel_watches_event_asset_idx
  on public.brian_catalyst_sentinel_watches(event_id, asset_id, started_at desc);
create index if not exists brian_catalyst_sentinel_watches_recheck_idx
  on public.brian_catalyst_sentinel_watches(next_recheck_at)
  where status in ('WATCHING','BUILDING','BREAKOUT_CANDIDATE','CONFIRMED');

create table if not exists public.brian_catalyst_sentinel_alerts (
  alert_id text primary key,
  watch_id text not null references public.brian_catalyst_sentinel_watches(watch_id) on delete cascade,
  event_id text not null,
  asset_id text not null,
  observed_at timestamptz not null,
  alert_type text not null,
  direction smallint not null default 0 check (direction in (-1,0,1)),
  reaction_score double precision not null default 0 check (reaction_score >= 0 and reaction_score <= 1),
  price_return double precision,
  spread_bps double precision,
  orderbook_imbalance double precision,
  realized_range_bps double precision,
  cross_market_confirmation double precision,
  alpha_dispatched boolean not null default false,
  alpha_dispatch_request_id text,
  alpha_dispatched_at timestamptz,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_CATALYST_SENTINEL_SHADOW',
  shadow_only boolean not null default true check (shadow_only = true),
  live_execution boolean not null default false check (live_execution = false),
  created_at timestamptz not null default now()
);

create index if not exists brian_catalyst_sentinel_alerts_event_asset_idx
  on public.brian_catalyst_sentinel_alerts(event_id, asset_id, observed_at desc);
create index if not exists brian_catalyst_sentinel_alerts_dispatch_idx
  on public.brian_catalyst_sentinel_alerts(alpha_dispatched, observed_at desc);

alter table public.brian_catalyst_sentinel_feed_state enable row level security;
alter table public.brian_catalyst_sentinel_watches enable row level security;
alter table public.brian_catalyst_sentinel_alerts enable row level security;

revoke all on table public.brian_catalyst_sentinel_feed_state from anon, authenticated;
revoke all on table public.brian_catalyst_sentinel_watches from anon, authenticated;
revoke all on table public.brian_catalyst_sentinel_alerts from anon, authenticated;

-- Recreate the scheduler deterministically if this migration is re-applied manually.
do $$
begin
  perform cron.unschedule('brian-catalyst-sentinel-10s');
exception when others then
  null;
end $$;

select cron.schedule(
  'brian-catalyst-sentinel-10s',
  '10 seconds',
  $job$
  select net.http_post(
    url := (select decrypted_secret || '/functions/v1/brian-catalyst-sentinel' from vault.decrypted_secrets where name='brian_project_url' limit 1),
    headers := jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1)
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 9000
  );
  $job$
);

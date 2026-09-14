create table if not exists public.brian_dip_multiasset_state (
  engine_id text primary key,
  started_at timestamptz not null,
  run_until timestamptz not null,
  starting_equity numeric not null check (starting_equity > 0),
  cash numeric not null check (cash >= 0),
  realized_pnl numeric not null default 0,
  trade_count integer not null default 0 check (trade_count >= 0),
  win_count integer not null default 0 check (win_count >= 0),
  loss_count integer not null default 0 check (loss_count >= 0),
  positions jsonb not null default '{}'::jsonb,
  cooldowns jsonb not null default '{}'::jsonb,
  last_scan jsonb not null default '{}'::jsonb,
  last_eval_minute timestamptz,
  updated_at timestamptz not null default now(),
  enabled boolean not null default true,
  shadow_only boolean not null default true check (shadow_only = true),
  live_execution boolean not null default false check (live_execution = false)
);

create table if not exists public.brian_dip_multiasset_events (
  event_id uuid primary key default gen_random_uuid(),
  engine_id text not null,
  observed_at timestamptz not null default now(),
  symbol text not null,
  action text not null check (action in ('BUY','SELL')),
  price numeric not null check (price > 0),
  qty numeric not null check (qty > 0),
  notional numeric not null check (notional > 0),
  pnl numeric,
  reason text not null,
  metadata jsonb not null default '{}'::jsonb
);

create index if not exists brian_dip_multiasset_events_time_idx
  on public.brian_dip_multiasset_events (observed_at desc);
create index if not exists brian_dip_multiasset_events_symbol_time_idx
  on public.brian_dip_multiasset_events (symbol, observed_at desc);

create table if not exists public.brian_dip_multiasset_evaluations (
  evaluation_id bigint generated always as identity primary key,
  engine_id text not null,
  observed_at timestamptz not null default now(),
  symbol text not null,
  price numeric not null check (price > 0),
  radar_score numeric not null,
  signal_score numeric not null,
  action text not null check (action in ('READY','WAIT')),
  reason text not null,
  metadata jsonb not null default '{}'::jsonb
);

create index if not exists brian_dip_multiasset_eval_time_idx
  on public.brian_dip_multiasset_evaluations (observed_at desc);
create index if not exists brian_dip_multiasset_eval_symbol_time_idx
  on public.brian_dip_multiasset_evaluations (symbol, observed_at desc);

alter table public.brian_dip_multiasset_state enable row level security;
alter table public.brian_dip_multiasset_events enable row level security;
alter table public.brian_dip_multiasset_evaluations enable row level security;
revoke all on public.brian_dip_multiasset_state from anon, authenticated;
revoke all on public.brian_dip_multiasset_events from anon, authenticated;
revoke all on public.brian_dip_multiasset_evaluations from anon, authenticated;
grant all on public.brian_dip_multiasset_state to service_role;
grant all on public.brian_dip_multiasset_events to service_role;
grant all on public.brian_dip_multiasset_evaluations to service_role;
grant usage, select on sequence public.brian_dip_multiasset_evaluations_evaluation_id_seq to service_role;

select cron.unschedule(jobid) from cron.job where jobname = 'brian-dip-multiasset-worker-15s';
select cron.schedule(
  'brian-dip-multiasset-worker-15s',
  '15 seconds',
  $job$
  select net.http_post(
    url := (select decrypted_secret || '/functions/v1/brian-dip-multiasset-worker' from vault.decrypted_secrets where name='brian_project_url' limit 1),
    headers := jsonb_build_object(
      'Content-Type','application/json',
      'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
      'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1)
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 50000
  );
  $job$
);

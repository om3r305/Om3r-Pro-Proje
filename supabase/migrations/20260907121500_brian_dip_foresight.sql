create table if not exists public.brian_dip_foresight (
  id text primary key,
  session_id text not null,
  symbol text not null,
  created_at timestamptz not null default now(),
  due_at timestamptz not null,
  entry_price numeric not null,
  direction text not null check (direction in ('UP','DOWN','RANGE')),
  confidence double precision not null check (confidence >= 0 and confidence <= 1),
  horizon_min integer not null check (horizon_min between 1 and 60),
  predicted_peak numeric not null,
  predicted_trough numeric not null,
  predicted_close numeric not null,
  path jsonb not null default '[]'::jsonb,
  resolved_at timestamptz,
  actual_price numeric,
  hit boolean,
  abs_error_pct double precision,
  evidence_class text not null default 'AGGRESSIVE_DIP_FORESIGHT_SHADOW',
  shadow_only boolean not null default true check (shadow_only = true),
  live_execution boolean not null default false check (live_execution = false)
);

create index if not exists brian_dip_foresight_session_symbol_created_idx
  on public.brian_dip_foresight(session_id, symbol, created_at desc);
create index if not exists brian_dip_foresight_due_unresolved_idx
  on public.brian_dip_foresight(due_at) where resolved_at is null;

alter table public.brian_dip_foresight enable row level security;

-- Service-role Edge Functions write/read this table. No browser table policy is exposed.

do $$
begin
  perform cron.unschedule('brian-dip-foresight-v7-1m');
exception when others then null;
end $$;

select cron.schedule(
  'brian-dip-foresight-v7-1m',
  '* * * * *',
  $job$
  select net.http_post(
    url := (
      select decrypted_secret || '/functions/v1/brian-dip-foresight'
      from vault.decrypted_secrets
      where name = 'brian_project_url'
      limit 1
    ),
    headers := jsonb_build_object(
      'Content-Type', 'application/json',
      'Authorization', 'Bearer ' || (
        select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1
      ),
      'apikey', (
        select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1
      ),
      'x-brian-cron-key', (
        select decrypted_secret from vault.decrypted_secrets where name = 'brian_dashboard_cron_key' limit 1
      )
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 50000
  );
  $job$
);

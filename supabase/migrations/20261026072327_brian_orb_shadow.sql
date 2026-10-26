-- Opening-range breakout prospective shadow test orb-shadow-v1 (research/research_lab/studies/orb_shadow/
-- PREREGISTRATION.md). Operator 2026-10-26: full authority, integrate if free. Hypothetical trades on
-- public 5-minute bars; append-only; client roles have no access. One small daily job after the US close.

create table if not exists public.brian_orb_shadow_results (
  asset_id text not null,
  asset_class text not null,
  symbol text not null,
  session_date date not null,
  variant text not null check (variant in ('ORB_1R', 'ORB_CLOSE', 'ORB_IN_PLAY_1R')),
  period text not null check (period in ('RETRO', 'PROSPECTIVE')),
  status text not null check (status in ('TRADE', 'NO_TRIGGER', 'NOT_IN_PLAY', 'INSUFFICIENT_DATA')),
  direction smallint check (direction in (-1, 1)),
  or_high double precision, or_low double precision, or_volume double precision, rel_volume double precision,
  entry_at timestamptz, entry_price double precision, stop_price double precision, target_price double precision,
  exit_at timestamptz, exit_price double precision, exit_reason text check (exit_reason in ('TARGET', 'STOP', 'TIME')),
  gross_bps double precision, cost_bps double precision not null, net_bps double precision,
  rules_version text not null,
  rules_fingerprint text not null check (rules_fingerprint ~ '^[0-9a-f]{64}$'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  primary key (asset_id, session_date, variant),
  check ((status = 'TRADE') = (net_bps is not null))
);
drop trigger if exists brian_orb_shadow_results_append_only on public.brian_orb_shadow_results;
create trigger brian_orb_shadow_results_append_only before update or delete on public.brian_orb_shadow_results
  for each row execute function public.brian_reject_mutation();
alter table public.brian_orb_shadow_results enable row level security;
revoke all on public.brian_orb_shadow_results from public, anon, authenticated;
grant select, insert on public.brian_orb_shadow_results to service_role;

create or replace function public.brian_orb_shadow_status_v1()
returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $fn$
  select jsonb_build_object(
    'study_id', 'orb-shadow-v1',
    'variants', coalesce((select jsonb_object_agg(k, v) from (
      select period || ':' || variant k, jsonb_build_object(
        'sessions', count(distinct session_date), 'trades', count(*) filter (where status = 'TRADE'),
        'mean_net_bps', round(avg(net_bps)::numeric, 1), 'win_rate', round(avg((net_bps > 0)::int) filter (where status = 'TRADE')::numeric, 2),
        'sum_net_usd_on_2000', round((sum(net_bps) / 1e4 * 2000)::numeric, 2)) v
      from public.brian_orb_shadow_results group by period, variant) x), '{}'::jsonb),
    'last_session', (select max(session_date) from public.brian_orb_shadow_results),
    'decision', 'PENDING_PREREGISTERED_RULE',
    'shadow_only', true, 'live_execution', false, 'trade_signal', false)
$fn$;
revoke all on function public.brian_orb_shadow_status_v1() from public, anon, authenticated;
grant execute on function public.brian_orb_shadow_status_v1() to service_role;

insert into public.brian_system_job_registry (job_name, component, operator_managed)
values ('brian-orb-shadow-daily', 'RESEARCH', true)
on conflict do nothing;

-- 21:35 UTC Mon-Fri: after the 16:00 New York close in both summer (20:00 UTC) and winter (21:00 UTC) time.
select cron.schedule(
  'brian-orb-shadow-daily',
  '35 21 * * 1-5',
  $cron$
  select net.http_post(
    url := (select decrypted_secret || '/functions/v1/brian-orb-shadow' from vault.decrypted_secrets where name = 'brian_project_url' limit 1),
    headers := jsonb_build_object(
      'Content-Type', 'application/json',
      'Authorization', 'Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1),
      'apikey', (select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1),
      'x-brian-cron-key', (select decrypted_secret from vault.decrypted_secrets where name = 'brian_cron_key' limit 1)
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 120000
  )
  where not exists (select 1 from net.http_request_queue q where q.url like '%/functions/v1/brian-orb-shadow');
  $cron$
);

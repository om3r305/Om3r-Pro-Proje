-- Prospective shadow test of the Polymarket political-calibration result (pm-calibration-shadow.v1,
-- pre-registered in research/research_lab/studies/pm_calibration_shadow/PREREGISTRATION.md, commit
-- 08ea1bc, before any data). Approved by the operator on 2026-10-21. Hypothetical entries only:
-- nothing here can place an order. Both tables are append-only; client roles have no access.

create table if not exists public.brian_pm_calibration_entries (
  entry_id text primary key,
  market_id text not null,
  event_id text,
  question text not null,
  end_date timestamptz not null,
  captured_at timestamptz not null,
  rules_version text not null,
  rules_fingerprint text not null check (rules_fingerprint ~ '^[0-9a-f]{64}$'),
  yes_bid numeric, yes_ask numeric, yes_ask_shares numeric,
  no_bid numeric, no_ask numeric, no_ask_shares numeric,
  fee_rate numeric,
  mid numeric, q numeric,
  action text not null check (action in ('BUY_YES', 'BUY_NO', 'SKIP')),
  skip_reason text,
  price numeric, fee numeric, edge numeric,
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  unique (market_id, rules_version),
  check ((action = 'SKIP') = (skip_reason is not null)),
  check (action = 'SKIP' or (price > 0 and price < 1 and fee >= 0 and edge is not null))
);
create index if not exists brian_pm_calibration_entries_end_idx on public.brian_pm_calibration_entries (end_date);

create table if not exists public.brian_pm_calibration_outcomes (
  entry_id text primary key references public.brian_pm_calibration_entries (entry_id),
  resolved_at timestamptz not null,
  yes_outcome smallint not null check (yes_outcome in (0, 1)),
  return_per_dollar numeric,
  uma_resolution_status text not null,
  closed_time timestamptz,
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

drop trigger if exists brian_pm_calibration_entries_append_only on public.brian_pm_calibration_entries;
create trigger brian_pm_calibration_entries_append_only before update or delete on public.brian_pm_calibration_entries
  for each row execute function public.brian_reject_mutation();
drop trigger if exists brian_pm_calibration_outcomes_append_only on public.brian_pm_calibration_outcomes;
create trigger brian_pm_calibration_outcomes_append_only before update or delete on public.brian_pm_calibration_outcomes
  for each row execute function public.brian_reject_mutation();

alter table public.brian_pm_calibration_entries enable row level security;
alter table public.brian_pm_calibration_outcomes enable row level security;
revoke all on public.brian_pm_calibration_entries, public.brian_pm_calibration_outcomes from public, anon, authenticated;
grant select, insert on public.brian_pm_calibration_entries, public.brian_pm_calibration_outcomes to service_role;

-- Read-only summary for status endpoints (decision itself is computed offline per the pre-registration).
create or replace function public.brian_pm_calibration_status_v1()
returns jsonb
language sql
stable
security definer
set search_path = pg_catalog, public
as $fn$
  with e as (select * from public.brian_pm_calibration_entries),
  t as (select e.*, o.return_per_dollar, o.yes_outcome from e left join public.brian_pm_calibration_outcomes o using (entry_id) where e.action <> 'SKIP')
  select jsonb_build_object(
    'study_id', 'pm-calibration-shadow-v1',
    'captured', (select count(*) from e),
    'skipped', (select count(*) from e where action = 'SKIP'),
    'skip_reasons', coalesce((select jsonb_object_agg(skip_reason, n) from (select skip_reason, count(*) n from e where action = 'SKIP' group by 1) s), '{}'::jsonb),
    'trades', (select count(*) from t),
    'resolved_trades', (select count(*) from t where return_per_dollar is not null),
    'mean_return_per_dollar', (select avg(return_per_dollar) from t where return_per_dollar is not null),
    'mean_edge_at_entry', (select avg(edge) from t),
    'first_captured_at', (select min(captured_at) from e),
    'last_captured_at', (select max(captured_at) from e),
    'decision', 'PENDING_PREREGISTERED_RULE',
    'shadow_only', true, 'live_execution', false, 'trade_signal', false
  )
$fn$;
revoke all on function public.brian_pm_calibration_status_v1() from public, anon, authenticated;
grant execute on function public.brian_pm_calibration_status_v1() to service_role;

-- Entries past their scheduled end that have no outcome yet (oldest first, bounded).
create or replace function public.brian_pm_calibration_pending_v1(p_limit integer default 50)
returns table (entry_id text, market_id text, action text, price numeric, fee numeric)
language sql
stable
security definer
set search_path = pg_catalog, public
as $fn$
  select e.entry_id, e.market_id, e.action, e.price, e.fee
  from public.brian_pm_calibration_entries e
  where e.end_date < now() - interval '1 hour'
    and not exists (select 1 from public.brian_pm_calibration_outcomes o where o.entry_id = e.entry_id)
  order by e.end_date
  limit greatest(1, least(coalesce(p_limit, 50), 100))
$fn$;
revoke all on function public.brian_pm_calibration_pending_v1(integer) from public, anon, authenticated;
grant execute on function public.brian_pm_calibration_pending_v1(integer) to service_role;

insert into public.brian_system_job_registry (job_name, component, operator_managed)
values ('brian-pm-calibration-shadow-hourly', 'RESEARCH', true)
on conflict do nothing;

select cron.schedule(
  'brian-pm-calibration-shadow-hourly',
  '41 * * * *',
  $cron$
  select net.http_post(
    url := (select decrypted_secret || '/functions/v1/brian-pm-calibration-shadow' from vault.decrypted_secrets where name = 'brian_project_url' limit 1),
    headers := jsonb_build_object(
      'Content-Type', 'application/json',
      'Authorization', 'Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1),
      'apikey', (select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1),
      'x-brian-cron-key', (select decrypted_secret from vault.decrypted_secrets where name = 'brian_cron_key' limit 1)
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 120000
  )
  where not exists (select 1 from net.http_request_queue q where q.url like '%/functions/v1/brian-pm-calibration-shadow');
  $cron$
);

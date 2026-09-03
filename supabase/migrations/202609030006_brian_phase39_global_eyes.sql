-- Brian Phase 3.9 Global Eyes runtime health and scheduling.
-- All collectors are shadow-only and append-only.

create table if not exists public.brian_collector_runs (
  run_id text primary key,
  collector_id text not null,
  started_at timestamptz not null,
  finished_at timestamptz not null,
  status text not null check (status in ('SUCCESS','DEGRADED','FAILED','SKIPPED')),
  observed_records integer not null default 0 check (observed_records >= 0),
  stored_records integer not null default 0 check (stored_records >= 0),
  degraded_sources text[] not null default '{}',
  error_class text,
  error_message text,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  constraint brian_collector_run_time_order check (finished_at >= started_at)
);

create index if not exists brian_collector_runs_id_time_idx
  on public.brian_collector_runs(collector_id, finished_at desc);

alter table public.brian_collector_runs enable row level security;
revoke all on public.brian_collector_runs from anon, authenticated;
revoke update, delete, truncate, references, trigger on public.brian_collector_runs from service_role;
grant select, insert on public.brian_collector_runs to service_role;

drop trigger if exists brian_collector_runs_append_only on public.brian_collector_runs;
create trigger brian_collector_runs_append_only
before update or delete on public.brian_collector_runs
for each row execute function public.brian_reject_mutation();

create or replace function brian_private.schedule_global_eyes()
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  existing_job bigint;
  derivatives_job bigint;
  news_job bigint;
  fx_job bigint;
begin
  if not exists (select 1 from vault.decrypted_secrets where name = 'brian_project_url') then
    raise exception 'BRIAN_RUNTIME: Vault secret brian_project_url is missing';
  end if;
  if not exists (select 1 from vault.decrypted_secrets where name = 'brian_anon_jwt') then
    raise exception 'BRIAN_RUNTIME: Vault secret brian_anon_jwt is missing';
  end if;

  for existing_job in select jobid from cron.job where jobname in (
    'brian-derivatives-eye-5m','brian-news-eye-10m','brian-fx-eye-hourly'
  ) loop
    perform cron.unschedule(existing_job);
  end loop;

  derivatives_job := cron.schedule(
    'brian-derivatives-eye-5m', '3-59/5 * * * *',
    $cron$select net.http_post(
      url := (select decrypted_secret || '/functions/v1/brian-derivatives-eye' from vault.decrypted_secrets where name='brian_project_url' limit 1),
      headers := jsonb_build_object('Content-Type','application/json','Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1)),
      body := '{}'::jsonb
    );$cron$
  );

  news_job := cron.schedule(
    'brian-news-eye-10m', '4-59/10 * * * *',
    $cron$select net.http_post(
      url := (select decrypted_secret || '/functions/v1/brian-news-eye' from vault.decrypted_secrets where name='brian_project_url' limit 1),
      headers := jsonb_build_object('Content-Type','application/json','Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1)),
      body := '{}'::jsonb
    );$cron$
  );

  fx_job := cron.schedule(
    'brian-fx-eye-hourly', '7 * * * *',
    $cron$select net.http_post(
      url := (select decrypted_secret || '/functions/v1/brian-fx-eye' from vault.decrypted_secrets where name='brian_project_url' limit 1),
      headers := jsonb_build_object('Content-Type','application/json','Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1)),
      body := '{}'::jsonb
    );$cron$
  );

  return jsonb_build_object('derivatives_job',derivatives_job,'news_job',news_job,'fx_job',fx_job);
end;
$$;

revoke all on function brian_private.schedule_global_eyes() from public, anon, authenticated, service_role;
grant execute on function brian_private.schedule_global_eyes() to postgres;

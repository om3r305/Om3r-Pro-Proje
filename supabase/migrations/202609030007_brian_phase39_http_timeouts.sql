-- Phase 3.9 collector timeout hardening.
-- GDELT DNS/response latency can exceed pg_net's 5s default.

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
      body := '{}'::jsonb,
      timeout_milliseconds := 10000
    );$cron$
  );

  news_job := cron.schedule(
    'brian-news-eye-10m', '4-59/10 * * * *',
    $cron$select net.http_post(
      url := (select decrypted_secret || '/functions/v1/brian-news-eye' from vault.decrypted_secrets where name='brian_project_url' limit 1),
      headers := jsonb_build_object('Content-Type','application/json','Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1)),
      body := '{}'::jsonb,
      timeout_milliseconds := 20000
    );$cron$
  );

  fx_job := cron.schedule(
    'brian-fx-eye-hourly', '7 * * * *',
    $cron$select net.http_post(
      url := (select decrypted_secret || '/functions/v1/brian-fx-eye' from vault.decrypted_secrets where name='brian_project_url' limit 1),
      headers := jsonb_build_object('Content-Type','application/json','Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1)),
      body := '{}'::jsonb,
      timeout_milliseconds := 10000
    );$cron$
  );

  return jsonb_build_object('derivatives_job',derivatives_job,'news_job',news_job,'fx_job',fx_job);
end;
$$;

revoke all on function brian_private.schedule_global_eyes() from public, anon, authenticated, service_role;
grant execute on function brian_private.schedule_global_eyes() to postgres;

select brian_private.schedule_global_eyes();

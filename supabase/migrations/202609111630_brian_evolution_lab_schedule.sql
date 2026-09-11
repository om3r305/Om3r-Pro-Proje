-- Brian Evolution OS Layer 3 cloud laboratory schedule.
-- GitHub-only until explicit rollout. MAIN/ALPHA research only; DIP is untouched.

create or replace function brian_private.schedule_evolution_lab()
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  existing_job bigint;
  sandbox_job bigint;
  experiment_job bigint;
  council_job bigint;
begin
  if not exists (select 1 from vault.decrypted_secrets where name='brian_project_url') then raise exception 'BRIAN_EVOLUTION_LAB: brian_project_url missing'; end if;
  if not exists (select 1 from vault.decrypted_secrets where name='brian_anon_jwt') then raise exception 'BRIAN_EVOLUTION_LAB: brian_anon_jwt missing'; end if;
  if not exists (select 1 from vault.decrypted_secrets where name='brian_cron_key') then raise exception 'BRIAN_EVOLUTION_LAB: brian_cron_key missing'; end if;

  for existing_job in
    select jobid from cron.job where jobname in (
      'brian-evolution-sandbox-30m',
      'brian-evolution-experiment-runner-hourly',
      'brian-evolution-promotion-council-hourly'
    )
  loop perform cron.unschedule(existing_job); end loop;

  sandbox_job := cron.schedule(
    'brian-evolution-sandbox-30m','22,52 * * * *',
    $cron$
      select net.http_post(
        url := (select decrypted_secret || '/functions/v1/brian-evolution-sandbox' from vault.decrypted_secrets where name='brian_project_url' limit 1),
        headers := jsonb_build_object(
          'Content-Type','application/json',
          'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
          'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
          'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_cron_key' limit 1)
        ),
        body := '{"action":"plan"}'::jsonb,
        timeout_milliseconds := 55000
      );
    $cron$
  );

  experiment_job := cron.schedule(
    'brian-evolution-experiment-runner-hourly','27 * * * *',
    $cron$
      select net.http_post(
        url := (select decrypted_secret || '/functions/v1/brian-evolution-experiment-runner' from vault.decrypted_secrets where name='brian_project_url' limit 1),
        headers := jsonb_build_object(
          'Content-Type','application/json',
          'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
          'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
          'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_cron_key' limit 1)
        ),
        body := '{}'::jsonb,
        timeout_milliseconds := 55000
      );
    $cron$
  );

  council_job := cron.schedule(
    'brian-evolution-promotion-council-hourly','34 * * * *',
    $cron$
      select net.http_post(
        url := (select decrypted_secret || '/functions/v1/brian-evolution-promotion-council' from vault.decrypted_secrets where name='brian_project_url' limit 1),
        headers := jsonb_build_object(
          'Content-Type','application/json',
          'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
          'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
          'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_cron_key' limit 1)
        ),
        body := '{}'::jsonb,
        timeout_milliseconds := 55000
      );
    $cron$
  );

  return jsonb_build_object('sandbox_job',sandbox_job,'experiment_job',experiment_job,'council_job',council_job);
end;
$$;

revoke all on function brian_private.schedule_evolution_lab() from public, anon, authenticated, service_role;
grant execute on function brian_private.schedule_evolution_lab() to postgres;

do $$
begin
  if exists (select 1 from vault.decrypted_secrets where name='brian_project_url')
     and exists (select 1 from vault.decrypted_secrets where name='brian_anon_jwt')
     and exists (select 1 from vault.decrypted_secrets where name='brian_cron_key') then
    perform brian_private.schedule_evolution_lab();
  else
    raise notice 'Evolution lab not scheduled yet; provision Brian Vault runtime secrets at rollout';
  end if;
end;
$$;

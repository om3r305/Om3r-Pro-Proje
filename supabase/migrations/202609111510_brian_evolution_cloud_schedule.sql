-- Brian Evolution OS cloud schedule preparation.
-- GitHub-only until explicit deployment.
-- Requires Vault secret `brian_cron_key` to match brian_dashboard_auth.control-v3 cron key.
-- MAIN/ALPHA Evolution only. DIP cron/runtime is not read or changed.
-- IMPORTANT: this migration only installs scheduler functions; it never activates cron jobs.

create or replace function brian_private.schedule_evolution_os()
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  existing_job bigint;
  evolution_job bigint;
  world_job bigint;
begin
  if not exists (select 1 from vault.decrypted_secrets where name = 'brian_project_url') then
    raise exception 'BRIAN_EVOLUTION_RUNTIME: Vault secret brian_project_url is missing';
  end if;
  if not exists (select 1 from vault.decrypted_secrets where name = 'brian_anon_jwt') then
    raise exception 'BRIAN_EVOLUTION_RUNTIME: Vault secret brian_anon_jwt is missing';
  end if;
  if not exists (select 1 from vault.decrypted_secrets where name = 'brian_cron_key') then
    raise exception 'BRIAN_EVOLUTION_RUNTIME: Vault secret brian_cron_key is missing';
  end if;

  for existing_job in
    select jobid from cron.job where jobname in (
      'brian-evolution-orchestrator-10m',
      'brian-world-brain-5m'
    )
  loop
    perform cron.unschedule(existing_job);
  end loop;

  evolution_job := cron.schedule(
    'brian-evolution-orchestrator-10m',
    '1-59/10 * * * *',
    $cron$
      select net.http_post(
        url := (
          select decrypted_secret || '/functions/v1/brian-evolution-orchestrator'
          from vault.decrypted_secrets where name = 'brian_project_url' limit 1
        ),
        headers := jsonb_build_object(
          'Content-Type','application/json',
          'Authorization','Bearer ' || (
            select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1
          ),
          'apikey',(
            select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1
          ),
          'x-brian-cron-key',(
            select decrypted_secret from vault.decrypted_secrets where name='brian_cron_key' limit 1
          )
        ),
        body := '{}'::jsonb,
        timeout_milliseconds := 55000
      );
    $cron$
  );

  world_job := cron.schedule(
    'brian-world-brain-5m',
    '6-59/5 * * * *',
    $cron$
      select net.http_post(
        url := (
          select decrypted_secret || '/functions/v1/brian-world-brain'
          from vault.decrypted_secrets where name = 'brian_project_url' limit 1
        ),
        headers := jsonb_build_object(
          'Content-Type','application/json',
          'Authorization','Bearer ' || (
            select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1
          ),
          'apikey',(
            select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1
          ),
          'x-brian-cron-key',(
            select decrypted_secret from vault.decrypted_secrets where name='brian_cron_key' limit 1
          )
        ),
        body := '{}'::jsonb,
        timeout_milliseconds := 55000
      );
    $cron$
  );

  return jsonb_build_object(
    'evolution_job', evolution_job,
    'world_brain_job', world_job,
    'browser_required', false,
    'shadow_only', true,
    'live_execution', false
  );
end;
$$;

revoke all on function brian_private.schedule_evolution_os() from public, anon, authenticated, service_role;
grant execute on function brian_private.schedule_evolution_os() to postgres;

-- Deliberately no DO block here. Existing Brian Vault secrets may already be present in
-- production; auto-scheduling from a migration would violate the explicit activation gate.

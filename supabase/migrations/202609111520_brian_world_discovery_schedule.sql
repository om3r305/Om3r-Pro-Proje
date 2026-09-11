-- Brian Evolution OS broad World Discovery Eye schedule.
-- GitHub-only until explicit deployment. DIP schedules are untouched.
-- Requires brian_project_url, brian_anon_jwt and brian_cron_key in Vault.

create or replace function brian_private.schedule_world_discovery_eye()
returns bigint
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  existing_job bigint;
  scheduled_job bigint;
begin
  if not exists (select 1 from vault.decrypted_secrets where name='brian_project_url') then
    raise exception 'BRIAN_WORLD_DISCOVERY: brian_project_url missing';
  end if;
  if not exists (select 1 from vault.decrypted_secrets where name='brian_anon_jwt') then
    raise exception 'BRIAN_WORLD_DISCOVERY: brian_anon_jwt missing';
  end if;
  if not exists (select 1 from vault.decrypted_secrets where name='brian_cron_key') then
    raise exception 'BRIAN_WORLD_DISCOVERY: brian_cron_key missing';
  end if;

  for existing_job in select jobid from cron.job where jobname='brian-world-discovery-eye-10m' loop
    perform cron.unschedule(existing_job);
  end loop;

  scheduled_job := cron.schedule(
    'brian-world-discovery-eye-10m',
    '*/10 * * * *',
    $cron$
      select net.http_post(
        url := (
          select decrypted_secret || '/functions/v1/brian-world-discovery-eye'
          from vault.decrypted_secrets where name='brian_project_url' limit 1
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
  return scheduled_job;
end;
$$;

revoke all on function brian_private.schedule_world_discovery_eye() from public, anon, authenticated, service_role;
grant execute on function brian_private.schedule_world_discovery_eye() to postgres;

do $$
begin
  if exists (select 1 from vault.decrypted_secrets where name='brian_project_url')
     and exists (select 1 from vault.decrypted_secrets where name='brian_anon_jwt')
     and exists (select 1 from vault.decrypted_secrets where name='brian_cron_key') then
    perform brian_private.schedule_world_discovery_eye();
  else
    raise notice 'World Discovery Eye not scheduled yet; provision Brian Vault runtime secrets at rollout';
  end if;
end;
$$;

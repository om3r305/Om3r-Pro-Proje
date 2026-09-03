-- Brian Phase 3.8 crypto sensor mesh scheduler.
-- Uses existing Vault secret names provisioned out-of-band.

create or replace function brian_private.schedule_sensor_mesh()
returns bigint
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  existing_job bigint;
  scheduled_job bigint;
begin
  if not exists (select 1 from vault.decrypted_secrets where name = 'brian_project_url') then
    raise exception 'BRIAN_RUNTIME: Vault secret brian_project_url is missing';
  end if;
  if not exists (select 1 from vault.decrypted_secrets where name = 'brian_anon_jwt') then
    raise exception 'BRIAN_RUNTIME: Vault secret brian_anon_jwt is missing';
  end if;

  for existing_job in select jobid from cron.job where jobname = 'brian-sensor-mesh-5m' loop
    perform cron.unschedule(existing_job);
  end loop;

  scheduled_job := cron.schedule(
    'brian-sensor-mesh-5m',
    '2-59/5 * * * *',
    $cron$
      select net.http_post(
        url := (
          select decrypted_secret || '/functions/v1/brian-sensor-mesh'
          from vault.decrypted_secrets where name = 'brian_project_url' limit 1
        ),
        headers := jsonb_build_object(
          'Content-Type','application/json',
          'Authorization','Bearer ' || (
            select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1
          ),
          'apikey',(
            select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1
          )
        ),
        body := '{}'::jsonb
      );
    $cron$
  );
  return scheduled_job;
end;
$$;

revoke all on function brian_private.schedule_sensor_mesh() from public;
revoke all on function brian_private.schedule_sensor_mesh() from anon, authenticated, service_role;
grant execute on function brian_private.schedule_sensor_mesh() to postgres;

do $$
begin
  if exists (select 1 from vault.decrypted_secrets where name = 'brian_project_url')
     and exists (select 1 from vault.decrypted_secrets where name = 'brian_anon_jwt') then
    perform brian_private.schedule_sensor_mesh();
  else
    raise notice 'Brian sensor mesh cron not scheduled: provision existing Brian Vault runtime secrets first';
  end if;
end;
$$;

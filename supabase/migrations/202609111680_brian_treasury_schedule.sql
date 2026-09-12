-- Brian Evolution OS Layer 5 Treasury schedule.
-- GitHub-only until rollout. SHADOW ONLY. DIP remains untouched.
-- IMPORTANT: this migration only installs the scheduler function; it never activates cron.

create or replace function brian_private.schedule_evolution_treasury()
returns bigint
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  existing_job bigint;
  scheduled_job bigint;
begin
  if not exists (select 1 from vault.decrypted_secrets where name='brian_project_url') then raise exception 'BRIAN_TREASURY: brian_project_url missing'; end if;
  if not exists (select 1 from vault.decrypted_secrets where name='brian_anon_jwt') then raise exception 'BRIAN_TREASURY: brian_anon_jwt missing'; end if;
  if not exists (select 1 from vault.decrypted_secrets where name='brian_cron_key') then raise exception 'BRIAN_TREASURY: brian_cron_key missing'; end if;

  for existing_job in select jobid from cron.job where jobname='brian-evolution-treasury-1m' loop
    perform cron.unschedule(existing_job);
  end loop;

  scheduled_job := cron.schedule(
    'brian-evolution-treasury-1m',
    '* * * * *',
    $cron$
      select net.http_post(
        url := (select decrypted_secret || '/functions/v1/brian-evolution-treasury' from vault.decrypted_secrets where name='brian_project_url' limit 1),
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
  return scheduled_job;
end;
$$;

revoke all on function brian_private.schedule_evolution_treasury() from public, anon, authenticated, service_role;
grant execute on function brian_private.schedule_evolution_treasury() to postgres;

-- Deliberately no auto-schedule DO block. Activation is a separate explicit rollout action.

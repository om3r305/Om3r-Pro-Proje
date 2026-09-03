-- Brian prospective shadow collector runtime.
-- Required Vault secret names (values are provisioned out-of-band):
--   brian_project_url
--   brian_anon_jwt

create extension if not exists pg_net with schema extensions;
create extension if not exists pg_cron with schema extensions;

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values (
  'brian-intelligence-raw',
  'brian-intelligence-raw',
  false,
  20971520,
  array['application/json','application/gzip']::text[]
)
on conflict (id) do update
set public = false,
    file_size_limit = excluded.file_size_limit,
    allowed_mime_types = excluded.allowed_mime_types;

create schema if not exists brian_private;
revoke all on schema brian_private from public;
revoke all on schema brian_private from anon, authenticated, service_role;
grant usage on schema brian_private to postgres;

create or replace function brian_private.schedule_universe_collector()
returns bigint
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  existing_job bigint;
  scheduled_job bigint;
begin
  if not exists (
    select 1 from vault.decrypted_secrets where name = 'brian_project_url'
  ) then
    raise exception 'BRIAN_RUNTIME: Vault secret brian_project_url is missing';
  end if;
  if not exists (
    select 1 from vault.decrypted_secrets where name = 'brian_anon_jwt'
  ) then
    raise exception 'BRIAN_RUNTIME: Vault secret brian_anon_jwt is missing';
  end if;

  for existing_job in
    select jobid from cron.job where jobname = 'brian-universe-collector-15m'
  loop
    perform cron.unschedule(existing_job);
  end loop;

  scheduled_job := cron.schedule(
    'brian-universe-collector-15m',
    '*/15 * * * *',
    $cron$
      select net.http_post(
        url := (
          select decrypted_secret || '/functions/v1/brian-universe-collector'
          from vault.decrypted_secrets
          where name = 'brian_project_url'
          limit 1
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

revoke all on function brian_private.schedule_universe_collector() from public;
revoke all on function brian_private.schedule_universe_collector() from anon, authenticated, service_role;
grant execute on function brian_private.schedule_universe_collector() to postgres;

do $$
begin
  if exists (select 1 from vault.decrypted_secrets where name = 'brian_project_url')
     and exists (select 1 from vault.decrypted_secrets where name = 'brian_anon_jwt') then
    perform brian_private.schedule_universe_collector();
  else
    raise notice 'Brian collector cron not scheduled: provision Vault secrets brian_project_url and brian_anon_jwt, then call brian_private.schedule_universe_collector()';
  end if;
end;
$$;

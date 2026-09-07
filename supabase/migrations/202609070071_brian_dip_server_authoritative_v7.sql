-- Brian DIP V7: server-authoritative SHADOW runtime.
-- Browser/PWA becomes a viewer only after the first authoritative server snapshot for a session.
-- Historical rows stay append-only. No live exchange execution is introduced.

create or replace function public.brian_reject_browser_dip_write_after_server_takeover()
returns trigger
language plpgsql
security definer
set search_path = pg_catalog, public
as $function$
declare
  v_server_owned boolean;
begin
  select exists(
    select 1
    from public.brian_dip_snapshots s
    where s.session_id = new.session_id
      and coalesce(s.state #>> '{serverRuntime,authoritative}', 'false') = 'true'
  ) into v_server_owned;

  if not v_server_owned then
    return new;
  end if;

  if tg_table_name = 'brian_dip_snapshots' then
    if coalesce(new.state #>> '{serverRuntime,authoritative}', 'false') <> 'true' then
      raise exception 'BRIAN_DIP_SERVER_AUTHORITATIVE: browser snapshot rejected';
    end if;
  elsif tg_table_name = 'brian_dip_events' then
    if coalesce(new.metadata ->> 'server_v7', 'false') <> 'true' then
      raise exception 'BRIAN_DIP_SERVER_AUTHORITATIVE: browser event rejected';
    end if;
  end if;

  return new;
end;
$function$;

revoke all on function public.brian_reject_browser_dip_write_after_server_takeover() from public, anon, authenticated;
grant execute on function public.brian_reject_browser_dip_write_after_server_takeover() to service_role;

drop trigger if exists brian_dip_server_authoritative_snapshot_fence on public.brian_dip_snapshots;
create trigger brian_dip_server_authoritative_snapshot_fence
  before insert on public.brian_dip_snapshots
  for each row execute function public.brian_reject_browser_dip_write_after_server_takeover();

drop trigger if exists brian_dip_server_authoritative_event_fence on public.brian_dip_events;
create trigger brian_dip_server_authoritative_event_fence
  before insert on public.brian_dip_events
  for each row execute function public.brian_reject_browser_dip_write_after_server_takeover();

comment on function public.brian_reject_browser_dip_write_after_server_takeover() is
  'Fails closed against browser DIP event/snapshot writes after Brian V7 server runtime has taken ownership of a session.';

do $brian$
declare
  v_jobid bigint;
begin
  for v_jobid in
    select jobid from cron.job where jobname = 'brian-dip-shadow-worker-v7-1m'
  loop
    perform cron.unschedule(v_jobid);
  end loop;
end
$brian$;

select cron.schedule(
  'brian-dip-shadow-worker-v7-1m',
  '* * * * *',
  $$
  select net.http_post(
    url := (
      select decrypted_secret || '/functions/v1/brian-dip-shadow-worker'
      from vault.decrypted_secrets
      where name = 'brian_project_url'
      limit 1
    ),
    headers := jsonb_build_object(
      'Content-Type', 'application/json',
      'Authorization', 'Bearer ' || (
        select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1
      ),
      'apikey', (
        select decrypted_secret from vault.decrypted_secrets where name = 'brian_anon_jwt' limit 1
      ),
      'x-brian-cron-key', (
        select decrypted_secret from vault.decrypted_secrets where name = 'brian_dashboard_cron_key' limit 1
      )
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 50000
  );
  $$
);

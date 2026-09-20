-- APPLY TO: brian-realtime only.
-- One lightweight pg_cron entry drives the Core scheduler Edge Function.

create or replace function brian_private.invoke_realtime_core_scheduler_v1()
returns bigint
language plpgsql
security definer
set search_path to 'public','extensions','vault','net','pg_temp'
as $$
declare
  v_key text;
  v_request_id bigint;
begin
  select decrypted_secret into v_key
  from vault.decrypted_secrets
  where name='brian_realtime_internal_key'
  limit 1;

  if v_key is null or length(v_key)<20 then
    raise exception 'BRIAN_REALTIME_INTERNAL_KEY_MISSING';
  end if;

  select net.http_post(
    url := 'https://dliediwlldojkfjzlznm.supabase.co/functions/v1/brian-realtime-core-scheduler',
    headers := jsonb_build_object(
      'content-type','application/json',
      'x-brian-internal-key',v_key
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 45000
  ) into v_request_id;

  return v_request_id;
end;
$$;

do $$
declare j bigint;
begin
  select jobid into j
  from cron.job
  where jobname='brian-realtime-core-scheduler-1m'
  limit 1;

  if j is null then
    perform cron.schedule(
      'brian-realtime-core-scheduler-1m',
      '* * * * *',
      'select brian_private.invoke_realtime_core_scheduler_v1();'
    );
  else
    perform cron.alter_job(
      j,
      schedule := '* * * * *',
      command := 'select brian_private.invoke_realtime_core_scheduler_v1();',
      active := true
    );
  end if;
end $$;

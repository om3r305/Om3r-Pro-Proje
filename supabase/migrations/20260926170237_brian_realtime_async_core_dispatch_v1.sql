-- APPLY TO: brian-realtime only.
-- Queue Core control-plane work through pg_net so the 1m realtime scheduler
-- does not block on slow Core execution. SHADOW/control-plane only.

create or replace function public.brian_realtime_enqueue_core_action_v1(p_action text)
returns jsonb
language plpgsql
security definer
set search_path to 'pg_catalog','public','net','vault','pg_temp'
as $function$
declare
  v_action text := lower(trim(coalesce(p_action,'')));
  v_key text;
  v_url text;
  v_request_id bigint;
  v_existing_id bigint;
  v_body jsonb;
begin
  if not (v_action = any(array[
    'alpha_sync','world','treasury','discovery',
    'official_primary','source_observer','source_registry','meeting_sync',
    'recovery','watchdog','dip','multiasset',
    'universe_heartbeat','sensor_heartbeat','intrabar_heartbeat',
    'derivatives_heartbeat','behavior_heartbeat','direct_wire_heartbeat',
    'fx_heartbeat','missed_auditor'
  ]::text[])) then
    raise exception 'unsupported realtime core action: %', p_action using errcode='22023';
  end if;

  select decrypted_secret into v_key
  from vault.decrypted_secrets
  where name='brian_realtime_internal_key'
  limit 1;

  if v_key is null or length(v_key) < 20 then
    raise exception 'BRIAN_REALTIME_INTERNAL_KEY_MISSING';
  end if;

  if v_action='missed_auditor' then
    v_url := 'https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-missed-opportunity-auditor-v3';
  else
    v_url := 'https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-core-scheduler-bridge';
  end if;

  v_body := jsonb_build_object('action',v_action);

  select q.id into v_existing_id
  from net.http_request_queue q
  where q.method='POST'
    and q.url=v_url
    and (
      v_action='missed_auditor'
      or (
        q.body is not null
        and (convert_from(q.body,'UTF8')::jsonb ->> 'action') = v_action
      )
    )
  order by q.id desc
  limit 1;

  if v_existing_id is not null then
    return jsonb_build_object(
      'status','SKIPPED_BUSY',
      'action',v_action,
      'existing_request_id',v_existing_id,
      'dispatch_mode','PG_NET_ASYNC'
    );
  end if;

  select net.http_post(
    url := v_url,
    headers := jsonb_build_object(
      'content-type','application/json',
      'x-brian-internal-key',v_key
    ),
    body := v_body,
    timeout_milliseconds := 45000
  ) into v_request_id;

  return jsonb_build_object(
    'status','QUEUED',
    'action',v_action,
    'request_id',v_request_id,
    'dispatch_mode','PG_NET_ASYNC'
  );
end;
$function$;

revoke all on function public.brian_realtime_enqueue_core_action_v1(text)
from public, anon, authenticated;

grant execute on function public.brian_realtime_enqueue_core_action_v1(text)
to service_role;

comment on function public.brian_realtime_enqueue_core_action_v1(text) is
'Service-role-only async dispatcher from brian-realtime to Brian Core scheduler/auditor. Queues pg_net work so the 1m realtime scheduler does not block on Core execution latency. Shadow control-plane only.';

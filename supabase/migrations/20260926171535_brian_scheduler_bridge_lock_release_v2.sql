-- APPLY TO: brian-market-intelligence only.
-- Keep the shared scheduler-state row lock short by touching it only after action work.

create or replace function public.brian_scheduler_bridge_v1(p_action text)
returns jsonb
language plpgsql
security definer
set search_path to 'pg_catalog','public','brian_private','net','vault'
as $function$
declare
  v_action text := lower(trim(coalesce(p_action,'')));
  v_request_id bigint;
  v_result jsonb;
begin
  if v_action='alpha_sync' then
    v_result := brian_private.sync_realtime_alpha_logged_v2();

  elsif v_action='world' then
    v_result := jsonb_build_object('status','QUEUED','action',v_action,'result',brian_private.launch_core_lane('world'));

  elsif v_action='treasury' then
    v_result := jsonb_build_object('status','QUEUED','action',v_action,'result',brian_private.launch_core_lane('treasury'));

  elsif v_action='discovery' then
    v_result := jsonb_build_object('status','QUEUED','action',v_action,'result',brian_private.launch_core_lane('discovery'));

  elsif v_action='official_primary' then
    perform brian_private.enqueue_aux_service('official_primary');
    v_result := jsonb_build_object('status','QUEUED','action',v_action);

  elsif v_action='source_observer' then
    perform brian_private.enqueue_aux_service('source_observer');
    v_result := jsonb_build_object('status','QUEUED','action',v_action);

  elsif v_action='source_registry' then
    perform brian_private.enqueue_aux_service('source_registry');
    v_result := jsonb_build_object('status','QUEUED','action',v_action);

  elsif v_action='meeting_sync' then
    perform public.brian_sync_meeting_shadow_ledger();
    v_result := jsonb_build_object('status','SUCCESS','action',v_action);

  elsif v_action='recovery' then
    v_result := coalesce(brian_private.dispatch_stale_core(),'{}'::jsonb)
      || jsonb_build_object('scheduler','REALTIME_BRIDGE');

  elsif v_action='watchdog' then
    v_result := coalesce(brian_private.pgnet_watchdog(),'{}'::jsonb)
      || jsonb_build_object('scheduler','REALTIME_BRIDGE');

  elsif v_action='dip' then
    select net.http_post(
      url := (select decrypted_secret || '/functions/v1/brian-dip-multiasset-worker-v860'
              from vault.decrypted_secrets where name='brian_project_url' limit 1),
      headers := jsonb_build_object(
        'Content-Type','application/json',
        'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
        'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
        'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_dashboard_cron_key' limit 1)
      ),
      body := '{}'::jsonb,
      timeout_milliseconds := 15000
    )
    into v_request_id
    where not exists (
      select 1 from net.http_request_queue q
      where q.url like '%/functions/v1/brian-dip-multiasset-worker-v860'
    );
    v_result := jsonb_build_object(
      'status',case when v_request_id is null then 'SKIPPED_BUSY' else 'QUEUED' end,
      'action',v_action,'request_id',v_request_id
    );

  elsif v_action='multiasset' then
    select net.http_post(
      url := (select decrypted_secret || '/functions/v1/brian-multiasset-opportunity-engine'
              from vault.decrypted_secrets where name='brian_project_url' limit 1),
      headers := jsonb_build_object(
        'Content-Type','application/json',
        'Authorization','Bearer ' || (select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
        'apikey',(select decrypted_secret from vault.decrypted_secrets where name='brian_anon_jwt' limit 1),
        'x-brian-cron-key',(select decrypted_secret from vault.decrypted_secrets where name='brian_cron_key' limit 1)
      ),
      body := '{}'::jsonb,
      timeout_milliseconds := 30000
    )
    into v_request_id
    where not exists (
      select 1 from net.http_request_queue q
      where q.url like '%/functions/v1/brian-multiasset-opportunity-engine'
    );
    v_result := jsonb_build_object(
      'status',case when v_request_id is null then 'SKIPPED_BUSY' else 'QUEUED' end,
      'action',v_action,'request_id',v_request_id
    );

  else
    raise exception 'unsupported scheduler action: %', p_action using errcode='22023';
  end if;

  insert into public.brian_scheduler_state(
    scheduler_id,owner_project,last_seen_at,last_action,status,metadata,updated_at
  )
  values(
    'brian-realtime-core-scheduler','brian-realtime',now(),v_action,'ONLINE',
    jsonb_build_object('version','v2-async-core-launch','target_project','brian-market-intelligence'),now()
  )
  on conflict(scheduler_id) do update
  set owner_project=excluded.owner_project,
      last_seen_at=excluded.last_seen_at,
      last_action=excluded.last_action,
      status='ONLINE',
      metadata=excluded.metadata,
      updated_at=excluded.updated_at;

  return v_result;
end;
$function$;

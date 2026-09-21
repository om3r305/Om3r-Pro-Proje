-- Target: brian-market-intelligence. Standby only; never marks Realtime alive.
create or replace function brian_private.frontier_standby_tick()
returns jsonb language plpgsql security definer
set search_path=pg_catalog,public,brian_private as $$
declare v_min integer:=extract(minute from now())::integer; v_action text; v_result jsonb;
begin
  if not exists(select 1 from public.brian_evolution_runtime_config
                where config_key='brian_system_enabled' and lower(config_value)='true') then
    return jsonb_build_object('status','OPERATOR_STOPPED');
  end if;
  if exists(select 1 from public.brian_scheduler_state
            where scheduler_id='brian-realtime-core-scheduler' and status='ONLINE'
              and last_seen_at>now()-interval '3 minutes') then
    return jsonb_build_object('status','STANDBY');
  end if;
  if not pg_try_advisory_xact_lock(hashtextextended('frontier-standby-tick',0)) then
    return jsonb_build_object('status','BUSY');
  end if;
  -- One bounded action per minute, using existing Core lane leases.
  case v_min%5
    when 0 then v_action:='alpha_sync'; v_result:=brian_private.sync_realtime_alpha_logged_v2();
    when 1 then v_action:='world'; v_result:=to_jsonb(brian_private.launch_core_lane('world'));
    when 2 then v_action:='treasury'; v_result:=to_jsonb(brian_private.launch_core_lane('treasury'));
    when 3 then v_action:='discovery'; v_result:=to_jsonb(brian_private.launch_core_lane('discovery'));
    else return jsonb_build_object('status','STANDBY_RECOVERY_COOLDOWN');
  end case;
  insert into public.brian_scheduler_state(scheduler_id,owner_project,last_seen_at,last_action,status,metadata,updated_at)
  values('brian-core-standby','brian-market-intelligence',now(),v_action,'FALLBACK',jsonb_build_object('result',v_result),now())
  on conflict(scheduler_id) do update set last_seen_at=excluded.last_seen_at,last_action=excluded.last_action,
    status=excluded.status,metadata=excluded.metadata,updated_at=excluded.updated_at;
  return jsonb_build_object('status','FALLBACK','action',v_action,'result',v_result);
end $$;
revoke all on function brian_private.frontier_standby_tick() from public,anon,authenticated;
select cron.schedule('brian-frontier-standby-1m','* * * * *','select brian_private.frontier_standby_tick();');

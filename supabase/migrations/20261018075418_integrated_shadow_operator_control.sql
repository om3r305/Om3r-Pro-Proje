-- Explicit Core ownership; Realtime data schedulers obey the same operator flag.
create or replace function public.brian_set_system_enabled(p_enabled boolean)
returns jsonb language plpgsql security definer
set search_path=pg_catalog,public,cron
as $fn$
declare
  j record; changed integer:=0; target_active boolean; heartbeat_ok boolean:=false;
  core_jobs text[]:=array[
    'brian-evolution-treasury-1m','brian-evolution-orchestrator-10m',
    'brian-evolution-researcher-30m','brian-evolution-sandbox-30m',
    'brian-evolution-ocean-5m','brian-intel-label-v2-5m',
    'brian-missed-opportunity-auditor-v3-5m','brian-trend-shadow-daily'
  ];
begin
  perform pg_advisory_xact_lock(hashtextextended('brian-integrated-system-control',0));
  for j in select jobid,jobname,active from cron.job
    where jobname like 'brian-%' and jobname not like 'brian-dip-%' order by jobid
  loop
    target_active:=p_enabled and j.jobname=any(core_jobs);
    if j.active is distinct from target_active then
      perform cron.alter_job(j.jobid,active:=target_active);
      changed:=changed+1;
    end if;
  end loop;
  insert into public.brian_evolution_runtime_config(config_key,config_value,updated_at)
  values ('brian_system_enabled',case when p_enabled then 'true' else 'false' end,clock_timestamp()),
    ('brian_operating_mode','SHADOW_INTEGRATED',clock_timestamp())
  on conflict(config_key) do update set config_value=excluded.config_value,updated_at=excluded.updated_at;
  begin
    perform brian_private.refresh_frontier_heartbeat_cache();
    heartbeat_ok:=true;
  exception when others then
    raise warning 'Control applied; heartbeat refresh failed: %',sqlerrm;
  end;
  return jsonb_build_object('status',case when p_enabled then 'RUNNING' else 'STOPPED' end,
    'operating_mode','SHADOW_INTEGRATED','changed_jobs',changed,
    'core_jobs',to_jsonb(core_jobs),'heartbeat_refreshed',heartbeat_ok,
    'realtime_obeys_operator_flag',true,'dip_touched',false,'shadow_only',true,'live_execution',false);
end;
$fn$;
revoke all on function public.brian_set_system_enabled(boolean) from public,anon,authenticated;
grant execute on function public.brian_set_system_enabled(boolean) to service_role;
insert into public.brian_system_job_registry(job_name,component,operator_managed) values
 ('brian-intel-label-v2-5m','INTELLIGENCE',true),
 ('brian-trend-shadow-daily','SHADOW_BENCHMARK',true),
 ('brian-evolution-treasury-1m','TREASURY',true)
on conflict(job_name) do update set component=excluded.component,operator_managed=true;
select cron.alter_job(job_id:=(select jobid from cron.job where jobname='brian-evolution-treasury-1m'),
 schedule:='2-59/5 * * * *');

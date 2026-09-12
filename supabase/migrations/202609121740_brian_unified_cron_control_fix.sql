-- Hotfix unified Brian start/stop after production pg_cron permission validation.
-- Direct UPDATE on cron.job is not permitted. Use the supported cron.alter_job API.
-- Registry remains an explicit non-DIP allowlist. SHADOW ONLY.

create or replace function public.brian_set_system_enabled(p_enabled boolean)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public, cron
as $$
declare
  v_job record;
  v_changed integer := 0;
  v_total integer := 0;
begin
  for v_job in
    select j.jobid, j.jobname, j.active
    from public.brian_system_job_registry r
    join cron.job j on j.jobname=r.job_name
    where r.operator_managed
      and j.jobname not like 'brian-dip-%'
    order by j.jobid
  loop
    if v_job.active is distinct from p_enabled then
      perform cron.alter_job(v_job.jobid, null, null, null, null, p_enabled);
      v_changed := v_changed + 1;
    end if;
  end loop;

  select count(*)::integer into v_total
  from public.brian_system_job_registry
  where operator_managed;

  insert into public.brian_evolution_runtime_config(config_key,config_value,updated_at)
  values('brian_system_enabled',case when p_enabled then 'true' else 'false' end,clock_timestamp())
  on conflict(config_key) do update
  set config_value=excluded.config_value,updated_at=excluded.updated_at;

  return jsonb_build_object(
    'status',case when p_enabled then 'RUNNING' else 'STOPPED' end,
    'changed_jobs',v_changed,
    'managed_jobs_total',v_total,
    'dip_touched',false,
    'shadow_only',true,
    'live_execution',false
  );
end;
$$;

revoke all on function public.brian_set_system_enabled(boolean) from public, anon, authenticated;
grant execute on function public.brian_set_system_enabled(boolean) to service_role;

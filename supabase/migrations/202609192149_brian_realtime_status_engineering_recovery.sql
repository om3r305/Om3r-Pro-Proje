-- Brian Core/Realtime ownership cleanup and Engineering stale-worker recovery.
-- Applied to production first on 2026-09-19; kept idempotent for repository parity.

update public.brian_system_job_registry
set operator_managed = false
where job_name in (
  'brian-universe-collector-15m',
  'brian-sensor-mesh-5m',
  'brian-derivatives-eye-5m',
  'brian-fx-eye-hourly',
  'brian-intrabar-eye-1m'
);

do $$
declare
  target_jobid bigint;
begin
  select jobid into target_jobid
  from cron.job
  where jobname = 'brian-engineering-stale-recovery-10m'
  order by jobid
  limit 1;

  if target_jobid is not null then
    perform cron.alter_job(target_jobid, active := true);
  end if;
end $$;

select brian_private.reap_expired_engineering_runs();

-- Re-enable the stale-core dispatcher at a staggered cadence after load shedding.
-- This is the queue drain/fallback path used by auxiliary recovery services.
do $$
begin
  perform cron.alter_job(j.jobid, schedule := '6,16,26,36,46,56 * * * *', active := true)
  from cron.job j
  where j.jobname='brian-core-recovery-dispatcher-v1';
end $$;

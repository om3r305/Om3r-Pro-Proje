-- Startup timeouts continued after pausing 82/97/19 (13 failed / 1 ok in 12 min at 07:35 UTC).
-- Return CORE to its last known-stable job set (before 2026-10-21) by also pausing the other jobs
-- added that day: alpha edge challenger (47), alpha horizon challenger (98) and the pm-calibration
-- shadow collector. Nothing is deleted; re-enable one by one after the IO budget recovers.
do $$
declare j record;
begin
  for j in select jobid from cron.job where jobname in ('brian-evolution-alpha-edge-2m', 'brian-evolution-alpha-horizon-5m', 'brian-pm-calibration-shadow-hourly') loop
    perform cron.alter_job(j.jobid, active := false);
  end loop;
end $$;

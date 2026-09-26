-- Further stagger the reliability lane to avoid same-minute database pressure.
do $$
begin
  perform cron.alter_job(j.jobid, schedule := '32 * * * *', active := true)
  from cron.job j
  where j.jobname='brian-alpha-reliability-feature-freeze-1m';

  perform cron.alter_job(j.jobid, schedule := '4,14,24,34,44,54 * * * *', active := true)
  from cron.job j
  where j.jobname='brian-sensor-reliability-calibration-5m';
end $$;

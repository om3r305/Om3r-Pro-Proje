-- Move the hourly shadow reliability refresh away from the congested :12 lane.
-- Evidence semantics and readiness thresholds are unchanged.
do $body$
begin
  perform cron.alter_job(
    j.jobid,
    schedule := '17 * * * *',
    command := $cmd$select set_config('statement_timeout','45000',false);
select public.brian_refresh_sensor_reliability_shadow(now(), interval '24 hours');$cmd$,
    active := true
  )
  from cron.job j
  where j.jobname='brian-sensor-reliability-shadow-hourly';
end
$body$;

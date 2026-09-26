-- Bound reliability maintenance work so it survives the small compute/pool budget.
-- This changes maintenance cadence and batch size only; evidence semantics,
-- thresholds, shadow-only boundaries, and execution behavior remain unchanged.
do $body$
begin
  perform cron.alter_job(
    j.jobid,
    schedule := '2,22,42 * * * *',
    command := $cmd$select set_config('statement_timeout','45000',false);
select public.brian_capture_alpha_reliability_shadow_features(now(), interval '15 minutes', 150);$cmd$,
    active := true
  )
  from cron.job j
  where j.jobname='brian-alpha-reliability-feature-freeze-1m';

  perform cron.alter_job(
    j.jobid,
    schedule := '4,24,44 * * * *',
    command := $cmd$select set_config('statement_timeout','45000',false);
select public.brian_resolve_sensor_reliability_prospective_calibration(now(), interval '12 hours', 500);$cmd$,
    active := true
  )
  from cron.job j
  where j.jobname='brian-sensor-reliability-calibration-5m';
end
$body$;

-- Brian Disk IO budget protection. MAIN/ALPHA support jobs only; DIP untouched.
-- Keep the live decision/sensor loop running while reducing expensive analytical churn.
-- Confirmed high-IO workloads are moved to lower cadences using pg_cron's supported API.

do $$
declare
  v record;
begin
  for v in
    select jobid, jobname
    from cron.job
    where jobname in (
      'brian-sensor-reliability-calibration-5m',
      'brian-alpha-reliability-feature-freeze-1m',
      'brian-alpha-calibration-challenger-v1-5m',
      'brian-missed-opportunity-auditor-v3-5m',
      'brian-compact-retention-hourly'
    )
  loop
    case v.jobname
      when 'brian-sensor-reliability-calibration-5m' then
        perform cron.alter_job(v.jobid, '39 * * * *', null, null, null, null);
      when 'brian-alpha-reliability-feature-freeze-1m' then
        perform cron.alter_job(v.jobid, '3,18,33,48 * * * *', null, null, null, null);
      when 'brian-alpha-calibration-challenger-v1-5m' then
        perform cron.alter_job(v.jobid, '49 * * * *', null, null, null, null);
      when 'brian-missed-opportunity-auditor-v3-5m' then
        perform cron.alter_job(v.jobid, '2,17,32,47 * * * *', null, null, null, null);
      when 'brian-compact-retention-hourly' then
        perform cron.alter_job(v.jobid, '17 */2 * * *', null, null, null, null);
    end case;
  end loop;
end $$;

-- Core real-time Brian loops intentionally remain unchanged:
-- ALPHA compiler */2m, intrabar */2m, sensor mesh every 10m, Treasury every 1m.

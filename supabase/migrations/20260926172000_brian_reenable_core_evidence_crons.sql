-- Re-enable only the critical prospective-evidence lane after market-intelligence
-- load shedding. Keep the heavier research/reporting jobs disabled until the DB
-- remains stable. Schedules are intentionally staggered.
do $$
declare
  r record;
begin
  for r in
    select * from (values
      ('brian-sensor-reliability-shadow-hourly','12 * * * *'),
      ('brian-alpha-reliability-feature-freeze-1m','33 * * * *'),
      ('brian-sensor-reliability-calibration-5m','13,23,33,43,53 * * * *'),
      ('brian-referenced-sensor-evidence-sync-2m','1,11,21,31,41,51 * * * *'),
      ('brian-missed-opportunity-auditor-v3-5m','7,17,27,37,47,57 * * * *')
    ) as x(jobname,schedule)
  loop
    perform cron.alter_job(j.jobid, schedule := r.schedule, active := true)
    from cron.job j
    where j.jobname = r.jobname;
  end loop;
end $$;

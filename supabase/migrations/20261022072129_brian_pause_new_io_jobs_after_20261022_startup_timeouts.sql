-- Applied twice on CORE (20261022072129 and a retry 20261022072350 after a dropped connection); idempotent.
-- 2026-10-22 04:13 UTC onward: pg_cron "job startup timeout" on almost every job, statement timeouts
-- and connection timeouts while the DB looked idle -> disk IO budget exhausted on small compute.
-- The IO increase came from jobs (re-)enabled on 2026-10-21: cold archive (82, hourly archive +
-- purge), referenced sensor evidence sync (97) and sensor reliability refresh (19). Pause exactly
-- those until the IO budget refills; nothing is deleted, they are re-enabled later at a lower cadence.
do $$
declare j record;
begin
  for j in select jobid from cron.job where jobname in ('brian-cold-archive-daily', 'brian-referenced-sensor-evidence-sync-2m', 'brian-sensor-reliability-shadow-hourly') loop
    perform cron.alter_job(j.jobid, active := false);
  end loop;
end $$;

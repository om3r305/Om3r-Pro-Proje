-- Staged re-enable after the 2026-10-22 IO-budget incident (operator: one at a time, low cadence,
-- watch IO). Step 1: pm-calibration shadow collector, hourly at :41 (lightest job: a few Gamma/CLOB
-- reads and at most a few dozen inserts per hour). CORE was clean for 30 min before this step.
select cron.alter_job(job_id := (select jobid from cron.job where jobname = 'brian-pm-calibration-shadow-hourly'), active := true);

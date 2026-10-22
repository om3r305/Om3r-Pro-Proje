-- Staged re-enable, step 2: referenced sensor evidence sync (97), now hourly at :01 instead of twice
-- per hour. Step 1 (pm-calibration) ran 60 min with 0 failed / 51 ok cron runs.
select cron.alter_job(job_id := (select jobid from cron.job where jobname = 'brian-referenced-sensor-evidence-sync-2m'), schedule := '1 * * * *', active := true);

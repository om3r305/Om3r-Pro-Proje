-- Staged re-enable, step 5 (last): verified cold archive (82) once a day at 03:09 UTC (off-peak)
-- instead of hourly. Step 4 (reliability every 2 h) ran 2 h with 0 failed / 113 ok cron runs.
select cron.alter_job(job_id := (select jobid from cron.job where jobname = 'brian-cold-archive-daily'), schedule := '9 3 * * *', active := true);

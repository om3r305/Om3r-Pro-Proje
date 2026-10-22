-- Staged re-enable, step 4: sensor reliability shadow refresh (19) every 2 hours at :17 instead of
-- hourly. Step 3b (horizon challenger 30 min) ran 60 min with 0 failed / 56 ok.
select cron.alter_job(job_id := (select jobid from cron.job where jobname = 'brian-sensor-reliability-shadow-hourly'), schedule := '17 */2 * * *', active := true);

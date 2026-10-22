-- Staged re-enable, step 3a: alpha expected-edge challenger (47) every 30 min (:09, :39) instead of
-- every 10 min. Step 2 (sensor sync hourly) ran 65 min with 0 failed / 57 ok.
select cron.alter_job(job_id := (select jobid from cron.job where jobname = 'brian-evolution-alpha-edge-2m'), schedule := '9,39 * * * *', active := true);

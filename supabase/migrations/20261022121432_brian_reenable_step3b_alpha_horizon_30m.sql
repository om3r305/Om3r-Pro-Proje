-- Staged re-enable, step 3b: alpha horizon challenger (98) every 30 min (:18, :48) instead of every
-- 10 min. Step 3a (edge challenger 30 min) ran 60 min with 0 failed / 54 ok.
select cron.alter_job(job_id := (select jobid from cron.job where jobname = 'brian-evolution-alpha-horizon-5m'), schedule := '18,48 * * * *', active := true);

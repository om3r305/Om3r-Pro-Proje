-- Cold archive (82) four times a day (03:09, 09:09, 15:09, 21:09 UTC) instead of once (operator 2026-10-24).
-- Since the 2026-10-22 IO incident it ran daily: ~6 batches/day while CORE grows ~190 MiB/day.
-- Same code, same policies (archive -> verify -> purge after 24 h); only the cadence changes.
select cron.alter_job(job_id := (select jobid from cron.job where jobname = 'brian-cold-archive-daily'), schedule := '9 3,9,15,21 * * *', active := true);

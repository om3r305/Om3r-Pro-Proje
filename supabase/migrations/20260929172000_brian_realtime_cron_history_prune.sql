-- Target: brian-realtime. Two per-minute jobs write ~2.9k pg_cron history rows/day;
-- it is operational logging only, keep 3 days.
select cron.schedule('brian-cron-history-prune-daily','53 3 * * *',
  $$delete from cron.job_run_details where end_time < now() - interval '3 days'$$);

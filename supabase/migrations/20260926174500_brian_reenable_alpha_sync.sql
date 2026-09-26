-- Re-enable the canonical realtime->market-intelligence ALPHA sync only.
-- The older local compiler cron remains disabled to avoid duplicate work.
do $$
begin
  perform cron.alter_job(
    j.jobid,
    schedule := '3,13,23,33,43,53 * * * *',
    active := true
  )
  from cron.job j
  where j.jobname='brian-core-alpha-sync-3m';
end $$;

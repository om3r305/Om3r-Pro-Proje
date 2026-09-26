-- Realtime owns the hot ALPHA auditor cadence.
-- Core pg_cron had repeated job-startup timeouts under DB pressure; the realtime
-- scheduler now invokes the auditor directly with the existing internal key.
-- The legacy Core cron is disabled to avoid duplicate work.
do $body$
begin
  perform cron.alter_job(j.jobid, active := false)
  from cron.job j
  where j.jobname='brian-missed-opportunity-auditor-v3-5m';
end
$body$;

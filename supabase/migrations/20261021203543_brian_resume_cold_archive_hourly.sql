-- Resume the verified cold archive (operator approved 2026-10-21): brian-cold-archive v4 is the real
-- code again (was a pause stub since 2026-10-03). Hourly instead of the old 10-minute cadence to keep
-- disk IO low. Rows are deleted only after a verified upload, a 24 h grace period, a checksum
-- re-check and only past each table's retention window (brian_archive_purge_manifest).
select cron.alter_job(
  job_id := (select jobid from cron.job where jobname = 'brian-cold-archive-daily'),
  schedule := '9 * * * *',
  active := true
);

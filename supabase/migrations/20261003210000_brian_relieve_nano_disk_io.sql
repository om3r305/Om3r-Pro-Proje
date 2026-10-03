-- Target: brian-market-intelligence (t3a.nano). Disk IO hit 100% on 2026-10-01 and the DB
-- stopped accepting connections. pg_stat_statements share of block IO at the time:
--   sync_realtime_referenced_sensor_evidence_v1        41.6%  (~11 s / call, every 10 min)
--   resolve_sensor_reliability_prospective_calibration 12.5%  (~4.6 s / call, every 20 min)
-- Both feed shadow-only reliability research, so lower cadence only adds latency there.
-- Previous schedules, to revert:
--   brian-referenced-sensor-evidence-sync-2m  '1,11,21,31,41,51 * * * *'
--   brian-sensor-reliability-calibration-5m   '4,24,44 * * * *'
--   brian-cold-archive-daily                  '9,19,29,39,49,59 * * * *'
select cron.alter_job(job_id := (select jobid from cron.job where jobname = 'brian-referenced-sensor-evidence-sync-2m'), schedule := '1,31 * * * *');
select cron.alter_job(job_id := (select jobid from cron.job where jobname = 'brian-sensor-reliability-calibration-5m'), schedule := '4 * * * *');
-- brian-cold-archive is deployed as a pause stub; stop invoking it every 10 minutes.
select cron.alter_job(job_id := (select jobid from cron.job where jobname = 'brian-cold-archive-daily'), schedule := '9 3 * * *');

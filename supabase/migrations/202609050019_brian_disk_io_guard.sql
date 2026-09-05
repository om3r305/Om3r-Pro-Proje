-- Brian 2026 Disk IO guard.
-- Keeps ALPHA decision cadence at 1m while reducing redundant raw telemetry writes.
-- No live execution changes. No Phase 3.7 changes.

DO $$
DECLARE
  v_jobid bigint;
BEGIN
  SELECT jobid INTO v_jobid FROM cron.job WHERE jobname = 'brian-intrabar-eye-1m' LIMIT 1;
  IF v_jobid IS NOT NULL THEN
    PERFORM cron.alter_job(v_jobid, schedule := '*/2 * * * *');
  END IF;

  SELECT jobid INTO v_jobid FROM cron.job WHERE jobname = 'brian-sensor-mesh-5m' LIMIT 1;
  IF v_jobid IS NOT NULL THEN
    PERFORM cron.alter_job(v_jobid, schedule := '2-59/10 * * * *');
  END IF;
END
$$;

COMMENT ON EXTENSION pg_cron IS 'Brian 2026 uses guarded cadences for high-write shadow telemetry; ALPHA compiler remains 1m.';

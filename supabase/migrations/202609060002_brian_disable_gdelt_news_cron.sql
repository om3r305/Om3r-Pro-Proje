-- Brian 2026 load shedding: disable the legacy GDELT discovery eye.
-- Production evidence on 2026-09-06: 116/116 collector runs failed in the prior 24h,
-- consumed ~1303s runtime, stored 0 records, and ALPHA explicitly excludes news_gdelt
-- from directional evidence. Keep the job definition present but inactive so it can be
-- deliberately re-enabled later after the source/runtime is repaired.
-- SHADOW ONLY. No Phase 3.7 or live-execution semantics are changed.

DO $$
DECLARE
  v_jobid bigint;
BEGIN
  SELECT jobid
    INTO v_jobid
  FROM cron.job
  WHERE jobname = 'brian-news-eye-10m';

  IF v_jobid IS NOT NULL THEN
    PERFORM cron.alter_job(v_jobid, active := false);
  END IF;
END;
$$;

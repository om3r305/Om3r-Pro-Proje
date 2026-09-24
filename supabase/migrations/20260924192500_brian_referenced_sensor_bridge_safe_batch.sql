-- Brian Phase123 follow-up: keep referenced sensor sync batches under the
-- proven PostgREST in-filter/query-string safety bound.
-- SHADOW ONLY. No execution or promotion surface.

do $$
declare
  v_jobid bigint;
begin
  select jobid
  into v_jobid
  from cron.job
  where jobname = 'brian-referenced-sensor-evidence-sync-2m'
  limit 1;

  if v_jobid is null then
    perform cron.schedule(
      'brian-referenced-sensor-evidence-sync-2m',
      '1-59/2 * * * *',
      $cron$
        select brian_private.sync_realtime_referenced_sensor_evidence_v1(
          200,
          interval '36 hours'
        );
      $cron$
    );
  else
    perform cron.alter_job(
      v_jobid,
      '1-59/2 * * * *',
      $cron$
        select brian_private.sync_realtime_referenced_sensor_evidence_v1(
          200,
          interval '36 hours'
        );
      $cron$,
      null,
      null,
      true
    );
  end if;
end
$$;

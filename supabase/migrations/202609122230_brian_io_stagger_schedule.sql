-- Stagger MAIN Brian workloads so database-heavy workers do not all start on the same minute.
-- DIP schedules are intentionally untouched.
do $$
declare v_id bigint;
begin
  select jobid into v_id from cron.job where jobname='brian-intrabar-eye-1m';
  if v_id is not null then perform cron.alter_job(v_id, schedule := '1-59/2 * * * *'); end if;

  select jobid into v_id from cron.job where jobname='brian-evolution-alpha-edge-2m';
  if v_id is not null then perform cron.alter_job(v_id, schedule := '4-59/5 * * * *'); end if;

  select jobid into v_id from cron.job where jobname='brian-missed-opportunity-auditor-v3-5m';
  if v_id is not null then perform cron.alter_job(v_id, schedule := '8,23,38,53 * * * *'); end if;

  select jobid into v_id from cron.job where jobname='brian-sensor-mesh-5m';
  if v_id is not null then perform cron.alter_job(v_id, schedule := '5-59/10 * * * *'); end if;

  select jobid into v_id from cron.job where jobname='brian-official-macro-eye-10m';
  if v_id is not null then perform cron.alter_job(v_id, schedule := '8-59/10 * * * *'); end if;

  select jobid into v_id from cron.job where jobname='brian-compact-retention-hourly';
  if v_id is not null then perform cron.alter_job(v_id, schedule := '37 */6 * * *'); end if;
end $$;

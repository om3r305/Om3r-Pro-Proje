-- Production scheduler authority cleanup.
-- Keep the legacy Catalyst Sentinel crons disabled; realtime catalyst reaction is the sole authority.
-- Re-enable Phase127 horizon challenger after a successful live shadow probe.
do $$
declare
  j bigint;
begin
  for j in
    select jobid
    from cron.job
    where jobname in (
      'brian-catalyst-sentinel-fast-10s',
      'brian-catalyst-sentinel-reaction-30s'
    )
  loop
    perform cron.alter_job(j, active:=false);
  end loop;

  select jobid into j
  from cron.job
  where jobname='brian-evolution-alpha-horizon-5m'
  limit 1;

  if j is not null then
    perform cron.alter_job(j, active:=true);
  end if;
end $$;

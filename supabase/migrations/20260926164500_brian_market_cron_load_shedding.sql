-- Reduce brian-market-intelligence cron stampede without weakening fail-closed gates.
-- Critical lanes remain frequent; research/evolution/reporting lanes are staggered.
-- Applied live first on 2026-09-26 after PostgREST pool timeouts and pg_cron startup timeouts.

do $$
declare
  r record;
begin
  for r in
    select * from (values
      ('brian-dip-multiasset-v874-shadow-1m','1-59/3 * * * *'),
      ('brian-dip-position-guardian-1m','2-59/3 * * * *'),
      ('brian-frontier-standby-1m','2-59/5 * * * *'),
      ('brian-referenced-sensor-evidence-sync-2m','1-59/4 * * * *'),
      ('brian-evolution-alpha-edge-2m','9,19,29,39,49,59 * * * *'),
      ('brian-evolution-ocean-5m','11,26,41,56 * * * *'),
      ('brian-shadow-scorecard-5m','12,22,32,42,52 * * * *'),
      ('brian-sensor-reliability-calibration-5m','13,23,33,43,53 * * * *'),
      ('brian-meeting-backend-sync-1m','3,8,13,18,23,28,33,38,43,48,53,58 * * * *'),
      ('brian-core-recovery-dispatcher-v1','6,16,26,36,46,56 * * * *'),
      ('brian-missed-opportunity-auditor-v3-5m','7,17,27,37,47,57 * * * *'),
      ('brian-evolution-alpha-horizon-5m','8,18,28,38,48,58 * * * *'),
      ('brian-core-world-sync-6m','14,24,34,44,54 * * * *'),
      ('brian-core-discovery-sync-10m','16,31,46 * * * *')
    ) as x(jobname,schedule)
  loop
    perform cron.alter_job(j.jobid, schedule := r.schedule)
    from cron.job j
    where j.jobname=r.jobname;
  end loop;
end $$;

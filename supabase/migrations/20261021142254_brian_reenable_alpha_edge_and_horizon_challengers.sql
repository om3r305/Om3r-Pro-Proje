-- Re-enable the alpha expected-edge challenger (jobid 47, every 10 min at :x9) and the alpha
-- horizon challenger (jobid 98, every 10 min at :x8), approved by the operator on 2026-10-21.
-- Both feed brian-evolution-alpha-intelligence-status (UI "belief" panel), whose data had gone stale
-- (edge 2026-09-26, horizon 2026-10-11). History: 547 / 2,040 successful runs, avg 2.4 s / 3.6 s.
-- Shadow only: they score decisions; nothing trades or changes risk/treasury state.
do $$
declare j record;
begin
  for j in select jobid from cron.job where jobname in ('brian-evolution-alpha-edge-2m', 'brian-evolution-alpha-horizon-5m') loop
    perform cron.alter_job(j.jobid, active := true);
  end loop;
end $$;

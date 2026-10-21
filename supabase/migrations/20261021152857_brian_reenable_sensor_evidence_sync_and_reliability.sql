-- Re-enable the referenced sensor evidence sync (jobid 97, :01/:31; copies at most 200 REALTIME
-- sensor observations that CORE alpha decisions already reference, 36 h lookback) and the hourly
-- sensor reliability shadow refresh (jobid 19, :17). Approved by the operator on 2026-10-21.
-- Both were off since 2026-10-11 ~21:00 UTC, so the alpha edge challenger found no evidence groups
-- (every decision INSUFFICIENT_LAGGED_EVIDENCE). History: 179 / 90 successful runs, avg 22 s / 4.6 s.
-- Shadow only: nothing trades or changes risk/treasury state.
do $$
declare j record;
begin
  for j in select jobid from cron.job where jobname in ('brian-referenced-sensor-evidence-sync-2m', 'brian-sensor-reliability-shadow-hourly') loop
    perform cron.alter_job(j.jobid, active := true);
  end loop;
end $$;

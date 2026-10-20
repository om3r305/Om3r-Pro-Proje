-- Re-enable the evolution experiment runner (jobid 44, :27 hourly) and promotion council
-- (jobid 45, :34 hourly). Both were switched off during the 2026-09-26 load shedding; all 19 runs
-- before that succeeded (runner avg 23 s, council avg 7 s). A manual run on 2026-10-20 21:35 UTC
-- with the unchanged deployed code succeeded in 1.1 s. Promotion still requires a human
-- (human_promotion_required); nothing here trades or changes risk/treasury state.
do $$
declare j record;
begin
  for j in select jobid from cron.job where jobname in ('brian-evolution-experiment-runner-hourly', 'brian-evolution-promotion-council-hourly') loop
    perform cron.alter_job(j.jobid, active := true);
  end loop;
end $$;

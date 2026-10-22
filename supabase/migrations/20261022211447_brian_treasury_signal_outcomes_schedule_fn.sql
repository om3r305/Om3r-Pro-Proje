-- Align CORE with the repository's dormant-migration contract: the outcomes job (already scheduled
-- by the operator-approved rollout on 2026-10-22) is (re)created only through this explicit function.
create or replace function brian_private.schedule_treasury_signal_outcomes_v1()
returns bigint
language plpgsql
security definer
set search_path = pg_catalog, public
as $fn$
declare j bigint;
begin
  j := cron.schedule('brian-treasury-signal-outcomes-hourly', '23 * * * *',
    $cron$select public.brian_record_treasury_signal_outcomes_v1(interval '3 hours')$cron$);
  return j;
end $fn$;
revoke all on function brian_private.schedule_treasury_signal_outcomes_v1() from public, anon, authenticated, service_role;

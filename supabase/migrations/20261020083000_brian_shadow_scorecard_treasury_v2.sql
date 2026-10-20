-- New 7-day forward scorecard period for the treasury engine (user-approved 20 Oct 2026).
--
-- forward-20260922-v1-treasury stays as it is (append-only; its gap after 2026-09-26 14:20 UTC is
-- reported, never back-filled). The new period has its own protocol_id and the SAME rules as v1,
-- so the two remain comparable. Only the treasury engine gets a new period; the DIP protocols
-- (separate owner) are not touched, and no cash, risk or trade state is changed.
--
-- Runs entirely on CORE: pg_cron job brian-shadow-scorecard-5m (12,22,32,42,52 * * * *) calls the
-- existing brian_capture_shadow_observations() (statement_timeout 8 s; one indexed read of the
-- newest treasury snapshot per run, ~6 rows/hour, ~1,000 rows in 7 days). No edge function, no
-- external host, no paid resource.

insert into public.brian_shadow_protocols (protocol_id, engine_id, starts_at, ends_at, rules)
select 'forward-20261020-v2-treasury', 'treasury', now(), now() + interval '7 days', v1.rules
from public.brian_shadow_protocols v1
where v1.protocol_id = 'forward-20260922-v1-treasury'
on conflict (protocol_id) do nothing;

do $$
declare j bigint;
begin
  if not exists (select 1 from public.brian_shadow_protocols where protocol_id = 'forward-20261020-v2-treasury') then
    raise exception 'treasury v2 protocol was not created (v1 rules missing)';
  end if;
  select jobid into j from cron.job where jobname = 'brian-shadow-scorecard-5m';
  if j is null then
    perform cron.schedule('brian-shadow-scorecard-5m', '12,22,32,42,52 * * * *', 'select public.brian_capture_shadow_observations()');
  else
    perform cron.alter_job(j, schedule := '12,22,32,42,52 * * * *', active := true);
  end if;
end $$;

-- First observation now, so the period starts with a recorded point.
select public.brian_capture_shadow_observations();

-- Period 3: a clean forward measurement of the treasury engine from 2026-10-21 (operator approved:
-- "temiz ölçüm dönemi"). Started at a moment with zero open shadow positions, so the period is not
-- polluted by trades opened earlier. Same rules as v1/v2 (comparable), 14 days so more trades close.
-- Nothing is deleted or reset: v1 and v2 stay intact; the scorecard shows the newest period.
insert into public.brian_shadow_protocols (protocol_id, engine_id, starts_at, ends_at, rules)
select 'forward-20261021-v3-treasury', 'treasury', now(), now() + interval '14 days', v1.rules
from public.brian_shadow_protocols v1
where v1.protocol_id = 'forward-20260922-v1-treasury'
  and (select jsonb_array_length(positions) from public.brian_treasury_shadow_snapshots order by observed_at desc limit 1) = 0
on conflict (protocol_id) do nothing;

do $$
begin
  if not exists (select 1 from public.brian_shadow_protocols where protocol_id = 'forward-20261021-v3-treasury') then
    raise exception 'v3 not created: open shadow positions at start (retry when flat)';
  end if;
end $$;

select public.brian_capture_shadow_observations();

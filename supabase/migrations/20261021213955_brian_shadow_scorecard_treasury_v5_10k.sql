-- Period 5 replaces period 4 as the measurement of the $10,000 restart. Period 4's first point was
-- captured inside the restart transaction and therefore still saw the old $5,000-session snapshot;
-- it stays recorded (append-only) but is superseded. Period 5 starts in a later transaction, on the
-- flat $10,000 genesis snapshot.
do $$
begin
  if (select starting_equity_usd from public.brian_treasury_shadow_latest) <> 10000
     or (select jsonb_array_length(positions) from public.brian_treasury_shadow_latest) <> 0 then
    raise exception 'v5 refused: latest treasury snapshot is not the flat $10,000 restart';
  end if;
end $$;

insert into public.brian_shadow_protocols (protocol_id, engine_id, starts_at, ends_at, rules)
select 'forward-20261021-v5-treasury-10k', 'treasury', now(), now() + interval '14 days', v1.rules
from public.brian_shadow_protocols v1
where v1.protocol_id = 'forward-20260922-v1-treasury'
on conflict (protocol_id) do nothing;

select public.brian_capture_shadow_observations();

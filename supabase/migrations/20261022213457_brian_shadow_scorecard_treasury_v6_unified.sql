-- Period 6: measurement of the unified evidence treasury (gate v7, brian.treasury-unified.v1),
-- started flat right after the switch-on. Same rules as v1..v5; 14 days. Earlier periods stay intact.
do $$
begin
  if (select jsonb_array_length(positions) from public.brian_treasury_shadow_latest) <> 0 then
    raise exception 'v6 refused: shadow book not flat';
  end if;
end $$;

insert into public.brian_shadow_protocols (protocol_id, engine_id, starts_at, ends_at, rules)
select 'forward-20261022-v6-treasury-unified', 'treasury', now(), now() + interval '14 days', v1.rules
from public.brian_shadow_protocols v1
where v1.protocol_id = 'forward-20260922-v1-treasury'
on conflict (protocol_id) do nothing;

select public.brian_capture_shadow_observations();

-- Guard Big Move precursor lane from becoming a dip/reversal strategy.
create or replace function public.brian_guard_big_move_opportunity()
returns trigger
language plpgsql
as $$
declare
  lane text := coalesce(new.metadata->>'lane','');
  ver text := coalesce(new.metadata->>'version','');
  d numeric := coalesce((new.metadata->>'direction')::numeric,0);
  ch numeric := coalesce((new.metadata->>'price_change_pct')::numeric,0);
begin
  if ver='brian.big-move-hunter.v1' and lane='PRECURSOR_CONVERGENCE' then
    if coalesce(new.market_confirmation,0) < 0.35 then return null; end if;
    if abs(ch) >= 4 and d * ch < 0 then return null; end if;
  end if;
  return new;
end;
$$;
drop trigger if exists brian_big_move_opportunity_guard on public.brian_opportunity_observations;
create trigger brian_big_move_opportunity_guard before insert or update on public.brian_opportunity_observations for each row execute function public.brian_guard_big_move_opportunity();

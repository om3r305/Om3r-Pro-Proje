-- Sparse long-tail Catalyst monitoring for important official events.
-- Hot lane stays dense for 30m; later rechecks happen only at 2h / 6h / 12h.

create or replace function public.brian_catalyst_long_tail_watch_guard_v1()
returns trigger
language plpgsql
set search_path to 'public','pg_temp'
as $$
begin
  if new.source_id ~ '^official:(fed|ecb|boe|boj|boc|sec|treasury|bundesbank|cftc)'
     and new.status in ('WATCHING','BUILDING','BREAKOUT_CANDIDATE','CONFIRMED') then
    new.expires_at := greatest(new.expires_at, new.started_at + interval '13 hours');
    new.metadata := coalesce(new.metadata,'{}'::jsonb) || jsonb_build_object(
      'long_tail_watch', true,
      'long_tail_rechecks_minutes', jsonb_build_array(120,360,720),
      'long_tail_policy', 'HOT_30M_THEN_2H_6H_12H'
    );
  end if;
  return new;
end;
$$;

drop trigger if exists brian_catalyst_long_tail_watch_guard_v1
  on public.brian_catalyst_sentinel_watches;

create trigger brian_catalyst_long_tail_watch_guard_v1
before insert or update of source_id,started_at,expires_at,status
on public.brian_catalyst_sentinel_watches
for each row
execute function public.brian_catalyst_long_tail_watch_guard_v1();

update public.brian_catalyst_sentinel_watches
set expires_at = greatest(expires_at, started_at + interval '13 hours'),
    recheck_count = case
      when now() - started_at >= interval '12 hours' then 7
      when now() - started_at >= interval '6 hours' then 6
      when now() - started_at >= interval '2 hours' then 5
      when now() - started_at >= interval '30 minutes' then 4
      when now() - started_at >= interval '15 minutes' then 3
      when now() - started_at >= interval '10 minutes' then 2
      when now() - started_at >= interval '5 minutes' then 1
      else 0
    end,
    next_recheck_at = now() - interval '1 second',
    metadata = coalesce(metadata,'{}'::jsonb) || jsonb_build_object(
      'long_tail_watch', true,
      'long_tail_rechecks_minutes', jsonb_build_array(120,360,720),
      'long_tail_policy', 'HOT_30M_THEN_2H_6H_12H',
      'long_tail_reactivated_at', now()
    ),
    updated_at = now()
where source_id ~ '^official:(fed|ecb|boe|boj|boc|sec|treasury|bundesbank|cftc)'
  and started_at > now() - interval '12 hours'
  and status in ('WATCHING','BUILDING','BREAKOUT_CANDIDATE','CONFIRMED')
  and expires_at <= now();

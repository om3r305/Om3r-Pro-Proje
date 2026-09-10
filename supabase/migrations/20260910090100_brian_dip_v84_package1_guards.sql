-- Additional V8.4-only guards. Additive; never mutates V8.3 objects.

create unique index if not exists brian_dip_v84_one_entry_per_episode
  on public.brian_dip_v84_ledger(episode_id)
  where event_kind in ('BUY','SHORT_OPEN');

-- A release may only be sealed explicitly, with a real source hash.
create or replace function public.brian_dip_v84_seal_release(
  p_release_id text,
  p_expected_manifest_hash text,
  p_logic_hash text
) returns boolean
language plpgsql
set search_path='pg_catalog','public'
as $$
declare n integer;
begin
  if p_release_id is null or p_expected_manifest_hash is null or p_logic_hash is null
     or p_logic_hash='UNSEALED_GITHUB_ONLY' or length(p_logic_hash)<32
  then raise exception 'V84_INVALID_RELEASE_SEAL'; end if;

  update public.brian_dip_v84_releases
  set status='SEALED',logic_hash=p_logic_hash,sealed_at=clock_timestamp()
  where release_id=p_release_id
    and status='PREPARED'
    and strategy_manifest_hash=p_expected_manifest_hash
    and logic_hash='UNSEALED_GITHUB_ONLY';
  get diagnostics n=row_count;
  if n<>1 then raise exception 'V84_RELEASE_SEAL_CONFLICT'; end if;
  return true;
end $$;

revoke all on function public.brian_dip_v84_seal_release(text,text,text) from public,anon,authenticated;
grant execute on function public.brian_dip_v84_seal_release(text,text,text) to service_role;

-- No schedule or deployment is activated by this migration.

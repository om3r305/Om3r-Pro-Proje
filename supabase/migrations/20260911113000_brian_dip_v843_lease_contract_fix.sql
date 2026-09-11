-- Brian DIP V8.4.3 lease contract repair.
-- SHADOW ONLY. Adds the V8.4.3 profit-protect lease identity to renewal so
-- acquire -> renew -> commit all remain release-scoped to the same sealed release.

create or replace function public.brian_dip_v84_renew_lease(
  p_lease_key text,
  p_owner_id text,
  p_lease_generation bigint,
  p_lease_seconds integer
) returns boolean
language plpgsql
set search_path='pg_catalog','public'
as $$
declare
  n integer;
  wanted_release text;
begin
  if p_lease_seconds not between 5 and 120 then return false; end if;

  wanted_release:=case p_lease_key
    when 'brian-dip-v84-worker' then 'dip-v84-package1-evidence-20260910.1'
    when 'brian-dip-v84-authority-worker' then 'dip-v841-brian-authority-20260911.1'
    when 'brian-dip-v84-foresight' then 'dip-v84-package1-evidence-20260910.1'
    when 'brian-dip-v842-long-stateful-worker' then 'dip-v842-long-stateful-20260911.1'
    when 'brian-dip-v842-foresight' then 'dip-v842-long-stateful-20260911.1'
    when 'brian-dip-v843-profit-protect-worker' then 'dip-v843-profit-protect-20260911.1'
    else null
  end;

  if wanted_release is null then return false; end if;

  update public.brian_dip_v84_worker_leases
  set heartbeat_at=clock_timestamp(),
      expires_at=clock_timestamp()+make_interval(secs=>p_lease_seconds)
  where lease_key=p_lease_key
    and owner_id=p_owner_id
    and lease_generation=p_lease_generation
    and expires_at>clock_timestamp()
    and release_id=wanted_release;

  get diagnostics n=row_count;
  return n=1;
end $$;

revoke all on function public.brian_dip_v84_renew_lease(text,text,bigint,integer) from public,anon,authenticated;
grant execute on function public.brian_dip_v84_renew_lease(text,text,bigint,integer) to service_role;

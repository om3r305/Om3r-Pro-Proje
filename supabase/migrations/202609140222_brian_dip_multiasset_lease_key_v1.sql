create or replace function public.brian_dip_v84_acquire_lease(p_lease_key text, p_owner_id text, p_lease_seconds integer)
returns table(acquired boolean, lease_generation bigint)
language plpgsql
set search_path to 'pg_catalog', 'public'
as $function$
declare r public.brian_dip_v84_worker_leases%rowtype; now_at timestamptz:=clock_timestamp(); wanted_release text;
begin
  if coalesce(length(p_lease_key),0)=0 or coalesce(length(p_owner_id),0)=0 or p_lease_seconds not between 5 and 120 then raise exception 'V84_INVALID_LEASE_REQUEST'; end if;
  wanted_release:=case p_lease_key
    when 'brian-dip-v84-worker' then 'dip-v84-package1-evidence-20260910.1'
    when 'brian-dip-v84-authority-worker' then 'dip-v841-brian-authority-20260911.1'
    when 'brian-dip-v84-foresight' then 'dip-v84-package1-evidence-20260910.1'
    when 'brian-dip-v842-long-stateful-worker' then 'dip-v842-long-stateful-20260911.1'
    when 'brian-dip-v842-foresight' then 'dip-v842-long-stateful-20260911.1'
    when 'brian-dip-v843-profit-protect-worker' then 'dip-v843-profit-protect-20260911.1'
    when 'brian-dip-v844-cycle-worker' then 'dip-v844-cycle-forecast-20260911.1'
    when 'brian-dip-multiasset-worker-v1' then 'dip-multiasset-shadow-v1'
    else null end;
  if wanted_release is null then raise exception 'V84_UNKNOWN_LEASE_KEY'; end if;
  insert into public.brian_dip_v84_worker_leases(lease_key) values(p_lease_key) on conflict do nothing;
  select * into r from public.brian_dip_v84_worker_leases where lease_key=p_lease_key for update;
  if r.owner_id is null or r.expires_at is null or r.expires_at<=now_at then
    update public.brian_dip_v84_worker_leases set owner_id=p_owner_id,lease_generation=r.lease_generation+1,acquired_at=now_at,heartbeat_at=now_at,expires_at=now_at+make_interval(secs=>p_lease_seconds),release_id=wanted_release where lease_key=p_lease_key returning * into r;
    return query select true,r.lease_generation;return;
  end if;
  return query select false,r.lease_generation;
end $function$;

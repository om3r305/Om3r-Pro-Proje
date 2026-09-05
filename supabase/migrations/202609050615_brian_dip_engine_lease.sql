-- Brian Dip V4 browser engine lease.
-- Ephemeral control-plane state only; trading evidence tables remain append-only.
-- SHADOW/PAPER ONLY. Frozen Brian Phase 3.7 is not referenced.

create table if not exists public.brian_dip_engine_leases (
  session_id text primary key,
  engine_token_sha256 text not null check (engine_token_sha256 ~ '^[0-9a-f]{64}$'),
  lease_generation bigint not null default 1 check (lease_generation > 0),
  claimed_at timestamptz not null default now(),
  heartbeat_at timestamptz not null default now(),
  claimed_by text not null default 'monster-coins-pro-dip-v4',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

alter table public.brian_dip_engine_leases enable row level security;
revoke all on public.brian_dip_engine_leases from anon, authenticated;
grant select, insert, update on public.brian_dip_engine_leases to service_role;

create index if not exists brian_dip_engine_leases_heartbeat_idx
  on public.brian_dip_engine_leases(heartbeat_at desc);

create or replace function public.brian_dip_claim_engine(
  p_session_id text,
  p_engine_token_sha256 text,
  p_stale_after_seconds integer default 25
) returns public.brian_dip_engine_leases
language plpgsql
security definer
set search_path=pg_catalog,public
as $$
declare
  l public.brian_dip_engine_leases%rowtype;
  inserted public.brian_dip_engine_leases%rowtype;
begin
  if p_stale_after_seconds < 10 or p_stale_after_seconds > 300 then
    raise exception 'BRIAN_DIP: invalid lease stale threshold';
  end if;
  if p_engine_token_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'BRIAN_DIP: invalid engine token hash';
  end if;

  perform pg_advisory_xact_lock(hashtextextended('brian-dip-engine-lease:'||p_session_id,0));
  select * into l from public.brian_dip_engine_leases where session_id=p_session_id for update;

  if not found then
    insert into public.brian_dip_engine_leases(session_id,engine_token_sha256,lease_generation,claimed_at,heartbeat_at)
    values(p_session_id,p_engine_token_sha256,1,now(),now())
    returning * into inserted;
    return inserted;
  end if;

  if l.engine_token_sha256 = p_engine_token_sha256 then
    update public.brian_dip_engine_leases
      set heartbeat_at=now()
      where session_id=p_session_id
      returning * into inserted;
    return inserted;
  end if;

  if l.heartbeat_at > now() - make_interval(secs => p_stale_after_seconds) then
    raise exception 'BRIAN_DIP_ENGINE_LEASE_ACTIVE';
  end if;

  update public.brian_dip_engine_leases
    set engine_token_sha256=p_engine_token_sha256,
        lease_generation=lease_generation+1,
        claimed_at=now(),
        heartbeat_at=now()
    where session_id=p_session_id
    returning * into inserted;
  return inserted;
end;
$$;

revoke all on function public.brian_dip_claim_engine(text,text,integer) from public,anon,authenticated;
grant execute on function public.brian_dip_claim_engine(text,text,integer) to service_role;

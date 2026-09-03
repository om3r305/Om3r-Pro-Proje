-- Brian collector-level atomic lease/mutex (Item 1, brian-2026 issue #32).
--
-- Standalone by design: creates only a plain table, two plpgsql functions, and an append-only
-- audit table, with no dependency on vault/pg_net/pg_cron or any other Supabase-specific
-- extension. That means this single file can be applied to, and exercised against, a vanilla
-- Postgres instance in CI (see tests/test_brian2026_collector_lease_postgres.py) independent of
-- the rest of the Supabase migration chain. The `create or replace function
-- public.brian_reject_mutation()` below is therefore an intentional, idempotent redefinition of
-- the identical function already created by 202609030001_brian_intelligence_memory.sql:6-17 --
-- harmless on the real deployed database (same name/args/body), but required for this file to be
-- independently applicable.
--
-- brian_collector_leases is a mutable OPERATIONAL-STATE row per collector_id -- a deliberate,
-- explicit exception to the append-only evidence doctrine used everywhere else in this schema,
-- because a mutex must be able to transition ownership. brian_collector_lease_events is the
-- append-only audit trail of that state machine and follows the normal doctrine.

create or replace function public.brian_reject_mutation()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  if current_user = 'postgres' then
    if tg_op = 'DELETE' then return old; else return new; end if;
  end if;
  raise exception 'BRIAN_APPEND_ONLY: % on %.% is forbidden', tg_op, tg_table_schema, tg_table_name;
end;
$$;

create table if not exists public.brian_collector_leases (
  collector_id text primary key,
  owner_token text not null,
  acquired_at timestamptz not null,
  lease_until timestamptz not null,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  updated_at timestamptz not null default now()
);

alter table public.brian_collector_leases enable row level security;
revoke all on public.brian_collector_leases from anon, authenticated;
grant select, insert, update on public.brian_collector_leases to service_role;
-- Deliberately no append-only trigger here: this row must be able to transition ownership.
-- That is exactly what makes it an operational-state table rather than evidence.

create table if not exists public.brian_collector_lease_events (
  event_id text primary key default gen_random_uuid()::text,
  collector_id text not null,
  owner_token text not null,
  event text not null check (event in ('ACQUIRED', 'BLOCKED_ACTIVE', 'RELEASED', 'EXPIRED_RECOVERY')),
  observed_at timestamptz not null default now(),
  lease_until timestamptz,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  metadata jsonb not null default '{}'::jsonb
);

create index if not exists brian_collector_lease_events_collector_time_idx
  on public.brian_collector_lease_events(collector_id, observed_at desc);

alter table public.brian_collector_lease_events enable row level security;
revoke all on public.brian_collector_lease_events from anon, authenticated;
revoke update, delete, truncate, references, trigger on public.brian_collector_lease_events from service_role;
grant select, insert on public.brian_collector_lease_events to service_role;

drop trigger if exists brian_collector_lease_events_append_only on public.brian_collector_lease_events;
create trigger brian_collector_lease_events_append_only
  before update or delete on public.brian_collector_lease_events
  for each row execute function public.brian_reject_mutation();

-- Atomic acquire: exactly the primitive brian-2026 issue #32 asked for -- one
-- INSERT ... ON CONFLICT (collector_id) DO UPDATE ... WHERE lease_until <= now(), which Postgres
-- resolves atomically at the row/unique-index level even when two callers race for the same
-- collector_id (including a brand-new collector_id that does not have a row yet: the unique
-- constraint on collector_id serializes the concurrent inserts regardless). The preceding
-- `select ... for update` does not provide the safety guarantee by itself -- it exists only to
-- read the pre-existing lease's state so the audit event below can be classified as a fresh
-- ACQUIRED vs. an EXPIRED_RECOVERY takeover; both statements run inside the single transaction of
-- this one RPC call, so no read-then-write race window is exposed to callers.
create or replace function public.brian_acquire_collector_lease(
  p_collector_id text,
  p_owner_token text,
  p_lease_seconds integer
) returns boolean
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_now timestamptz := clock_timestamp();
  v_prior_lease_until timestamptz;
  v_had_prior boolean;
  v_rows integer;
  v_acquired boolean;
begin
  if p_collector_id is null or length(trim(p_collector_id)) = 0 then
    raise exception 'BRIAN_LEASE: p_collector_id is required';
  end if;
  if p_owner_token is null or length(trim(p_owner_token)) = 0 then
    raise exception 'BRIAN_LEASE: p_owner_token is required';
  end if;
  if p_lease_seconds is null or p_lease_seconds <= 0 then
    raise exception 'BRIAN_LEASE: p_lease_seconds must be positive';
  end if;

  select lease_until into v_prior_lease_until
    from public.brian_collector_leases
    where collector_id = p_collector_id
    for update;
  v_had_prior := found;

  insert into public.brian_collector_leases (collector_id, owner_token, acquired_at, lease_until, updated_at)
  values (p_collector_id, p_owner_token, v_now, v_now + make_interval(secs => p_lease_seconds), v_now)
  on conflict (collector_id) do update
    set owner_token = excluded.owner_token,
        acquired_at = excluded.acquired_at,
        lease_until = excluded.lease_until,
        updated_at = excluded.updated_at
    where public.brian_collector_leases.lease_until <= v_now;

  get diagnostics v_rows = row_count;
  v_acquired := v_rows > 0;

  insert into public.brian_collector_lease_events (collector_id, owner_token, event, observed_at, lease_until, metadata)
  values (
    p_collector_id,
    p_owner_token,
    case
      when not v_acquired then 'BLOCKED_ACTIVE'
      when v_had_prior and v_prior_lease_until <= v_now then 'EXPIRED_RECOVERY'
      else 'ACQUIRED'
    end,
    v_now,
    case when v_acquired then v_now + make_interval(secs => p_lease_seconds) else v_prior_lease_until end,
    jsonb_build_object('lease_seconds', p_lease_seconds, 'had_prior_lease', v_had_prior)
  );

  return v_acquired;
end;
$$;

revoke all on function public.brian_acquire_collector_lease(text, text, integer) from public;
revoke all on function public.brian_acquire_collector_lease(text, text, integer) from anon, authenticated;
grant execute on function public.brian_acquire_collector_lease(text, text, integer) to service_role;

-- Atomic release: succeeds only when collector_id + owner_token match the current lease row, so
-- an old/stale invocation (e.g. one whose lease already expired and was taken over by a newer
-- owner_token) can never release a newer owner's lease. Backdating lease_until to now() rather
-- than deleting the row keeps the append-only event history's lease_until values meaningful and
-- avoids a delete-then-reinsert race on the next acquire.
create or replace function public.brian_release_collector_lease(
  p_collector_id text,
  p_owner_token text
) returns boolean
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_now timestamptz := clock_timestamp();
  v_rows integer;
  v_released boolean;
begin
  update public.brian_collector_leases
    set lease_until = v_now, updated_at = v_now
    where collector_id = p_collector_id and owner_token = p_owner_token;
  get diagnostics v_rows = row_count;
  v_released := v_rows > 0;

  if v_released then
    insert into public.brian_collector_lease_events (collector_id, owner_token, event, observed_at, lease_until, metadata)
    values (p_collector_id, p_owner_token, 'RELEASED', v_now, v_now, '{}'::jsonb);
  end if;

  return v_released;
end;
$$;

revoke all on function public.brian_release_collector_lease(text, text) from public;
revoke all on function public.brian_release_collector_lease(text, text) from anon, authenticated;
grant execute on function public.brian_release_collector_lease(text, text) to service_role;

-- Brian Phase 74: immutable governed-cycle authorization binding.
--
-- GitHub-only until explicit rollout.
--
-- A Phase69 governed cycle may enter the Phase71 write-ahead execution path only
-- after this database boundary binds its content-addressed cycle_id to the
-- current, already-persisted Phase72/73 risk receipt. The binding uses the same
-- Phase70 runtime lease/fence/advisory-lock boundary.
--
-- A crash after BOUND but before cycle write-ahead is safe: there is no paper
-- side effect yet, and the immutable orphan binding remains audit evidence.
-- A crash after cycle write-ahead has both sides durable.

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

create table if not exists public.brian_governed_cycle_bindings (
  runtime_id text not null,
  cycle_id text not null,
  governed_result_id text not null,
  policy_fingerprint text not null,
  risk_version bigint not null check (risk_version > 0),
  risk_ledger_hash text not null,
  risk_receipt_id text not null,
  runtime_version_before bigint not null check (runtime_version_before > 0),
  fencing_token bigint not null check (fencing_token > 0),
  bound_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, cycle_id),
  unique (runtime_id, governed_result_id),
  check (length(cycle_id) = 64),
  check (length(governed_result_id) = 64),
  check (length(policy_fingerprint) = 64),
  check (length(risk_ledger_hash) = 64),
  check (length(risk_receipt_id) = 64)
);

create index if not exists brian_governed_cycle_bindings_risk_idx
  on public.brian_governed_cycle_bindings(
    runtime_id, risk_version, risk_receipt_id, bound_at desc
  );

alter table public.brian_governed_cycle_bindings enable row level security;
revoke all on public.brian_governed_cycle_bindings from anon, authenticated, service_role;

drop trigger if exists brian_governed_cycle_bindings_append_only
  on public.brian_governed_cycle_bindings;
create trigger brian_governed_cycle_bindings_append_only
  before update or delete on public.brian_governed_cycle_bindings
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_governed_cycle_binding_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  cycle_id text not null,
  owner_token text not null,
  fencing_token bigint,
  runtime_version bigint,
  risk_version bigint,
  event text not null check (
    event in (
      'BOUND',
      'DUPLICATE',
      'LEASE_LOST',
      'RUNTIME_VERSION_CONFLICT',
      'RISK_VERSION_CONFLICT'
    )
  ),
  observed_at timestamptz not null default now(),
  governed_result_id text,
  risk_receipt_id text,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_governed_cycle_binding_events_runtime_idx
  on public.brian_governed_cycle_binding_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_governed_cycle_binding_events enable row level security;
revoke all on public.brian_governed_cycle_binding_events from anon, authenticated, service_role;

drop trigger if exists brian_governed_cycle_binding_events_append_only
  on public.brian_governed_cycle_binding_events;
create trigger brian_governed_cycle_binding_events_append_only
  before update or delete on public.brian_governed_cycle_binding_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_bind_governed_shadow_cycle(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_expected_runtime_version bigint,
  p_risk_version bigint,
  p_risk_ledger_hash text,
  p_risk_receipt_id text,
  p_cycle_id text,
  p_governed_result_id text,
  p_policy_fingerprint text
) returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_now timestamptz;
  v_owner text;
  v_fence bigint;
  v_runtime_version bigint;
  v_lease_until timestamptz;
  v_risk_version bigint;
  v_risk_ledger_hash text;
  v_risk_head_entry_id text;
  v_head_receipt_id text;
  v_existing public.brian_governed_cycle_bindings%rowtype;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_expected_runtime_version is null or p_expected_runtime_version <= 0
     or p_risk_version is null or p_risk_version <= 0
     or p_risk_ledger_hash is null or length(p_risk_ledger_hash) <> 64
     or p_risk_receipt_id is null or length(p_risk_receipt_id) <> 64
     or p_cycle_id is null or length(p_cycle_id) <> 64
     or p_governed_result_id is null or length(p_governed_result_id) <> 64
     or p_policy_fingerprint is null or length(p_policy_fingerprint) <> 64 then
    raise exception 'PHASE74_BIND: valid runtime/fence/version/hash identities required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, lease_until
    into v_owner, v_fence, v_runtime_version, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if not found
     or v_owner <> p_owner_token
     or v_fence <> p_fencing_token
     or v_lease_until <= v_now then
    insert into public.brian_governed_cycle_binding_events(
      runtime_id, cycle_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      governed_result_id, risk_receipt_id
    ) values (
      p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
      v_runtime_version, p_risk_version, 'LEASE_LOST', v_now,
      p_governed_result_id, p_risk_receipt_id
    );
    return jsonb_build_object(
      'bound', false,
      'duplicate', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'risk_version', p_risk_version,
      'fencing_token', v_fence
    );
  end if;

  if v_runtime_version <> p_expected_runtime_version then
    insert into public.brian_governed_cycle_binding_events(
      runtime_id, cycle_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      governed_result_id, risk_receipt_id,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
      v_runtime_version, p_risk_version, 'RUNTIME_VERSION_CONFLICT', v_now,
      p_governed_result_id, p_risk_receipt_id,
      jsonb_build_object('expected_runtime_version', p_expected_runtime_version)
    );
    return jsonb_build_object(
      'bound', false,
      'duplicate', false,
      'status', 'RUNTIME_VERSION_CONFLICT',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'risk_version', p_risk_version,
      'fencing_token', p_fencing_token
    );
  end if;

  select version, ledger_hash, head_entry_id
    into v_risk_version, v_risk_ledger_hash, v_risk_head_entry_id
  from public.brian_operational_risk_heads
  where runtime_id = p_runtime_id
  for update;

  if not found
     or v_risk_version <> p_risk_version
     or v_risk_ledger_hash is distinct from p_risk_ledger_hash
     or v_risk_head_entry_id is null then
    insert into public.brian_governed_cycle_binding_events(
      runtime_id, cycle_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      governed_result_id, risk_receipt_id,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
      v_runtime_version, v_risk_version, 'RISK_VERSION_CONFLICT', v_now,
      p_governed_result_id, p_risk_receipt_id,
      jsonb_build_object(
        'expected_risk_version', p_risk_version,
        'expected_risk_ledger_hash', p_risk_ledger_hash,
        'current_risk_ledger_hash', v_risk_ledger_hash
      )
    );
    return jsonb_build_object(
      'bound', false,
      'duplicate', false,
      'status', 'RISK_VERSION_CONFLICT',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'risk_version', v_risk_version,
      'fencing_token', p_fencing_token
    );
  end if;

  select receipt_id
    into v_head_receipt_id
  from public.brian_operational_risk_entries
  where runtime_id = p_runtime_id
    and entry_id = v_risk_head_entry_id;

  if v_head_receipt_id is distinct from p_risk_receipt_id then
    insert into public.brian_governed_cycle_binding_events(
      runtime_id, cycle_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      governed_result_id, risk_receipt_id,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
      v_runtime_version, v_risk_version, 'RISK_VERSION_CONFLICT', v_now,
      p_governed_result_id, p_risk_receipt_id,
      jsonb_build_object(
        'expected_head_receipt_id', v_head_receipt_id,
        'submitted_risk_receipt_id', p_risk_receipt_id
      )
    );
    return jsonb_build_object(
      'bound', false,
      'duplicate', false,
      'status', 'RISK_VERSION_CONFLICT',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'risk_version', v_risk_version,
      'fencing_token', p_fencing_token
    );
  end if;

  select *
    into v_existing
  from public.brian_governed_cycle_bindings
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if found then
    if v_existing.governed_result_id <> p_governed_result_id
       or v_existing.policy_fingerprint <> p_policy_fingerprint
       or v_existing.risk_version <> p_risk_version
       or v_existing.risk_ledger_hash <> p_risk_ledger_hash
       or v_existing.risk_receipt_id <> p_risk_receipt_id
       or v_existing.runtime_version_before <> p_expected_runtime_version then
      raise exception 'PHASE74_BIND_CONFLICT: cycle % already bound to different evidence',
        p_cycle_id;
    end if;

    insert into public.brian_governed_cycle_binding_events(
      runtime_id, cycle_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      governed_result_id, risk_receipt_id
    ) values (
      p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
      v_runtime_version, v_risk_version, 'DUPLICATE', v_now,
      p_governed_result_id, p_risk_receipt_id
    );
    return jsonb_build_object(
      'bound', true,
      'duplicate', true,
      'status', 'DUPLICATE',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'risk_version', v_risk_version,
      'fencing_token', p_fencing_token
    );
  end if;

  insert into public.brian_governed_cycle_bindings(
    runtime_id, cycle_id, governed_result_id, policy_fingerprint,
    risk_version, risk_ledger_hash, risk_receipt_id,
    runtime_version_before, fencing_token, bound_at
  ) values (
    p_runtime_id, p_cycle_id, p_governed_result_id, p_policy_fingerprint,
    p_risk_version, p_risk_ledger_hash, p_risk_receipt_id,
    p_expected_runtime_version, p_fencing_token, v_now
  );

  insert into public.brian_governed_cycle_binding_events(
    runtime_id, cycle_id, owner_token, fencing_token,
    runtime_version, risk_version, event, observed_at,
    governed_result_id, risk_receipt_id
  ) values (
    p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
    v_runtime_version, v_risk_version, 'BOUND', v_now,
    p_governed_result_id, p_risk_receipt_id
  );

  return jsonb_build_object(
    'bound', true,
    'duplicate', false,
    'status', 'BOUND',
    'runtime_id', p_runtime_id,
    'cycle_id', p_cycle_id,
    'runtime_version', v_runtime_version,
    'risk_version', v_risk_version,
    'fencing_token', p_fencing_token,
    'risk_ledger_hash', p_risk_ledger_hash,
    'risk_receipt_id', p_risk_receipt_id,
    'governed_result_id', p_governed_result_id,
    'policy_fingerprint', p_policy_fingerprint
  );
end;
$$;

create or replace function public.brian_read_governed_cycle_binding(
  p_runtime_id text,
  p_cycle_id text
) returns jsonb
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  select case when b.runtime_id is null then null else jsonb_build_object(
    'runtime_id', b.runtime_id,
    'cycle_id', b.cycle_id,
    'governed_result_id', b.governed_result_id,
    'policy_fingerprint', b.policy_fingerprint,
    'risk_version', b.risk_version,
    'risk_ledger_hash', b.risk_ledger_hash,
    'risk_receipt_id', b.risk_receipt_id,
    'runtime_version_before', b.runtime_version_before,
    'fencing_token', b.fencing_token,
    'bound_at', b.bound_at,
    'shadow_only', b.shadow_only,
    'live_execution', b.live_execution
  ) end
  from (select p_runtime_id as runtime_id, p_cycle_id as cycle_id) q
  left join public.brian_governed_cycle_bindings b
    on b.runtime_id=q.runtime_id and b.cycle_id=q.cycle_id;
$$;

revoke all on function public.brian_bind_governed_shadow_cycle(
  text,text,bigint,bigint,bigint,text,text,text,text,text
) from public, anon, authenticated;
revoke all on function public.brian_read_governed_cycle_binding(text,text)
  from public, anon, authenticated;

grant execute on function public.brian_bind_governed_shadow_cycle(
  text,text,bigint,bigint,bigint,text,text,text,text,text
) to service_role;
grant execute on function public.brian_read_governed_cycle_binding(text,text)
  to service_role;

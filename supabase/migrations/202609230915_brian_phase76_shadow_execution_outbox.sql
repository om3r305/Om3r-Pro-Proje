-- Brian Phase 76: durable shadow execution outbox.
--
-- GitHub-only until explicit rollout.
--
-- Phase75 atomically authorizes a governed cycle and persists CYCLE_CREATED.
-- Phase76 defines the next durable boundary: before any paper execution begins,
-- that exact authorization must still match the current runtime/risk heads and
-- be recorded as an immutable SUBMITTED dispatch.
--
-- Risk changes after SUBMITTED are a post-submission cancel/kill-switch problem,
-- not an authorization race. No live exchange transport is introduced here.

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

create table if not exists public.brian_shadow_execution_dispatches (
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  governed_result_id text not null,
  policy_fingerprint text not null,
  authorization_checkpoint_id text not null,
  authorization_runtime_version bigint not null check (authorization_runtime_version > 0),
  risk_version bigint not null check (risk_version > 0),
  risk_ledger_hash text not null,
  risk_receipt_id text not null,
  fencing_token bigint not null check (fencing_token > 0),
  submitted_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, dispatch_id),
  unique (runtime_id, cycle_id),
  unique (runtime_id, governed_result_id),
  check (length(dispatch_id) = 64),
  check (length(cycle_id) = 64),
  check (length(governed_result_id) = 64),
  check (length(policy_fingerprint) = 64),
  check (length(authorization_checkpoint_id) = 64),
  check (length(risk_ledger_hash) = 64),
  check (length(risk_receipt_id) = 64)
);

create index if not exists brian_shadow_execution_dispatches_risk_idx
  on public.brian_shadow_execution_dispatches(
    runtime_id, risk_version, risk_receipt_id, submitted_at desc
  );

alter table public.brian_shadow_execution_dispatches enable row level security;
revoke all on public.brian_shadow_execution_dispatches
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_execution_dispatches_append_only
  on public.brian_shadow_execution_dispatches;
create trigger brian_shadow_execution_dispatches_append_only
  before update or delete on public.brian_shadow_execution_dispatches
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_shadow_execution_dispatch_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  cycle_id text not null,
  dispatch_id text not null,
  owner_token text not null,
  fencing_token bigint,
  runtime_version bigint,
  risk_version bigint,
  event text not null check (
    event in (
      'SUBMITTED',
      'DUPLICATE',
      'AUTHORIZATION_MISSING',
      'LEASE_LOST',
      'RUNTIME_VERSION_CONFLICT',
      'RISK_VERSION_CONFLICT'
    )
  ),
  observed_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_shadow_execution_dispatch_events_runtime_idx
  on public.brian_shadow_execution_dispatch_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_shadow_execution_dispatch_events enable row level security;
revoke all on public.brian_shadow_execution_dispatch_events
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_execution_dispatch_events_append_only
  on public.brian_shadow_execution_dispatch_events;
create trigger brian_shadow_execution_dispatch_events_append_only
  before update or delete on public.brian_shadow_execution_dispatch_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_submit_shadow_execution_dispatch(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_cycle_id text,
  p_dispatch_id text
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
  v_checkpoint_id text;
  v_lease_until timestamptz;
  v_risk_version bigint;
  v_risk_hash text;
  v_risk_head_entry_id text;
  v_risk_receipt_id text;
  v_auth public.brian_governed_cycle_authorizations%rowtype;
  v_existing public.brian_shadow_execution_dispatches%rowtype;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_cycle_id is null or length(p_cycle_id) <> 64
     or p_dispatch_id is null or length(p_dispatch_id) <> 64 then
    raise exception 'PHASE76_SUBMIT: valid runtime/owner/fence/cycle/dispatch required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, checkpoint_id, lease_until
    into v_owner, v_fence, v_runtime_version, v_checkpoint_id, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if not found
     or v_owner <> p_owner_token
     or v_fence <> p_fencing_token
     or v_lease_until <= v_now then
    insert into public.brian_shadow_execution_dispatch_events(
      runtime_id, cycle_id, dispatch_id, owner_token, fencing_token,
      runtime_version, event, observed_at
    ) values (
      p_runtime_id, p_cycle_id, p_dispatch_id, p_owner_token, p_fencing_token,
      v_runtime_version, 'LEASE_LOST', v_now
    );
    return jsonb_build_object(
      'submitted', false,
      'duplicate', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'dispatch_id', p_dispatch_id,
      'runtime_version', v_runtime_version,
      'fencing_token', v_fence
    );
  end if;

  select *
    into v_existing
  from public.brian_shadow_execution_dispatches
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if found then
    if v_existing.dispatch_id <> p_dispatch_id then
      raise exception 'PHASE76_DISPATCH_CONFLICT: cycle % already submitted as %',
        p_cycle_id, v_existing.dispatch_id;
    end if;

    insert into public.brian_shadow_execution_dispatch_events(
      runtime_id, cycle_id, dispatch_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, p_dispatch_id, p_owner_token, p_fencing_token,
      v_runtime_version, v_existing.risk_version, 'DUPLICATE', v_now,
      jsonb_build_object(
        'authorization_runtime_version', v_existing.authorization_runtime_version,
        'current_runtime_version', v_runtime_version
      )
    );

    return jsonb_build_object(
      'submitted', true,
      'duplicate', true,
      'status', case
        when v_runtime_version = v_existing.authorization_runtime_version
          then 'DUPLICATE_CURRENT'
        else 'DUPLICATE_HISTORICAL'
      end,
      'runtime_id', p_runtime_id,
      'cycle_id', v_existing.cycle_id,
      'dispatch_id', v_existing.dispatch_id,
      'governed_result_id', v_existing.governed_result_id,
      'policy_fingerprint', v_existing.policy_fingerprint,
      'authorization_checkpoint_id', v_existing.authorization_checkpoint_id,
      'authorization_runtime_version', v_existing.authorization_runtime_version,
      'current_runtime_version', v_runtime_version,
      'risk_version', v_existing.risk_version,
      'risk_ledger_hash', v_existing.risk_ledger_hash,
      'risk_receipt_id', v_existing.risk_receipt_id,
      'fencing_token', v_existing.fencing_token
    );
  end if;

  select *
    into v_auth
  from public.brian_governed_cycle_authorizations
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if not found then
    insert into public.brian_shadow_execution_dispatch_events(
      runtime_id, cycle_id, dispatch_id, owner_token, fencing_token,
      runtime_version, event, observed_at
    ) values (
      p_runtime_id, p_cycle_id, p_dispatch_id, p_owner_token, p_fencing_token,
      v_runtime_version, 'AUTHORIZATION_MISSING', v_now
    );
    return jsonb_build_object(
      'submitted', false,
      'duplicate', false,
      'status', 'AUTHORIZATION_MISSING',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'dispatch_id', p_dispatch_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  if v_runtime_version <> v_auth.runtime_version_after
     or v_checkpoint_id is distinct from v_auth.write_ahead_checkpoint_id then
    insert into public.brian_shadow_execution_dispatch_events(
      runtime_id, cycle_id, dispatch_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, p_dispatch_id, p_owner_token, p_fencing_token,
      v_runtime_version, v_auth.risk_version,
      'RUNTIME_VERSION_CONFLICT', v_now,
      jsonb_build_object(
        'authorization_runtime_version', v_auth.runtime_version_after,
        'authorization_checkpoint_id', v_auth.write_ahead_checkpoint_id,
        'current_checkpoint_id', v_checkpoint_id
      )
    );
    return jsonb_build_object(
      'submitted', false,
      'duplicate', false,
      'status', 'RUNTIME_VERSION_CONFLICT',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'dispatch_id', p_dispatch_id,
      'runtime_version', v_runtime_version,
      'authorization_runtime_version', v_auth.runtime_version_after,
      'fencing_token', p_fencing_token
    );
  end if;

  select version, ledger_hash, head_entry_id
    into v_risk_version, v_risk_hash, v_risk_head_entry_id
  from public.brian_operational_risk_heads
  where runtime_id = p_runtime_id
  for update;

  select receipt_id
    into v_risk_receipt_id
  from public.brian_operational_risk_entries
  where runtime_id = p_runtime_id
    and entry_id = v_risk_head_entry_id;

  if v_risk_version is null
     or v_risk_version <> v_auth.risk_version
     or v_risk_hash is distinct from v_auth.risk_ledger_hash
     or v_risk_receipt_id is distinct from v_auth.risk_receipt_id then
    insert into public.brian_shadow_execution_dispatch_events(
      runtime_id, cycle_id, dispatch_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, p_dispatch_id, p_owner_token, p_fencing_token,
      v_runtime_version, v_risk_version,
      'RISK_VERSION_CONFLICT', v_now,
      jsonb_build_object(
        'authorization_risk_version', v_auth.risk_version,
        'authorization_risk_hash', v_auth.risk_ledger_hash,
        'authorization_risk_receipt_id', v_auth.risk_receipt_id,
        'current_risk_hash', v_risk_hash,
        'current_risk_receipt_id', v_risk_receipt_id
      )
    );
    return jsonb_build_object(
      'submitted', false,
      'duplicate', false,
      'status', 'RISK_VERSION_CONFLICT',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'dispatch_id', p_dispatch_id,
      'runtime_version', v_runtime_version,
      'risk_version', v_risk_version,
      'fencing_token', p_fencing_token
    );
  end if;

  insert into public.brian_shadow_execution_dispatches(
    runtime_id, dispatch_id, cycle_id,
    governed_result_id, policy_fingerprint,
    authorization_checkpoint_id, authorization_runtime_version,
    risk_version, risk_ledger_hash, risk_receipt_id,
    fencing_token, submitted_at
  ) values (
    p_runtime_id, p_dispatch_id, p_cycle_id,
    v_auth.governed_result_id, v_auth.policy_fingerprint,
    v_auth.write_ahead_checkpoint_id, v_auth.runtime_version_after,
    v_auth.risk_version, v_auth.risk_ledger_hash, v_auth.risk_receipt_id,
    p_fencing_token, v_now
  );

  insert into public.brian_shadow_execution_dispatch_events(
    runtime_id, cycle_id, dispatch_id, owner_token, fencing_token,
    runtime_version, risk_version, event, observed_at,
    metadata
  ) values (
    p_runtime_id, p_cycle_id, p_dispatch_id, p_owner_token, p_fencing_token,
    v_runtime_version, v_auth.risk_version, 'SUBMITTED', v_now,
    jsonb_build_object(
      'authorization_checkpoint_id', v_auth.write_ahead_checkpoint_id,
      'governed_result_id', v_auth.governed_result_id,
      'policy_fingerprint', v_auth.policy_fingerprint
    )
  );

  return jsonb_build_object(
    'submitted', true,
    'duplicate', false,
    'status', 'SUBMITTED',
    'runtime_id', p_runtime_id,
    'cycle_id', p_cycle_id,
    'dispatch_id', p_dispatch_id,
    'governed_result_id', v_auth.governed_result_id,
    'policy_fingerprint', v_auth.policy_fingerprint,
    'authorization_checkpoint_id', v_auth.write_ahead_checkpoint_id,
    'authorization_runtime_version', v_auth.runtime_version_after,
    'current_runtime_version', v_runtime_version,
    'risk_version', v_auth.risk_version,
    'risk_ledger_hash', v_auth.risk_ledger_hash,
    'risk_receipt_id', v_auth.risk_receipt_id,
    'fencing_token', p_fencing_token
  );
end;
$$;

create or replace function public.brian_read_shadow_execution_dispatch(
  p_runtime_id text,
  p_cycle_id text
) returns jsonb
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  select case when d.runtime_id is null then null else jsonb_build_object(
    'runtime_id', d.runtime_id,
    'dispatch_id', d.dispatch_id,
    'cycle_id', d.cycle_id,
    'governed_result_id', d.governed_result_id,
    'policy_fingerprint', d.policy_fingerprint,
    'authorization_checkpoint_id', d.authorization_checkpoint_id,
    'authorization_runtime_version', d.authorization_runtime_version,
    'risk_version', d.risk_version,
    'risk_ledger_hash', d.risk_ledger_hash,
    'risk_receipt_id', d.risk_receipt_id,
    'fencing_token', d.fencing_token,
    'submitted_at', d.submitted_at,
    'shadow_only', d.shadow_only,
    'live_execution', d.live_execution
  ) end
  from (select p_runtime_id as runtime_id, p_cycle_id as cycle_id) q
  left join public.brian_shadow_execution_dispatches d
    on d.runtime_id=q.runtime_id and d.cycle_id=q.cycle_id;
$$;

revoke all on function public.brian_submit_shadow_execution_dispatch(
  text,text,bigint,text,text
) from public, anon, authenticated;
revoke all on function public.brian_read_shadow_execution_dispatch(text,text)
  from public, anon, authenticated;

grant execute on function public.brian_submit_shadow_execution_dispatch(
  text,text,bigint,text,text
) to service_role;
grant execute on function public.brian_read_shadow_execution_dispatch(text,text)
  to service_role;

-- Brian Phase 75: atomic governed authorization + durable write-ahead.
--
-- GitHub-only until explicit rollout.
--
-- Phase74 proves a governed cycle is bound to the current persisted risk head.
-- Phase71 proves a cycle body is durably journaled before paper side effects.
-- Phase75 removes the transaction gap between those two proofs: risk-head
-- validation, immutable authorization, and Phase70 CYCLE_CREATED checkpoint
-- persistence happen under the same runtime advisory lock / transaction.
--
-- This remains shadow/paper-only. No exchange transport is introduced.

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

create table if not exists public.brian_governed_cycle_authorizations (
  runtime_id text not null,
  cycle_id text not null,
  governed_result_id text not null,
  policy_fingerprint text not null,
  risk_version bigint not null check (risk_version > 0),
  risk_ledger_hash text not null,
  risk_receipt_id text not null,
  runtime_version_before bigint not null check (runtime_version_before > 0),
  runtime_version_after bigint not null check (runtime_version_after > runtime_version_before),
  write_ahead_checkpoint_id text not null,
  fencing_token bigint not null check (fencing_token > 0),
  authorized_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, cycle_id),
  unique (runtime_id, governed_result_id),
  unique (runtime_id, write_ahead_checkpoint_id),
  check (length(cycle_id) = 64),
  check (length(governed_result_id) = 64),
  check (length(policy_fingerprint) = 64),
  check (length(risk_ledger_hash) = 64),
  check (length(risk_receipt_id) = 64),
  check (length(write_ahead_checkpoint_id) = 64)
);

create index if not exists brian_governed_cycle_authorizations_risk_idx
  on public.brian_governed_cycle_authorizations(
    runtime_id, risk_version, risk_receipt_id, authorized_at desc
  );

alter table public.brian_governed_cycle_authorizations enable row level security;
revoke all on public.brian_governed_cycle_authorizations
  from anon, authenticated, service_role;

drop trigger if exists brian_governed_cycle_authorizations_append_only
  on public.brian_governed_cycle_authorizations;
create trigger brian_governed_cycle_authorizations_append_only
  before update or delete on public.brian_governed_cycle_authorizations
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_governed_cycle_authorization_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  cycle_id text not null,
  owner_token text not null,
  fencing_token bigint,
  runtime_version bigint,
  risk_version bigint,
  event text not null check (
    event in (
      'AUTHORIZED_AND_PERSISTED',
      'DUPLICATE',
      'LEASE_LOST',
      'RUNTIME_VERSION_CONFLICT',
      'RISK_VERSION_CONFLICT',
      'CHECKPOINT_REJECTED'
    )
  ),
  observed_at timestamptz not null default now(),
  governed_result_id text,
  risk_receipt_id text,
  checkpoint_id text,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_governed_cycle_authorization_events_runtime_idx
  on public.brian_governed_cycle_authorization_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_governed_cycle_authorization_events enable row level security;
revoke all on public.brian_governed_cycle_authorization_events
  from anon, authenticated, service_role;

drop trigger if exists brian_governed_cycle_authorization_events_append_only
  on public.brian_governed_cycle_authorization_events;
create trigger brian_governed_cycle_authorization_events_append_only
  before update or delete on public.brian_governed_cycle_authorization_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_authorize_and_persist_governed_cycle(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_expected_runtime_version bigint,
  p_risk_version bigint,
  p_risk_ledger_hash text,
  p_risk_receipt_id text,
  p_cycle_id text,
  p_governed_result_id text,
  p_policy_fingerprint text,
  p_checkpoint jsonb
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
  v_current_risk_version bigint;
  v_current_risk_hash text;
  v_risk_head_entry_id text;
  v_current_receipt_id text;
  v_checkpoint_id text;
  v_checkpoint_cycle jsonb;
  v_last_cycle_stage text;
  v_last_cycle_hash text;
  v_checkpoint_commit jsonb;
  v_commit_status text;
  v_runtime_version_after bigint;
  v_existing public.brian_governed_cycle_authorizations%rowtype;
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
    raise exception 'PHASE75_AUTHORIZE: valid runtime/fence/version/hash identities required';
  end if;
  if p_checkpoint is null or jsonb_typeof(p_checkpoint) <> 'object' then
    raise exception 'PHASE75_AUTHORIZE: checkpoint object required';
  end if;
  if coalesce((p_checkpoint->>'live_execution')::boolean, false) then
    raise exception 'PHASE75_AUTHORIZE: live checkpoint rejected';
  end if;

  v_checkpoint_id := nullif(trim(p_checkpoint->>'checkpoint_id'), '');
  if v_checkpoint_id is null or length(v_checkpoint_id) <> 64 then
    raise exception 'PHASE75_AUTHORIZE: checkpoint_id must be a content hash';
  end if;

  v_checkpoint_cycle :=
    p_checkpoint->'journal_manifest'->'cycles'->p_cycle_id;
  if v_checkpoint_cycle is null
     or jsonb_typeof(v_checkpoint_cycle) <> 'object'
     or nullif(trim(v_checkpoint_cycle->>'cycle_id'), '') is distinct from p_cycle_id
     or coalesce((v_checkpoint_cycle->>'shadow_only')::boolean, false) is not true
     or coalesce((v_checkpoint_cycle->>'live_execution')::boolean, false) is not false
     or coalesce((v_checkpoint_cycle->>'account_state_mutated')::boolean, false) is not false then
    raise exception 'PHASE75_AUTHORIZE: checkpoint does not contain the shadow cycle body';
  end if;

  select e.value->>'stage', e.value->>'cycle_hash'
    into v_last_cycle_stage, v_last_cycle_hash
  from jsonb_array_elements(
    coalesce(p_checkpoint->'journal_manifest'->'entries', '[]'::jsonb)
  ) with ordinality e(value, ord)
  where e.value->>'cycle_id' = p_cycle_id
  order by e.ord desc
  limit 1;

  if v_last_cycle_stage is distinct from 'CYCLE_CREATED'
     or v_last_cycle_hash is null
     or length(v_last_cycle_hash) <> 64 then
    raise exception 'PHASE75_AUTHORIZE: write-ahead checkpoint must stop at CYCLE_CREATED';
  end if;

  -- Phase71 write-ahead happens before Phase60 pending state is created.
  if nullif(trim(p_checkpoint->'runtime_checkpoint'->>'pending_cycle_id'), '') is not null
     or nullif(trim(
       p_checkpoint->'runtime_checkpoint'->'shadow_ledger_manifest'->>'pending_cycle_id'
     ), '') is not null then
    raise exception 'PHASE75_AUTHORIZE: write-ahead checkpoint unexpectedly has ledger pending state';
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
    insert into public.brian_governed_cycle_authorization_events(
      runtime_id, cycle_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      governed_result_id, risk_receipt_id, checkpoint_id
    ) values (
      p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
      v_runtime_version, p_risk_version, 'LEASE_LOST', v_now,
      p_governed_result_id, p_risk_receipt_id, v_checkpoint_id
    );
    return jsonb_build_object(
      'authorized', false,
      'duplicate', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'risk_version', p_risk_version,
      'fencing_token', v_fence,
      'checkpoint_id', v_checkpoint_id
    );
  end if;

  -- Lost-response retry: if the exact authorization already exists, validate
  -- immutable anchors before considering the now-advanced runtime version.
  select *
    into v_existing
  from public.brian_governed_cycle_authorizations
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if found then
    if v_existing.governed_result_id <> p_governed_result_id
       or v_existing.policy_fingerprint <> p_policy_fingerprint
       or v_existing.risk_version <> p_risk_version
       or v_existing.risk_ledger_hash <> p_risk_ledger_hash
       or v_existing.risk_receipt_id <> p_risk_receipt_id
       or v_existing.runtime_version_before <> p_expected_runtime_version
       or v_existing.write_ahead_checkpoint_id <> v_checkpoint_id then
      raise exception 'PHASE75_AUTHORIZATION_CONFLICT: cycle % already authorized differently',
        p_cycle_id;
    end if;

    if v_runtime_version < v_existing.runtime_version_after then
      raise exception 'PHASE75_AUTHORIZATION_HEAD_REGRESSION: runtime head % < authorization %',
        v_runtime_version, v_existing.runtime_version_after;
    end if;

    insert into public.brian_governed_cycle_authorization_events(
      runtime_id, cycle_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      governed_result_id, risk_receipt_id, checkpoint_id,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
      v_runtime_version, p_risk_version, 'DUPLICATE', v_now,
      p_governed_result_id, p_risk_receipt_id, v_checkpoint_id,
      jsonb_build_object(
        'authorized_runtime_version_after', v_existing.runtime_version_after
      )
    );

    return jsonb_build_object(
      'authorized', true,
      'duplicate', true,
      'status', case
        when v_runtime_version = v_existing.runtime_version_after
          then 'DUPLICATE_CURRENT'
        else 'DUPLICATE_HISTORICAL'
      end,
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version_before', v_existing.runtime_version_before,
      'runtime_version_after', v_existing.runtime_version_after,
      'current_runtime_version', v_runtime_version,
      'risk_version', v_existing.risk_version,
      'fencing_token', p_fencing_token,
      'checkpoint_id', v_existing.write_ahead_checkpoint_id,
      'risk_ledger_hash', v_existing.risk_ledger_hash,
      'risk_receipt_id', v_existing.risk_receipt_id,
      'governed_result_id', v_existing.governed_result_id,
      'policy_fingerprint', v_existing.policy_fingerprint
    );
  end if;

  if v_runtime_version <> p_expected_runtime_version then
    insert into public.brian_governed_cycle_authorization_events(
      runtime_id, cycle_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      governed_result_id, risk_receipt_id, checkpoint_id,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
      v_runtime_version, p_risk_version, 'RUNTIME_VERSION_CONFLICT', v_now,
      p_governed_result_id, p_risk_receipt_id, v_checkpoint_id,
      jsonb_build_object('expected_runtime_version', p_expected_runtime_version)
    );
    return jsonb_build_object(
      'authorized', false,
      'duplicate', false,
      'status', 'RUNTIME_VERSION_CONFLICT',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'risk_version', p_risk_version,
      'fencing_token', p_fencing_token,
      'checkpoint_id', v_checkpoint_id
    );
  end if;

  select version, ledger_hash, head_entry_id
    into v_current_risk_version, v_current_risk_hash, v_risk_head_entry_id
  from public.brian_operational_risk_heads
  where runtime_id = p_runtime_id
  for update;

  select receipt_id
    into v_current_receipt_id
  from public.brian_operational_risk_entries
  where runtime_id = p_runtime_id
    and entry_id = v_risk_head_entry_id;

  if v_current_risk_version is null
     or v_current_risk_version <> p_risk_version
     or v_current_risk_hash is distinct from p_risk_ledger_hash
     or v_current_receipt_id is distinct from p_risk_receipt_id then
    insert into public.brian_governed_cycle_authorization_events(
      runtime_id, cycle_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      governed_result_id, risk_receipt_id, checkpoint_id,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
      v_runtime_version, v_current_risk_version, 'RISK_VERSION_CONFLICT', v_now,
      p_governed_result_id, p_risk_receipt_id, v_checkpoint_id,
      jsonb_build_object(
        'expected_risk_version', p_risk_version,
        'expected_risk_ledger_hash', p_risk_ledger_hash,
        'current_risk_ledger_hash', v_current_risk_hash,
        'expected_risk_receipt_id', p_risk_receipt_id,
        'current_risk_receipt_id', v_current_receipt_id
      )
    );
    return jsonb_build_object(
      'authorized', false,
      'duplicate', false,
      'status', 'RISK_VERSION_CONFLICT',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'risk_version', v_current_risk_version,
      'fencing_token', p_fencing_token,
      'checkpoint_id', v_checkpoint_id
    );
  end if;

  -- Call the already-tested Phase70 commit inside this same transaction.
  -- pg_advisory_xact_lock is transaction-scoped/reentrant for this session, so
  -- risk head cannot advance between validation and write-ahead persistence.
  select public.brian_commit_shadow_runtime_checkpoint(
    p_runtime_id,
    p_owner_token,
    p_fencing_token,
    p_expected_runtime_version,
    p_checkpoint
  ) into v_checkpoint_commit;

  v_commit_status := v_checkpoint_commit->>'status';
  if v_commit_status not in ('COMMITTED', 'DUPLICATE_CURRENT') then
    insert into public.brian_governed_cycle_authorization_events(
      runtime_id, cycle_id, owner_token, fencing_token,
      runtime_version, risk_version, event, observed_at,
      governed_result_id, risk_receipt_id, checkpoint_id,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
      v_runtime_version, v_current_risk_version, 'CHECKPOINT_REJECTED', v_now,
      p_governed_result_id, p_risk_receipt_id, v_checkpoint_id,
      jsonb_build_object('checkpoint_status', v_commit_status)
    );
    raise exception 'PHASE75_CHECKPOINT_REJECTED:%', coalesce(v_commit_status, '<null>');
  end if;

  v_runtime_version_after := (v_checkpoint_commit->>'version')::bigint;
  if v_runtime_version_after <= p_expected_runtime_version then
    raise exception 'PHASE75_RUNTIME_VERSION_NOT_ADVANCED: before %, after %',
      p_expected_runtime_version, v_runtime_version_after;
  end if;

  insert into public.brian_governed_cycle_authorizations(
    runtime_id, cycle_id, governed_result_id, policy_fingerprint,
    risk_version, risk_ledger_hash, risk_receipt_id,
    runtime_version_before, runtime_version_after,
    write_ahead_checkpoint_id, fencing_token, authorized_at
  ) values (
    p_runtime_id, p_cycle_id, p_governed_result_id, p_policy_fingerprint,
    p_risk_version, p_risk_ledger_hash, p_risk_receipt_id,
    p_expected_runtime_version, v_runtime_version_after,
    v_checkpoint_id, p_fencing_token, v_now
  );

  insert into public.brian_governed_cycle_authorization_events(
    runtime_id, cycle_id, owner_token, fencing_token,
    runtime_version, risk_version, event, observed_at,
    governed_result_id, risk_receipt_id, checkpoint_id,
    metadata
  ) values (
    p_runtime_id, p_cycle_id, p_owner_token, p_fencing_token,
    v_runtime_version_after, p_risk_version,
    'AUTHORIZED_AND_PERSISTED', v_now,
    p_governed_result_id, p_risk_receipt_id, v_checkpoint_id,
    jsonb_build_object(
      'runtime_version_before', p_expected_runtime_version,
      'runtime_version_after', v_runtime_version_after,
      'checkpoint_status', v_commit_status
    )
  );

  return jsonb_build_object(
    'authorized', true,
    'duplicate', false,
    'status', 'AUTHORIZED_AND_PERSISTED',
    'runtime_id', p_runtime_id,
    'cycle_id', p_cycle_id,
    'runtime_version_before', p_expected_runtime_version,
    'runtime_version_after', v_runtime_version_after,
    'current_runtime_version', v_runtime_version_after,
    'risk_version', p_risk_version,
    'fencing_token', p_fencing_token,
    'checkpoint_id', v_checkpoint_id,
    'risk_ledger_hash', p_risk_ledger_hash,
    'risk_receipt_id', p_risk_receipt_id,
    'governed_result_id', p_governed_result_id,
    'policy_fingerprint', p_policy_fingerprint
  );
end;
$$;

create or replace function public.brian_read_governed_cycle_authorization(
  p_runtime_id text,
  p_cycle_id text
) returns jsonb
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  select case when a.runtime_id is null then null else jsonb_build_object(
    'runtime_id', a.runtime_id,
    'cycle_id', a.cycle_id,
    'governed_result_id', a.governed_result_id,
    'policy_fingerprint', a.policy_fingerprint,
    'risk_version', a.risk_version,
    'risk_ledger_hash', a.risk_ledger_hash,
    'risk_receipt_id', a.risk_receipt_id,
    'runtime_version_before', a.runtime_version_before,
    'runtime_version_after', a.runtime_version_after,
    'write_ahead_checkpoint_id', a.write_ahead_checkpoint_id,
    'fencing_token', a.fencing_token,
    'authorized_at', a.authorized_at,
    'shadow_only', a.shadow_only,
    'live_execution', a.live_execution
  ) end
  from (select p_runtime_id as runtime_id, p_cycle_id as cycle_id) q
  left join public.brian_governed_cycle_authorizations a
    on a.runtime_id=q.runtime_id and a.cycle_id=q.cycle_id;
$$;

revoke all on function public.brian_authorize_and_persist_governed_cycle(
  text,text,bigint,bigint,bigint,text,text,text,text,text,jsonb
) from public, anon, authenticated;
revoke all on function public.brian_read_governed_cycle_authorization(text,text)
  from public, anon, authenticated;

grant execute on function public.brian_authorize_and_persist_governed_cycle(
  text,text,bigint,bigint,bigint,text,text,text,text,text,jsonb
) to service_role;
grant execute on function public.brian_read_governed_cycle_authorization(text,text)
  to service_role;

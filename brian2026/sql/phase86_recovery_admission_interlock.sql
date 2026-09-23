-- Brian Phase 86 DRAFT SQL: unresolved-recovery admission interlock.
--
-- IMPORTANT: draft/undeployed SQL. At rollout freeze create the official
-- Supabase migration with `supabase migration new`, then rerun the complete
-- Postgres suite before deployment.
--
-- Once a Phase78 AFTER_START cancel exists, normal new governed cycles must not
-- enter the durable execution path until the recovery obligation is resolved.
-- Phase86 wraps the existing Phase75 authorization and Phase76 dispatch RPCs
-- under the same runtime advisory lock. Exact duplicate reads remain allowed.
--
-- A Phase81 NO_RECOVERY_REQUIRED directive resolves the barrier without a
-- recovery cycle. All other AFTER_START obligations remain blocking until an
-- immutable Phase85 completion certificate exists.

create table if not exists public.brian_shadow_recovery_admission_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  candidate_cycle_id text not null,
  stage text not null check (stage in ('AUTHORIZATION','DISPATCH')),
  event text not null check (
    event in ('RECOVERY_BARRIER')
  ),
  barrier_original_cycle_id text not null,
  barrier_cancel_risk_receipt_id text not null,
  barrier_reason text not null,
  observed_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  check (length(candidate_cycle_id) = 64),
  check (length(barrier_original_cycle_id) = 64),
  check (length(barrier_cancel_risk_receipt_id) = 64)
);

create index if not exists brian_shadow_recovery_admission_events_runtime_idx
  on public.brian_shadow_recovery_admission_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_shadow_recovery_admission_events enable row level security;
revoke all on public.brian_shadow_recovery_admission_events
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_recovery_admission_events_append_only
  on public.brian_shadow_recovery_admission_events;
create trigger brian_shadow_recovery_admission_events_append_only
  before update or delete on public.brian_shadow_recovery_admission_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_read_shadow_recovery_admission(
  p_runtime_id text
) returns jsonb
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  with unresolved as (
    select
      c.cycle_id as original_cycle_id,
      c.risk_receipt_id as cancel_risk_receipt_id,
      c.reason,
      c.requested_at
    from public.brian_shadow_execution_cancel_requests c
    where c.runtime_id = p_runtime_id
      and c.phase = 'AFTER_START'
      and not exists (
        select 1
        from public.brian_shadow_recovery_completion_certificates cert
        where cert.runtime_id = c.runtime_id
          and cert.original_cycle_id = c.cycle_id
      )
      and not exists (
        select 1
        from public.brian_shadow_cancel_recovery_directives d
        where d.runtime_id = c.runtime_id
          and d.cycle_id = c.cycle_id
          and d.recovery_status = 'NO_RECOVERY_REQUIRED'
      )
    order by c.requested_at asc, c.risk_version asc, c.risk_receipt_id asc
    limit 1
  )
  select case
    when u.original_cycle_id is null then jsonb_build_object(
      'blocked', false,
      'status', 'OPEN',
      'runtime_id', p_runtime_id
    )
    else jsonb_build_object(
      'blocked', true,
      'status', 'RECOVERY_BARRIER',
      'runtime_id', p_runtime_id,
      'original_cycle_id', u.original_cycle_id,
      'cancel_risk_receipt_id', u.cancel_risk_receipt_id,
      'reason', u.reason,
      'requested_at', u.requested_at
    )
  end
  from (select 1) seed
  left join unresolved u on true;
$$;

-- Preserve the Phase75 implementation behind a private base name exactly once.
do $$
begin
  if pg_catalog.to_regprocedure(
    'public.brian_authorize_and_persist_governed_cycle_phase75_base(text,text,bigint,bigint,bigint,text,text,text,text,text,jsonb)'
  ) is null then
    alter function public.brian_authorize_and_persist_governed_cycle(
      text,text,bigint,bigint,bigint,text,text,text,text,text,jsonb
    ) rename to brian_authorize_and_persist_governed_cycle_phase75_base;
  end if;
end
$$;

revoke all on function public.brian_authorize_and_persist_governed_cycle_phase75_base(
  text,text,bigint,bigint,bigint,text,text,text,text,text,jsonb
) from public, anon, authenticated, service_role;

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
  v_barrier jsonb;
  v_existing boolean := false;
  v_owner text;
  v_current_fence bigint;
  v_runtime_version bigint;
  v_lease_until timestamptz;
  v_checkpoint_id text;
begin
  -- Let the original Phase75 validator own malformed-input semantics.
  if nullif(trim(p_runtime_id),'') is null
     or p_cycle_id is null or length(p_cycle_id) <> 64 then
    return public.brian_authorize_and_persist_governed_cycle_phase75_base(
      p_runtime_id,p_owner_token,p_fencing_token,p_expected_runtime_version,
      p_risk_version,p_risk_ledger_hash,p_risk_receipt_id,p_cycle_id,
      p_governed_result_id,p_policy_fingerprint,p_checkpoint
    );
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );

  select owner_token,fencing_token,version,lease_until
    into v_owner,v_current_fence,v_runtime_version,v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id=p_runtime_id
  for update;

  -- Preserve Phase75 lease/version precedence.
  if v_owner is null
     or v_owner <> p_owner_token
     or v_current_fence <> p_fencing_token
     or v_lease_until <= clock_timestamp()
     or v_runtime_version <> p_expected_runtime_version then
    return public.brian_authorize_and_persist_governed_cycle_phase75_base(
      p_runtime_id,p_owner_token,p_fencing_token,p_expected_runtime_version,
      p_risk_version,p_risk_ledger_hash,p_risk_receipt_id,p_cycle_id,
      p_governed_result_id,p_policy_fingerprint,p_checkpoint
    );
  end if;

  select exists(
    select 1 from public.brian_governed_cycle_authorizations a
    where a.runtime_id=p_runtime_id and a.cycle_id=p_cycle_id
  ) into v_existing;

  -- Exact lost-response retries remain readable after a later recovery barrier.
  if v_existing then
    return public.brian_authorize_and_persist_governed_cycle_phase75_base(
      p_runtime_id,p_owner_token,p_fencing_token,p_expected_runtime_version,
      p_risk_version,p_risk_ledger_hash,p_risk_receipt_id,p_cycle_id,
      p_governed_result_id,p_policy_fingerprint,p_checkpoint
    );
  end if;

  v_barrier := public.brian_read_shadow_recovery_admission(p_runtime_id);
  if coalesce((v_barrier->>'blocked')::boolean,false) then
    v_checkpoint_id := nullif(trim(p_checkpoint->>'checkpoint_id'),'');

    insert into public.brian_shadow_recovery_admission_events(
      runtime_id,candidate_cycle_id,stage,event,
      barrier_original_cycle_id,barrier_cancel_risk_receipt_id,
      barrier_reason,metadata
    ) values (
      p_runtime_id,p_cycle_id,'AUTHORIZATION','RECOVERY_BARRIER',
      v_barrier->>'original_cycle_id',
      v_barrier->>'cancel_risk_receipt_id',
      coalesce(v_barrier->>'reason','UNKNOWN'),
      jsonb_build_object(
        'runtime_version',v_runtime_version,
        'candidate_checkpoint_id',v_checkpoint_id,
        'candidate_risk_version',p_risk_version
      )
    );

    return jsonb_build_object(
      'authorized',false,
      'duplicate',false,
      'status','RECOVERY_BARRIER',
      'runtime_id',p_runtime_id,
      'cycle_id',p_cycle_id,
      'runtime_version',v_runtime_version,
      'risk_version',p_risk_version,
      'fencing_token',p_fencing_token,
      'checkpoint_id',v_checkpoint_id,
      'barrier_original_cycle_id',v_barrier->>'original_cycle_id',
      'barrier_cancel_risk_receipt_id',v_barrier->>'cancel_risk_receipt_id',
      'barrier_reason',v_barrier->>'reason'
    );
  end if;

  return public.brian_authorize_and_persist_governed_cycle_phase75_base(
    p_runtime_id,p_owner_token,p_fencing_token,p_expected_runtime_version,
    p_risk_version,p_risk_ledger_hash,p_risk_receipt_id,p_cycle_id,
    p_governed_result_id,p_policy_fingerprint,p_checkpoint
  );
end;
$$;

revoke all on function public.brian_authorize_and_persist_governed_cycle(
  text,text,bigint,bigint,bigint,text,text,text,text,text,jsonb
) from public, anon, authenticated;
grant execute on function public.brian_authorize_and_persist_governed_cycle(
  text,text,bigint,bigint,bigint,text,text,text,text,text,jsonb
) to service_role;

-- Preserve the Phase76 implementation behind a private base name exactly once.
do $$
begin
  if pg_catalog.to_regprocedure(
    'public.brian_submit_shadow_execution_dispatch_phase76_base(text,text,bigint,text,text)'
  ) is null then
    alter function public.brian_submit_shadow_execution_dispatch(
      text,text,bigint,text,text
    ) rename to brian_submit_shadow_execution_dispatch_phase76_base;
  end if;
end
$$;

revoke all on function public.brian_submit_shadow_execution_dispatch_phase76_base(
  text,text,bigint,text,text
) from public, anon, authenticated, service_role;

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
  v_barrier jsonb;
  v_existing boolean := false;
  v_auth public.brian_governed_cycle_authorizations%rowtype;
  v_owner text;
  v_current_fence bigint;
  v_runtime_version bigint;
  v_checkpoint_id text;
  v_lease_until timestamptz;
begin
  if nullif(trim(p_runtime_id),'') is null
     or p_cycle_id is null or length(p_cycle_id) <> 64 then
    return public.brian_submit_shadow_execution_dispatch_phase76_base(
      p_runtime_id,p_owner_token,p_fencing_token,p_cycle_id,p_dispatch_id
    );
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );

  select owner_token,fencing_token,version,checkpoint_id,lease_until
    into v_owner,v_current_fence,v_runtime_version,v_checkpoint_id,v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id=p_runtime_id
  for update;

  if v_owner is null
     or v_owner <> p_owner_token
     or v_current_fence <> p_fencing_token
     or v_lease_until <= clock_timestamp() then
    return public.brian_submit_shadow_execution_dispatch_phase76_base(
      p_runtime_id,p_owner_token,p_fencing_token,p_cycle_id,p_dispatch_id
    );
  end if;

  select exists(
    select 1 from public.brian_shadow_execution_dispatches d
    where d.runtime_id=p_runtime_id and d.cycle_id=p_cycle_id
  ) into v_existing;

  if v_existing then
    return public.brian_submit_shadow_execution_dispatch_phase76_base(
      p_runtime_id,p_owner_token,p_fencing_token,p_cycle_id,p_dispatch_id
    );
  end if;

  select *
    into v_auth
  from public.brian_governed_cycle_authorizations
  where runtime_id=p_runtime_id and cycle_id=p_cycle_id;

  -- Preserve Phase76 missing/stale-authorization precedence.
  if not found
     or v_runtime_version <> v_auth.runtime_version_after
     or v_checkpoint_id is distinct from v_auth.write_ahead_checkpoint_id then
    return public.brian_submit_shadow_execution_dispatch_phase76_base(
      p_runtime_id,p_owner_token,p_fencing_token,p_cycle_id,p_dispatch_id
    );
  end if;

  v_barrier := public.brian_read_shadow_recovery_admission(p_runtime_id);
  if coalesce((v_barrier->>'blocked')::boolean,false) then
    insert into public.brian_shadow_recovery_admission_events(
      runtime_id,candidate_cycle_id,stage,event,
      barrier_original_cycle_id,barrier_cancel_risk_receipt_id,
      barrier_reason,metadata
    ) values (
      p_runtime_id,p_cycle_id,'DISPATCH','RECOVERY_BARRIER',
      v_barrier->>'original_cycle_id',
      v_barrier->>'cancel_risk_receipt_id',
      coalesce(v_barrier->>'reason','UNKNOWN'),
      jsonb_build_object(
        'runtime_version',v_runtime_version,
        'authorization_runtime_version',v_auth.runtime_version_after,
        'authorization_checkpoint_id',v_auth.write_ahead_checkpoint_id,
        'dispatch_id',p_dispatch_id
      )
    );

    return jsonb_build_object(
      'submitted',false,
      'duplicate',false,
      'status','RECOVERY_BARRIER',
      'runtime_id',p_runtime_id,
      'cycle_id',p_cycle_id,
      'dispatch_id',p_dispatch_id,
      'runtime_version',v_runtime_version,
      'current_runtime_version',v_runtime_version,
      'authorization_runtime_version',v_auth.runtime_version_after,
      'risk_version',v_auth.risk_version,
      'fencing_token',p_fencing_token,
      'barrier_original_cycle_id',v_barrier->>'original_cycle_id',
      'barrier_cancel_risk_receipt_id',v_barrier->>'cancel_risk_receipt_id',
      'barrier_reason',v_barrier->>'reason'
    );
  end if;

  return public.brian_submit_shadow_execution_dispatch_phase76_base(
    p_runtime_id,p_owner_token,p_fencing_token,p_cycle_id,p_dispatch_id
  );
end;
$$;

revoke all on function public.brian_submit_shadow_execution_dispatch(
  text,text,bigint,text,text
) from public, anon, authenticated;
grant execute on function public.brian_submit_shadow_execution_dispatch(
  text,text,bigint,text,text
) to service_role;

revoke all on function public.brian_read_shadow_recovery_admission(text)
  from public, anon, authenticated;
grant execute on function public.brian_read_shadow_recovery_admission(text)
  to service_role;

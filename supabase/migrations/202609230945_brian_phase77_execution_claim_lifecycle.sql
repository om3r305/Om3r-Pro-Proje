-- Brian Phase 77: durable execution claim / pre-execution kill-switch lifecycle.
--
-- GitHub-only until explicit rollout.
--
-- A Phase76 SUBMITTED dispatch is not executable until a worker owns a fenced,
-- expiring claim. At fresh CYCLE_CREATED state the current persisted Phase73 risk
-- head is re-evaluated against the immutable Phase57 cycle:
--   HALTED => cancel before paper execution;
--   REDUCING => cancel if any allowed non-reduce-only leg exists;
--   ACTIVE => cancel if a newly blocked asset has an allowed new-risk leg.
--
-- If a prior worker already advanced past CYCLE_CREATED, an expired-claim
-- takeover is resume-only: Phase67 idempotent recovery/reconciliation must finish
-- the already-started paper cycle rather than pretending no side effect exists.

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

create table if not exists public.brian_shadow_execution_claims (
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  status text not null check (
    status in ('CLAIMED', 'CANCELLED_BEFORE_EXECUTION', 'COMPLETED')
  ),
  worker_token text,
  claim_fencing_token bigint not null default 0 check (claim_fencing_token >= 0),
  claimed_at timestamptz,
  claim_until timestamptz,
  risk_version_at_decision bigint,
  risk_receipt_id_at_decision text,
  journal_stage_at_decision text,
  resume_only boolean not null default false,
  cancel_reason text,
  completed_at timestamptz,
  completion_checkpoint_id text,
  updated_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, dispatch_id),
  unique (runtime_id, cycle_id),
  check (risk_version_at_decision is null or risk_version_at_decision > 0),
  check (
    risk_receipt_id_at_decision is null
    or length(risk_receipt_id_at_decision) = 64
  ),
  check (
    completion_checkpoint_id is null
    or length(completion_checkpoint_id) = 64
  ),
  check (
    (status = 'CLAIMED'
      and worker_token is not null
      and claim_fencing_token > 0
      and claimed_at is not null
      and claim_until is not null
      and cancel_reason is null
      and completed_at is null)
    or
    (status = 'CANCELLED_BEFORE_EXECUTION'
      and claim_until is null
      and cancel_reason is not null
      and completed_at is null)
    or
    (status = 'COMPLETED'
      and claim_until is null
      and cancel_reason is null
      and completed_at is not null
      and completion_checkpoint_id is not null)
  )
);

alter table public.brian_shadow_execution_claims enable row level security;
revoke all on public.brian_shadow_execution_claims
  from anon, authenticated, service_role;
-- Mutable operational-state row by design. All writes are RPC-gated.

create table if not exists public.brian_shadow_execution_claim_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  worker_token text,
  claim_fencing_token bigint,
  event text not null check (
    event in (
      'CLAIMED',
      'CLAIMED_RESUME',
      'ALREADY_CLAIMED',
      'BLOCKED_ACTIVE',
      'EXPIRED_RECOVERY',
      'CANCELLED_BEFORE_EXECUTION',
      'RECOVERED_COMPLETED',
      'RENEWED',
      'RENEWAL_LOST',
      'COMPLETED',
      'COMPLETION_DUPLICATE',
      'COMPLETION_LOST',
      'RUNTIME_NOT_COMMITTED',
      'LEASE_LOST',
      'DISPATCH_MISSING',
      'RISK_STATE_UNAVAILABLE'
    )
  ),
  observed_at timestamptz not null default now(),
  risk_version bigint,
  risk_receipt_id text,
  journal_stage text,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_shadow_execution_claim_events_runtime_idx
  on public.brian_shadow_execution_claim_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_shadow_execution_claim_events enable row level security;
revoke all on public.brian_shadow_execution_claim_events
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_execution_claim_events_append_only
  on public.brian_shadow_execution_claim_events;
create trigger brian_shadow_execution_claim_events_append_only
  before update or delete on public.brian_shadow_execution_claim_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_claim_shadow_execution_dispatch(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_cycle_id text,
  p_worker_token text,
  p_claim_seconds integer
) returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_now timestamptz;
  v_owner text;
  v_runtime_fence bigint;
  v_runtime_version bigint;
  v_checkpoint_id text;
  v_checkpoint jsonb;
  v_lease_until timestamptz;
  v_dispatch public.brian_shadow_execution_dispatches%rowtype;
  v_claim public.brian_shadow_execution_claims%rowtype;
  v_cycle_payload jsonb;
  v_journal_stage text;
  v_risk_version bigint;
  v_risk_head_entry_id text;
  v_risk_receipt jsonb;
  v_risk_receipt_id text;
  v_risk_state text;
  v_blocked_assets jsonb;
  v_has_allowed_new_risk boolean := false;
  v_has_blocked_new_risk boolean := false;
  v_cancel_reason text;
  v_next_claim_fence bigint;
  v_event text;
  v_resume_only boolean := false;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_cycle_id is null or length(p_cycle_id) <> 64
     or nullif(trim(p_worker_token), '') is null
     or p_claim_seconds is null or p_claim_seconds <= 0 then
    raise exception 'PHASE77_CLAIM: valid runtime/owner/fence/cycle/worker/ttl required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, checkpoint_id,
         checkpoint_payload, lease_until
    into v_owner, v_runtime_fence, v_runtime_version, v_checkpoint_id,
         v_checkpoint, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if not found
     or v_owner <> p_owner_token
     or v_runtime_fence <> p_fencing_token
     or v_lease_until <= v_now then
    insert into public.brian_shadow_execution_claim_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at, metadata
    ) values (
      p_runtime_id, '<unknown>', p_cycle_id, p_worker_token,
      null, 'LEASE_LOST', v_now,
      jsonb_build_object('runtime_version', v_runtime_version)
    );
    return jsonb_build_object(
      'claimed', false,
      'cancelled', false,
      'terminal', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', v_runtime_fence
    );
  end if;

  select *
    into v_dispatch
  from public.brian_shadow_execution_dispatches
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if not found then
    insert into public.brian_shadow_execution_claim_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at
    ) values (
      p_runtime_id, '<missing>', p_cycle_id, p_worker_token,
      null, 'DISPATCH_MISSING', v_now
    );
    return jsonb_build_object(
      'claimed', false,
      'cancelled', false,
      'terminal', false,
      'status', 'DISPATCH_MISSING',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  select cycle_payload
    into v_cycle_payload
  from public.brian_shadow_runtime_cycles
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if v_cycle_payload is null then
    raise exception 'PHASE77_CLAIM: durable cycle body missing for %', p_cycle_id;
  end if;

  select e.value->>'stage'
    into v_journal_stage
  from jsonb_array_elements(
    coalesce(v_checkpoint->'journal_manifest'->'entries', '[]'::jsonb)
  ) with ordinality e(value, ord)
  where e.value->>'cycle_id' = p_cycle_id
  order by e.ord desc
  limit 1;

  if v_journal_stage = 'COMMITTED' then
    insert into public.brian_shadow_execution_claims(
      runtime_id, dispatch_id, cycle_id, status,
      worker_token, claim_fencing_token, claimed_at, claim_until,
      journal_stage_at_decision, resume_only,
      cancel_reason, completed_at, completion_checkpoint_id, updated_at
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, 'COMPLETED',
      null, 0, null, null,
      'COMMITTED', true,
      null, v_now, v_checkpoint_id, v_now
    )
    on conflict (runtime_id, dispatch_id) do update
      set status = 'COMPLETED',
          claim_until = null,
          journal_stage_at_decision = 'COMMITTED',
          resume_only = true,
          cancel_reason = null,
          completed_at = v_now,
          completion_checkpoint_id = v_checkpoint_id,
          updated_at = v_now;

    insert into public.brian_shadow_execution_claim_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at, journal_stage,
      metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      null, 'RECOVERED_COMPLETED', v_now, 'COMMITTED',
      jsonb_build_object('checkpoint_id', v_checkpoint_id)
    );

    return jsonb_build_object(
      'claimed', false,
      'cancelled', false,
      'terminal', true,
      'status', 'COMPLETED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'journal_stage', 'COMMITTED',
      'completion_checkpoint_id', v_checkpoint_id
    );
  end if;

  if v_journal_stage = 'ABORTED' then
    insert into public.brian_shadow_execution_claims(
      runtime_id, dispatch_id, cycle_id, status,
      worker_token, claim_fencing_token, claimed_at, claim_until,
      journal_stage_at_decision, resume_only,
      cancel_reason, completed_at, completion_checkpoint_id, updated_at
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id,
      'CANCELLED_BEFORE_EXECUTION',
      null, 0, null, null,
      'ABORTED', false,
      'RUNTIME_ABORTED', null, null, v_now
    )
    on conflict (runtime_id, dispatch_id) do update
      set status = 'CANCELLED_BEFORE_EXECUTION',
          worker_token = null,
          claim_until = null,
          journal_stage_at_decision = 'ABORTED',
          resume_only = false,
          cancel_reason = 'RUNTIME_ABORTED',
          completed_at = null,
          completion_checkpoint_id = null,
          updated_at = v_now;

    return jsonb_build_object(
      'claimed', false,
      'cancelled', true,
      'terminal', true,
      'status', 'CANCELLED_BEFORE_EXECUTION',
      'cancel_reason', 'RUNTIME_ABORTED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'journal_stage', 'ABORTED'
    );
  end if;

  select *
    into v_claim
  from public.brian_shadow_execution_claims
  where runtime_id = p_runtime_id
    and dispatch_id = v_dispatch.dispatch_id
  for update;

  if found and v_claim.status = 'CANCELLED_BEFORE_EXECUTION' then
    return jsonb_build_object(
      'claimed', false,
      'cancelled', true,
      'terminal', true,
      'status', 'CANCELLED_BEFORE_EXECUTION',
      'cancel_reason', v_claim.cancel_reason,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', v_claim.claim_fencing_token,
      'journal_stage', v_claim.journal_stage_at_decision
    );
  end if;

  if found and v_claim.status = 'COMPLETED' then
    return jsonb_build_object(
      'claimed', false,
      'cancelled', false,
      'terminal', true,
      'status', 'COMPLETED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', v_claim.claim_fencing_token,
      'journal_stage', v_claim.journal_stage_at_decision,
      'completion_checkpoint_id', v_claim.completion_checkpoint_id
    );
  end if;

  if found
     and v_claim.status = 'CLAIMED'
     and v_claim.claim_until > v_now then
    if v_claim.worker_token = p_worker_token then
      v_event := 'ALREADY_CLAIMED';
    else
      v_event := 'BLOCKED_ACTIVE';
    end if;

    insert into public.brian_shadow_execution_claim_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at,
      risk_version, risk_receipt_id, journal_stage,
      metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      v_claim.claim_fencing_token, v_event, v_now,
      v_claim.risk_version_at_decision,
      v_claim.risk_receipt_id_at_decision,
      v_claim.journal_stage_at_decision,
      jsonb_build_object('claim_until', v_claim.claim_until)
    );

    return jsonb_build_object(
      'claimed', v_claim.worker_token = p_worker_token,
      'cancelled', false,
      'terminal', false,
      'status', v_event,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', v_claim.claim_fencing_token,
      'claim_until', v_claim.claim_until,
      'risk_version', v_claim.risk_version_at_decision,
      'risk_receipt_id', v_claim.risk_receipt_id_at_decision,
      'journal_stage', v_claim.journal_stage_at_decision,
      'resume_only', v_claim.resume_only
    );
  end if;

  select version, head_entry_id
    into v_risk_version, v_risk_head_entry_id
  from public.brian_operational_risk_heads
  where runtime_id = p_runtime_id
  for update;

  select entry_payload->'receipt'
    into v_risk_receipt
  from public.brian_operational_risk_entries
  where runtime_id = p_runtime_id
    and entry_id = v_risk_head_entry_id;

  if v_risk_version is null
     or v_risk_receipt is null
     or jsonb_typeof(v_risk_receipt) <> 'object' then
    insert into public.brian_shadow_execution_claim_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at, journal_stage
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      null, 'RISK_STATE_UNAVAILABLE', v_now, v_journal_stage
    );
    return jsonb_build_object(
      'claimed', false,
      'cancelled', false,
      'terminal', false,
      'status', 'RISK_STATE_UNAVAILABLE',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'journal_stage', v_journal_stage
    );
  end if;

  v_risk_receipt_id := nullif(trim(v_risk_receipt->>'receipt_id'), '');
  v_risk_state := nullif(trim(v_risk_receipt->>'trading_state'), '');
  v_blocked_assets := coalesce(v_risk_receipt->'blocked_assets', '[]'::jsonb);
  if v_risk_receipt_id is null
     or length(v_risk_receipt_id) <> 64
     or v_risk_state not in ('ACTIVE','REDUCING','HALTED')
     or jsonb_typeof(v_blocked_assets) <> 'array' then
    raise exception 'PHASE77_CLAIM: malformed current risk receipt';
  end if;

  select exists(
    select 1
    from jsonb_array_elements(
      coalesce(v_cycle_payload->'items', '[]'::jsonb)
    ) item
    where coalesce((item->'risk_receipt'->>'allowed')::boolean, false)
      and not coalesce((item->'risk_receipt'->>'reduce_only')::boolean, false)
  ) into v_has_allowed_new_risk;

  select exists(
    select 1
    from jsonb_array_elements(
      coalesce(v_cycle_payload->'items', '[]'::jsonb)
    ) item
    where coalesce((item->'risk_receipt'->>'allowed')::boolean, false)
      and not coalesce((item->'risk_receipt'->>'reduce_only')::boolean, false)
      and exists(
        select 1
        from jsonb_array_elements_text(v_blocked_assets) blocked(asset)
        where blocked.asset = item->>'asset_id'
      )
  ) into v_has_blocked_new_risk;

  -- Once paper work started, takeover is resume-only. Do not erase/restart an
  -- already-applied side effect because current risk changed; Phase78 handles
  -- post-claim cancellation/kill semantics.
  v_resume_only := v_journal_stage is not null
                   and v_journal_stage <> 'CYCLE_CREATED';

  if not v_resume_only then
    if v_risk_state = 'HALTED' then
      v_cancel_reason := 'HALTED';
    elsif v_risk_state = 'REDUCING' and v_has_allowed_new_risk then
      v_cancel_reason := 'REDUCING_NEW_RISK';
    elsif v_risk_state = 'ACTIVE' and v_has_blocked_new_risk then
      v_cancel_reason := 'ASSET_COOLDOWN';
    end if;
  end if;

  if v_cancel_reason is not null then
    insert into public.brian_shadow_execution_claims(
      runtime_id, dispatch_id, cycle_id, status,
      worker_token, claim_fencing_token, claimed_at, claim_until,
      risk_version_at_decision, risk_receipt_id_at_decision,
      journal_stage_at_decision, resume_only,
      cancel_reason, completed_at, completion_checkpoint_id, updated_at
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id,
      'CANCELLED_BEFORE_EXECUTION',
      null, coalesce(v_claim.claim_fencing_token, 0), null, null,
      v_risk_version, v_risk_receipt_id,
      coalesce(v_journal_stage, 'CYCLE_CREATED'), false,
      v_cancel_reason, null, null, v_now
    )
    on conflict (runtime_id, dispatch_id) do update
      set status = 'CANCELLED_BEFORE_EXECUTION',
          worker_token = null,
          claim_until = null,
          risk_version_at_decision = excluded.risk_version_at_decision,
          risk_receipt_id_at_decision = excluded.risk_receipt_id_at_decision,
          journal_stage_at_decision = excluded.journal_stage_at_decision,
          resume_only = false,
          cancel_reason = excluded.cancel_reason,
          completed_at = null,
          completion_checkpoint_id = null,
          updated_at = v_now;

    insert into public.brian_shadow_execution_claim_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at,
      risk_version, risk_receipt_id, journal_stage, metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      coalesce(v_claim.claim_fencing_token, 0),
      'CANCELLED_BEFORE_EXECUTION', v_now,
      v_risk_version, v_risk_receipt_id,
      coalesce(v_journal_stage, 'CYCLE_CREATED'),
      jsonb_build_object(
        'reason', v_cancel_reason,
        'trading_state', v_risk_state,
        'blocked_assets', v_blocked_assets,
        'has_allowed_new_risk', v_has_allowed_new_risk
      )
    );

    return jsonb_build_object(
      'claimed', false,
      'cancelled', true,
      'terminal', true,
      'status', 'CANCELLED_BEFORE_EXECUTION',
      'cancel_reason', v_cancel_reason,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', coalesce(v_claim.claim_fencing_token, 0),
      'risk_version', v_risk_version,
      'risk_receipt_id', v_risk_receipt_id,
      'journal_stage', coalesce(v_journal_stage, 'CYCLE_CREATED'),
      'resume_only', false
    );
  end if;

  v_next_claim_fence := coalesce(v_claim.claim_fencing_token, 0) + 1;
  if found then
    v_event := 'EXPIRED_RECOVERY';
  elsif v_resume_only then
    v_event := 'CLAIMED_RESUME';
  else
    v_event := 'CLAIMED';
  end if;

  insert into public.brian_shadow_execution_claims(
    runtime_id, dispatch_id, cycle_id, status,
    worker_token, claim_fencing_token, claimed_at, claim_until,
    risk_version_at_decision, risk_receipt_id_at_decision,
    journal_stage_at_decision, resume_only,
    cancel_reason, completed_at, completion_checkpoint_id, updated_at
  ) values (
    p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, 'CLAIMED',
    p_worker_token, v_next_claim_fence, v_now,
    v_now + make_interval(secs => p_claim_seconds),
    v_risk_version, v_risk_receipt_id,
    coalesce(v_journal_stage, 'CYCLE_CREATED'), v_resume_only,
    null, null, null, v_now
  )
  on conflict (runtime_id, dispatch_id) do update
    set status = 'CLAIMED',
        worker_token = excluded.worker_token,
        claim_fencing_token = excluded.claim_fencing_token,
        claimed_at = excluded.claimed_at,
        claim_until = excluded.claim_until,
        risk_version_at_decision = excluded.risk_version_at_decision,
        risk_receipt_id_at_decision = excluded.risk_receipt_id_at_decision,
        journal_stage_at_decision = excluded.journal_stage_at_decision,
        resume_only = excluded.resume_only,
        cancel_reason = null,
        completed_at = null,
        completion_checkpoint_id = null,
        updated_at = v_now;

  insert into public.brian_shadow_execution_claim_events(
    runtime_id, dispatch_id, cycle_id, worker_token,
    claim_fencing_token, event, observed_at,
    risk_version, risk_receipt_id, journal_stage,
    metadata
  ) values (
    p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
    v_next_claim_fence, v_event, v_now,
    v_risk_version, v_risk_receipt_id,
    coalesce(v_journal_stage, 'CYCLE_CREATED'),
    jsonb_build_object(
      'claim_seconds', p_claim_seconds,
      'resume_only', v_resume_only,
      'trading_state', v_risk_state
    )
  );

  return jsonb_build_object(
    'claimed', true,
    'cancelled', false,
    'terminal', false,
    'status', v_event,
    'runtime_id', p_runtime_id,
    'dispatch_id', v_dispatch.dispatch_id,
    'cycle_id', p_cycle_id,
    'runtime_version', v_runtime_version,
    'fencing_token', p_fencing_token,
    'claim_fencing_token', v_next_claim_fence,
    'claim_until', v_now + make_interval(secs => p_claim_seconds),
    'risk_version', v_risk_version,
    'risk_receipt_id', v_risk_receipt_id,
    'journal_stage', coalesce(v_journal_stage, 'CYCLE_CREATED'),
    'resume_only', v_resume_only
  );
end;
$$;

create or replace function public.brian_renew_shadow_execution_claim(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_cycle_id text,
  p_worker_token text,
  p_claim_fencing_token bigint,
  p_claim_seconds integer
) returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_now timestamptz;
  v_runtime_ok boolean;
  v_dispatch_id text;
  v_rows integer;
  v_claim_until timestamptz;
  v_renewed boolean;
begin
  if p_claim_fencing_token is null or p_claim_fencing_token <= 0
     or p_claim_seconds is null or p_claim_seconds <= 0 then
    raise exception 'PHASE77_RENEW: positive claim fence/ttl required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select exists(
    select 1
    from public.brian_shadow_runtime_heads
    where runtime_id = p_runtime_id
      and owner_token = p_owner_token
      and fencing_token = p_fencing_token
      and lease_until > v_now
  ) into v_runtime_ok;

  select dispatch_id into v_dispatch_id
  from public.brian_shadow_execution_dispatches
  where runtime_id = p_runtime_id and cycle_id = p_cycle_id;

  update public.brian_shadow_execution_claims
    set claim_until = v_now + make_interval(secs => p_claim_seconds),
        updated_at = v_now
    where v_runtime_ok
      and runtime_id = p_runtime_id
      and dispatch_id = v_dispatch_id
      and status = 'CLAIMED'
      and worker_token = p_worker_token
      and claim_fencing_token = p_claim_fencing_token
      and claim_until > v_now
    returning claim_until into v_claim_until;

  get diagnostics v_rows = row_count;
  v_renewed := v_rows > 0;

  if v_dispatch_id is not null then
    insert into public.brian_shadow_execution_claim_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at, metadata
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token,
      case when v_renewed then 'RENEWED' else 'RENEWAL_LOST' end,
      v_now,
      jsonb_build_object('claim_seconds', p_claim_seconds)
    );
  end if;

  return jsonb_build_object(
    'renewed', v_renewed,
    'status', case when v_renewed then 'RENEWED' else 'RENEWAL_LOST' end,
    'runtime_id', p_runtime_id,
    'dispatch_id', v_dispatch_id,
    'cycle_id', p_cycle_id,
    'fencing_token', p_fencing_token,
    'claim_fencing_token', p_claim_fencing_token,
    'claim_until', v_claim_until
  );
end;
$$;

create or replace function public.brian_complete_shadow_execution_claim(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_cycle_id text,
  p_worker_token text,
  p_claim_fencing_token bigint
) returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_now timestamptz;
  v_owner text;
  v_runtime_fence bigint;
  v_checkpoint_id text;
  v_checkpoint jsonb;
  v_lease_until timestamptz;
  v_dispatch_id text;
  v_claim public.brian_shadow_execution_claims%rowtype;
  v_journal_stage text;
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, checkpoint_id, checkpoint_payload, lease_until
    into v_owner, v_runtime_fence, v_checkpoint_id, v_checkpoint, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if not found
     or v_owner <> p_owner_token
     or v_runtime_fence <> p_fencing_token
     or v_lease_until <= v_now then
    return jsonb_build_object(
      'completed', false,
      'duplicate', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id
    );
  end if;

  select dispatch_id into v_dispatch_id
  from public.brian_shadow_execution_dispatches
  where runtime_id = p_runtime_id and cycle_id = p_cycle_id;

  select * into v_claim
  from public.brian_shadow_execution_claims
  where runtime_id = p_runtime_id and dispatch_id = v_dispatch_id
  for update;

  if found and v_claim.status = 'COMPLETED' then
    insert into public.brian_shadow_execution_claim_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at,
      journal_stage, metadata
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, 'COMPLETION_DUPLICATE', v_now,
      v_claim.journal_stage_at_decision,
      jsonb_build_object('checkpoint_id', v_claim.completion_checkpoint_id)
    );
    return jsonb_build_object(
      'completed', true,
      'duplicate', true,
      'status', 'COMPLETED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'claim_fencing_token', v_claim.claim_fencing_token,
      'completion_checkpoint_id', v_claim.completion_checkpoint_id
    );
  end if;

  if not found
     or v_claim.status <> 'CLAIMED'
     or v_claim.worker_token <> p_worker_token
     or v_claim.claim_fencing_token <> p_claim_fencing_token
     or v_claim.claim_until <= v_now then
    if v_dispatch_id is not null then
      insert into public.brian_shadow_execution_claim_events(
        runtime_id, dispatch_id, cycle_id, worker_token,
        claim_fencing_token, event, observed_at
      ) values (
        p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
        p_claim_fencing_token, 'COMPLETION_LOST', v_now
      );
    end if;
    return jsonb_build_object(
      'completed', false,
      'duplicate', false,
      'status', 'CLAIM_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  select e.value->>'stage'
    into v_journal_stage
  from jsonb_array_elements(
    coalesce(v_checkpoint->'journal_manifest'->'entries', '[]'::jsonb)
  ) with ordinality e(value, ord)
  where e.value->>'cycle_id' = p_cycle_id
  order by e.ord desc
  limit 1;

  if v_journal_stage is distinct from 'COMMITTED' then
    insert into public.brian_shadow_execution_claim_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at, journal_stage
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, 'RUNTIME_NOT_COMMITTED', v_now,
      v_journal_stage
    );
    return jsonb_build_object(
      'completed', false,
      'duplicate', false,
      'status', 'RUNTIME_NOT_COMMITTED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'claim_fencing_token', p_claim_fencing_token,
      'journal_stage', v_journal_stage
    );
  end if;

  update public.brian_shadow_execution_claims
    set status = 'COMPLETED',
        claim_until = null,
        journal_stage_at_decision = 'COMMITTED',
        completed_at = v_now,
        completion_checkpoint_id = v_checkpoint_id,
        updated_at = v_now
    where runtime_id = p_runtime_id
      and dispatch_id = v_dispatch_id;

  insert into public.brian_shadow_execution_claim_events(
    runtime_id, dispatch_id, cycle_id, worker_token,
    claim_fencing_token, event, observed_at,
    journal_stage, metadata
  ) values (
    p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
    p_claim_fencing_token, 'COMPLETED', v_now,
    'COMMITTED',
    jsonb_build_object('checkpoint_id', v_checkpoint_id)
  );

  return jsonb_build_object(
    'completed', true,
    'duplicate', false,
    'status', 'COMPLETED',
    'runtime_id', p_runtime_id,
    'dispatch_id', v_dispatch_id,
    'cycle_id', p_cycle_id,
    'claim_fencing_token', p_claim_fencing_token,
    'completion_checkpoint_id', v_checkpoint_id
  );
end;
$$;

create or replace function public.brian_read_shadow_execution_claim(
  p_runtime_id text,
  p_cycle_id text
) returns jsonb
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  select case when c.runtime_id is null then null else jsonb_build_object(
    'runtime_id', c.runtime_id,
    'dispatch_id', c.dispatch_id,
    'cycle_id', c.cycle_id,
    'status', c.status,
    'worker_token', c.worker_token,
    'claim_fencing_token', c.claim_fencing_token,
    'claimed_at', c.claimed_at,
    'claim_until', c.claim_until,
    'risk_version_at_decision', c.risk_version_at_decision,
    'risk_receipt_id_at_decision', c.risk_receipt_id_at_decision,
    'journal_stage_at_decision', c.journal_stage_at_decision,
    'resume_only', c.resume_only,
    'cancel_reason', c.cancel_reason,
    'completed_at', c.completed_at,
    'completion_checkpoint_id', c.completion_checkpoint_id,
    'shadow_only', c.shadow_only,
    'live_execution', c.live_execution
  ) end
  from (select p_runtime_id as runtime_id, p_cycle_id as cycle_id) q
  left join public.brian_shadow_execution_claims c
    on c.runtime_id=q.runtime_id and c.cycle_id=q.cycle_id;
$$;

revoke all on function public.brian_claim_shadow_execution_dispatch(
  text,text,bigint,text,text,integer
) from public, anon, authenticated;
revoke all on function public.brian_renew_shadow_execution_claim(
  text,text,bigint,text,text,bigint,integer
) from public, anon, authenticated;
revoke all on function public.brian_complete_shadow_execution_claim(
  text,text,bigint,text,text,bigint
) from public, anon, authenticated;
revoke all on function public.brian_read_shadow_execution_claim(text,text)
  from public, anon, authenticated;

grant execute on function public.brian_claim_shadow_execution_dispatch(
  text,text,bigint,text,text,integer
) to service_role;
grant execute on function public.brian_renew_shadow_execution_claim(
  text,text,bigint,text,text,bigint,integer
) to service_role;
grant execute on function public.brian_complete_shadow_execution_claim(
  text,text,bigint,text,text,bigint
) to service_role;
grant execute on function public.brian_read_shadow_execution_claim(text,text)
  to service_role;

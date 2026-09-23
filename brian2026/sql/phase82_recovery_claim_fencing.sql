-- Brian Phase 82 DRAFT SQL: recovery-claim fencing lifecycle.
--
-- IMPORTANT: This remains draft/undeployed SQL. At rollout freeze create the
-- official Supabase migration with `supabase migration new` and rerun all
-- Postgres tests before deployment.
--
-- Phase81 creates an immutable recovery obligation. Phase82 makes that
-- obligation single-owner work. A recovery worker receives an independent
-- fencing token; expired work can be taken over without allowing the stale
-- worker to remain authoritative.
--
-- No paper/exchange side effect happens in this phase.

create table if not exists public.brian_shadow_recovery_claims (
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  cancel_risk_receipt_id text not null,
  status text not null check (status in ('CLAIMED','COMPLETED')),
  worker_token text,
  claim_fencing_token bigint not null default 0 check (claim_fencing_token >= 0),
  claimed_at timestamptz,
  claim_until timestamptz,
  runtime_version_at_claim bigint,
  head_state_id_at_claim text,
  risk_version_at_claim bigint,
  risk_receipt_id_at_claim text,
  risk_state_at_claim text check (
    risk_state_at_claim is null
    or risk_state_at_claim in ('ACTIVE','REDUCING','HALTED')
  ),
  recovery_cycle_id text,
  progress_runtime_version bigint,
  progress_head_state_id text,
  progress_checkpoint_id text,
  completed_at timestamptz,
  completion_ref text,
  updated_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, dispatch_id, cancel_risk_receipt_id),
  check (length(cancel_risk_receipt_id) = 64),
  check (head_state_id_at_claim is null or length(head_state_id_at_claim) = 64),
  check (risk_receipt_id_at_claim is null or length(risk_receipt_id_at_claim) = 64),
  check (recovery_cycle_id is null or length(recovery_cycle_id) = 64),
  check (progress_runtime_version is null or progress_runtime_version > 0),
  check (progress_head_state_id is null or length(progress_head_state_id) = 64),
  check (progress_checkpoint_id is null or length(progress_checkpoint_id) = 64),
  check (
    (status = 'CLAIMED'
      and worker_token is not null
      and claim_fencing_token > 0
      and claimed_at is not null
      and claim_until is not null
      and completed_at is null)
    or
    (status = 'COMPLETED'
      and claim_fencing_token > 0
      and completed_at is not null)
  )
);

create index if not exists brian_shadow_recovery_claims_cycle_idx
  on public.brian_shadow_recovery_claims(
    runtime_id, cycle_id, updated_at desc
  );

alter table public.brian_shadow_recovery_claims enable row level security;
revoke all on public.brian_shadow_recovery_claims
  from anon, authenticated, service_role;

create table if not exists public.brian_shadow_recovery_claim_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  dispatch_id text,
  cycle_id text not null,
  cancel_risk_receipt_id text,
  worker_token text,
  claim_fencing_token bigint,
  runtime_version bigint,
  head_state_id text,
  risk_version bigint,
  risk_receipt_id text,
  risk_state text,
  event text not null check (
    event in (
      'CLAIMED',
      'ALREADY_OWNED',
      'BLOCKED_ACTIVE',
      'EXPIRED_RECOVERY',
      'WAIT_RISK_RELEASE',
      'HEAD_MOVED',
      'DIRECTIVE_MISSING',
      'NO_RECOVERY_REQUIRED',
      'MANUAL_REVIEW',
      'EVIDENCE_INVALID',
      'LEASE_LOST',
      'RISK_STATE_UNAVAILABLE',
      'RENEWED',
      'RENEWAL_LOST',
      'RENEWAL_BLOCKED_RISK'
    )
  ),
  observed_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_shadow_recovery_claim_events_runtime_idx
  on public.brian_shadow_recovery_claim_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_shadow_recovery_claim_events enable row level security;
revoke all on public.brian_shadow_recovery_claim_events
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_recovery_claim_events_append_only
  on public.brian_shadow_recovery_claim_events;
create trigger brian_shadow_recovery_claim_events_append_only
  before update or delete on public.brian_shadow_recovery_claim_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_claim_shadow_cancel_recovery(
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
  v_head_state_id text;
  v_lease_until timestamptz;
  v_directive public.brian_shadow_cancel_recovery_directives%rowtype;
  v_claim public.brian_shadow_recovery_claims%rowtype;
  v_risk_version bigint;
  v_risk_head_entry_id text;
  v_risk_receipt jsonb;
  v_risk_receipt_id text;
  v_risk_state text;
  v_anchor_runtime_version bigint;
  v_anchor_head_state_id text;
  v_event text;
  v_claimed boolean := false;
  v_claim_fence bigint;
  v_claim_until timestamptz;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_cycle_id is null or length(p_cycle_id) <> 64
     or nullif(trim(p_worker_token), '') is null
     or p_claim_seconds is null or p_claim_seconds <= 0 then
    raise exception 'PHASE82_CLAIM: valid runtime/lease/cycle/worker/seconds required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, head_state_id, lease_until
    into v_owner, v_runtime_fence, v_runtime_version, v_head_state_id, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if not found
     or v_owner <> p_owner_token
     or v_runtime_fence <> p_fencing_token
     or v_lease_until <= v_now then
    insert into public.brian_shadow_recovery_claim_events(
      runtime_id, cycle_id, worker_token,
      runtime_version, head_state_id, event, observed_at,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, p_worker_token,
      v_runtime_version, v_head_state_id, 'LEASE_LOST', v_now,
      jsonb_build_object(
        'requested_runtime_fence', p_fencing_token,
        'current_runtime_fence', v_runtime_fence
      )
    );
    return jsonb_build_object(
      'claimed', false,
      'terminal', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', coalesce(v_runtime_version,0),
      'fencing_token', coalesce(v_runtime_fence,p_fencing_token)
    );
  end if;

  select *
    into v_directive
  from public.brian_shadow_cancel_recovery_directives
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id
  order by prepared_at asc, cancel_risk_version asc
  limit 1;

  if not found then
    insert into public.brian_shadow_recovery_claim_events(
      runtime_id, cycle_id, worker_token,
      runtime_version, head_state_id, event, observed_at
    ) values (
      p_runtime_id, p_cycle_id, p_worker_token,
      v_runtime_version, v_head_state_id, 'DIRECTIVE_MISSING', v_now
    );
    return jsonb_build_object(
      'claimed', false,
      'terminal', false,
      'status', 'DIRECTIVE_MISSING',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  if v_directive.recovery_status = 'NO_RECOVERY_REQUIRED' then
    insert into public.brian_shadow_recovery_claim_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, runtime_version, head_state_id,
      event, observed_at
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id, p_worker_token,
      v_runtime_version, v_head_state_id,
      'NO_RECOVERY_REQUIRED', v_now
    );
    return jsonb_build_object(
      'claimed', false,
      'terminal', true,
      'status', 'NO_RECOVERY_REQUIRED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  if v_directive.recovery_status = 'MANUAL_REVIEW' then
    insert into public.brian_shadow_recovery_claim_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, runtime_version, head_state_id,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id, p_worker_token,
      v_runtime_version, v_head_state_id,
      'MANUAL_REVIEW', v_now,
      jsonb_build_object('unsafe_assets', v_directive.unsafe_assets)
    );
    return jsonb_build_object(
      'claimed', false,
      'terminal', true,
      'status', 'MANUAL_REVIEW',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  if jsonb_array_length(v_directive.recovery_legs) = 0 then
    insert into public.brian_shadow_recovery_claim_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, runtime_version, head_state_id,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id, p_worker_token,
      v_runtime_version, v_head_state_id,
      'EVIDENCE_INVALID', v_now,
      jsonb_build_object('reason', 'RECOVERY_LEGS_EMPTY')
    );
    return jsonb_build_object(
      'claimed', false,
      'terminal', false,
      'status', 'EVIDENCE_INVALID',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  select *
    into v_claim
  from public.brian_shadow_recovery_claims
  where runtime_id = p_runtime_id
    and dispatch_id = v_directive.dispatch_id
    and cancel_risk_receipt_id = v_directive.cancel_risk_receipt_id
  for update;

  if found and v_claim.status = 'COMPLETED' then
    return jsonb_build_object(
      'claimed', false,
      'terminal', true,
      'status', 'COMPLETED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', v_claim.claim_fencing_token,
      'completion_ref', v_claim.completion_ref
    );
  end if;

  v_anchor_runtime_version := coalesce(
    v_claim.progress_runtime_version,
    v_directive.source_runtime_version
  );
  v_anchor_head_state_id := coalesce(
    v_claim.progress_head_state_id,
    v_directive.current_state_id
  );

  if v_runtime_version <> v_anchor_runtime_version
     or v_head_state_id is distinct from v_anchor_head_state_id then
    insert into public.brian_shadow_recovery_claim_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, runtime_version, head_state_id,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id, p_worker_token,
      v_runtime_version, v_head_state_id,
      'HEAD_MOVED', v_now,
      jsonb_build_object(
        'anchor_runtime_version', v_anchor_runtime_version,
        'anchor_state_id', v_anchor_head_state_id,
        'recovery_cycle_id', v_claim.recovery_cycle_id,
        'progress_checkpoint_id', v_claim.progress_checkpoint_id
      )
    );
    return jsonb_build_object(
      'claimed', false,
      'terminal', false,
      'status', 'HEAD_MOVED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token
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
    insert into public.brian_shadow_recovery_claim_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, runtime_version, head_state_id,
      event, observed_at
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id, p_worker_token,
      v_runtime_version, v_head_state_id,
      'RISK_STATE_UNAVAILABLE', v_now
    );
    return jsonb_build_object(
      'claimed', false,
      'terminal', false,
      'status', 'RISK_STATE_UNAVAILABLE',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  v_risk_receipt_id := nullif(trim(v_risk_receipt->>'receipt_id'), '');
  v_risk_state := nullif(trim(v_risk_receipt->>'trading_state'), '');
  if v_risk_receipt_id is null
     or length(v_risk_receipt_id) <> 64
     or v_risk_state not in ('ACTIVE','REDUCING','HALTED') then
    raise exception 'PHASE82_CLAIM: malformed current risk receipt';
  end if;

  if v_risk_state = 'HALTED' then
    insert into public.brian_shadow_recovery_claim_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, runtime_version, head_state_id,
      risk_version, risk_receipt_id, risk_state,
      event, observed_at
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id, p_worker_token,
      v_runtime_version, v_head_state_id,
      v_risk_version, v_risk_receipt_id, v_risk_state,
      'WAIT_RISK_RELEASE', v_now
    );
    return jsonb_build_object(
      'claimed', false,
      'terminal', false,
      'status', 'WAIT_RISK_RELEASE',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'risk_version', v_risk_version,
      'risk_receipt_id', v_risk_receipt_id,
      'risk_state', v_risk_state
    );
  end if;

  if found
     and v_claim.status = 'CLAIMED'
     and v_claim.claim_until > v_now
     and v_claim.worker_token = p_worker_token then
    v_event := 'ALREADY_OWNED';
    v_claimed := true;
    v_claim_fence := v_claim.claim_fencing_token;
    v_claim_until := v_claim.claim_until;
  elsif found
     and v_claim.status = 'CLAIMED'
     and v_claim.claim_until > v_now then
    v_event := 'BLOCKED_ACTIVE';
    v_claimed := false;
    v_claim_fence := v_claim.claim_fencing_token;
    v_claim_until := v_claim.claim_until;
  elsif found then
    v_event := 'EXPIRED_RECOVERY';
    v_claimed := true;
    v_claim_fence := v_claim.claim_fencing_token + 1;
    v_claim_until := v_now + make_interval(secs => p_claim_seconds);

    update public.brian_shadow_recovery_claims
      set status = 'CLAIMED',
          worker_token = p_worker_token,
          claim_fencing_token = v_claim_fence,
          claimed_at = v_now,
          claim_until = v_claim_until,
          runtime_version_at_claim = v_runtime_version,
          head_state_id_at_claim = v_head_state_id,
          risk_version_at_claim = v_risk_version,
          risk_receipt_id_at_claim = v_risk_receipt_id,
          risk_state_at_claim = v_risk_state,
          completed_at = null,
          completion_ref = null,
          updated_at = v_now
      where runtime_id = p_runtime_id
        and dispatch_id = v_directive.dispatch_id
        and cancel_risk_receipt_id = v_directive.cancel_risk_receipt_id;
  else
    v_event := 'CLAIMED';
    v_claimed := true;
    v_claim_fence := 1;
    v_claim_until := v_now + make_interval(secs => p_claim_seconds);

    insert into public.brian_shadow_recovery_claims(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      status, worker_token, claim_fencing_token,
      claimed_at, claim_until,
      runtime_version_at_claim, head_state_id_at_claim,
      risk_version_at_claim, risk_receipt_id_at_claim, risk_state_at_claim,
      updated_at
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id,
      'CLAIMED', p_worker_token, v_claim_fence,
      v_now, v_claim_until,
      v_runtime_version, v_head_state_id,
      v_risk_version, v_risk_receipt_id, v_risk_state,
      v_now
    );
  end if;

  insert into public.brian_shadow_recovery_claim_events(
    runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
    worker_token, claim_fencing_token,
    runtime_version, head_state_id,
    risk_version, risk_receipt_id, risk_state,
    event, observed_at,
    metadata
  ) values (
    p_runtime_id, v_directive.dispatch_id, p_cycle_id,
    v_directive.cancel_risk_receipt_id,
    p_worker_token, v_claim_fence,
    v_runtime_version, v_head_state_id,
    v_risk_version, v_risk_receipt_id, v_risk_state,
    v_event, v_now,
    jsonb_build_object(
      'claim_seconds', p_claim_seconds,
      'claim_until', v_claim_until,
      'recovery_status', v_directive.recovery_status,
      'recovery_leg_count', jsonb_array_length(v_directive.recovery_legs)
    )
  );

  return jsonb_build_object(
    'claimed', v_claimed,
    'terminal', false,
    'status', v_event,
    'runtime_id', p_runtime_id,
    'dispatch_id', v_directive.dispatch_id,
    'cycle_id', p_cycle_id,
    'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
    'runtime_version', v_runtime_version,
    'head_state_id', v_head_state_id,
    'fencing_token', p_fencing_token,
    'claim_fencing_token', v_claim_fence,
    'claim_until', v_claim_until,
    'worker_token', case when v_claimed then p_worker_token else v_claim.worker_token end,
    'risk_version', v_risk_version,
    'risk_receipt_id', v_risk_receipt_id,
    'risk_state', v_risk_state,
    'recovery_status', v_directive.recovery_status,
    'recovery_legs', v_directive.recovery_legs
  );
end;
$$;

create or replace function public.brian_renew_shadow_cancel_recovery_claim(
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
  v_owner text;
  v_runtime_fence bigint;
  v_runtime_version bigint;
  v_head_state_id text;
  v_lease_until timestamptz;
  v_claim public.brian_shadow_recovery_claims%rowtype;
  v_directive public.brian_shadow_cancel_recovery_directives%rowtype;
  v_risk_version bigint;
  v_risk_head_entry_id text;
  v_risk_receipt jsonb;
  v_risk_receipt_id text;
  v_risk_state text;
  v_anchor_runtime_version bigint;
  v_anchor_head_state_id text;
  v_until timestamptz;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_cycle_id is null or length(p_cycle_id) <> 64
     or nullif(trim(p_worker_token), '') is null
     or p_claim_fencing_token is null or p_claim_fencing_token <= 0
     or p_claim_seconds is null or p_claim_seconds <= 0 then
    raise exception 'PHASE82_RENEW: valid runtime/lease/cycle/worker/claim/seconds required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, head_state_id, lease_until
    into v_owner, v_runtime_fence, v_runtime_version, v_head_state_id, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if not found
     or v_owner <> p_owner_token
     or v_runtime_fence <> p_fencing_token
     or v_lease_until <= v_now then
    return jsonb_build_object(
      'renewed', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', coalesce(v_runtime_version,0),
      'fencing_token', coalesce(v_runtime_fence,p_fencing_token),
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  select c.*
    into v_claim
  from public.brian_shadow_recovery_claims c
  where c.runtime_id = p_runtime_id
    and c.cycle_id = p_cycle_id
  limit 1
  for update;

  select d.*
    into v_directive
  from public.brian_shadow_cancel_recovery_directives d
  where d.runtime_id=p_runtime_id and d.cycle_id=p_cycle_id
  order by d.prepared_at asc
  limit 1;

  if v_claim.runtime_id is null
     or v_claim.status <> 'CLAIMED'
     or v_claim.worker_token <> p_worker_token
     or v_claim.claim_fencing_token <> p_claim_fencing_token
     or v_claim.claim_until <= v_now then
    insert into public.brian_shadow_recovery_claim_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, claim_fencing_token,
      runtime_version, head_state_id,
      event, observed_at
    ) values (
      p_runtime_id, v_claim.dispatch_id, p_cycle_id,
      v_claim.cancel_risk_receipt_id,
      p_worker_token, p_claim_fencing_token,
      v_runtime_version, v_head_state_id,
      'RENEWAL_LOST', v_now
    );
    return jsonb_build_object(
      'renewed', false,
      'status', 'RENEWAL_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_claim.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  v_anchor_runtime_version := coalesce(
    v_claim.progress_runtime_version,
    v_directive.source_runtime_version
  );
  v_anchor_head_state_id := coalesce(
    v_claim.progress_head_state_id,
    v_directive.current_state_id
  );

  if v_runtime_version <> v_anchor_runtime_version
     or v_head_state_id is distinct from v_anchor_head_state_id then
    insert into public.brian_shadow_recovery_claim_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, claim_fencing_token,
      runtime_version, head_state_id,
      event, observed_at
    ) values (
      p_runtime_id, v_claim.dispatch_id, p_cycle_id,
      v_claim.cancel_risk_receipt_id,
      p_worker_token, p_claim_fencing_token,
      v_runtime_version, v_head_state_id,
      'RENEWAL_LOST', v_now
    );
    return jsonb_build_object(
      'renewed', false,
      'status', 'RENEWAL_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_claim.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
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
    return jsonb_build_object(
      'renewed', false,
      'status', 'RISK_STATE_UNAVAILABLE',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_claim.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  v_risk_receipt_id := nullif(trim(v_risk_receipt->>'receipt_id'), '');
  v_risk_state := nullif(trim(v_risk_receipt->>'trading_state'), '');
  if v_risk_receipt_id is null
     or length(v_risk_receipt_id) <> 64
     or v_risk_state not in ('ACTIVE','REDUCING','HALTED') then
    raise exception 'PHASE82_RENEW: malformed current risk receipt';
  end if;

  if v_risk_state = 'HALTED' then
    insert into public.brian_shadow_recovery_claim_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, claim_fencing_token,
      runtime_version, head_state_id,
      risk_version, risk_receipt_id, risk_state,
      event, observed_at
    ) values (
      p_runtime_id, v_claim.dispatch_id, p_cycle_id,
      v_claim.cancel_risk_receipt_id,
      p_worker_token, p_claim_fencing_token,
      v_runtime_version, v_head_state_id,
      v_risk_version, v_risk_receipt_id, v_risk_state,
      'RENEWAL_BLOCKED_RISK', v_now
    );
    return jsonb_build_object(
      'renewed', false,
      'status', 'RENEWAL_BLOCKED_RISK',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_claim.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token,
      'risk_version', v_risk_version,
      'risk_receipt_id', v_risk_receipt_id,
      'risk_state', v_risk_state
    );
  end if;

  v_until := v_now + make_interval(secs => p_claim_seconds);
  update public.brian_shadow_recovery_claims
    set claim_until=v_until,
        risk_version_at_claim=v_risk_version,
        risk_receipt_id_at_claim=v_risk_receipt_id,
        risk_state_at_claim=v_risk_state,
        updated_at=v_now
    where runtime_id=p_runtime_id
      and dispatch_id=v_claim.dispatch_id
      and cancel_risk_receipt_id=v_claim.cancel_risk_receipt_id;

  insert into public.brian_shadow_recovery_claim_events(
    runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
    worker_token, claim_fencing_token,
    runtime_version, head_state_id,
    risk_version, risk_receipt_id, risk_state,
    event, observed_at,
    metadata
  ) values (
    p_runtime_id, v_claim.dispatch_id, p_cycle_id,
    v_claim.cancel_risk_receipt_id,
    p_worker_token, p_claim_fencing_token,
    v_runtime_version, v_head_state_id,
    v_risk_version, v_risk_receipt_id, v_risk_state,
    'RENEWED', v_now,
    jsonb_build_object('claim_until', v_until, 'claim_seconds', p_claim_seconds)
  );

  return jsonb_build_object(
    'renewed', true,
    'status', 'RENEWED',
    'runtime_id', p_runtime_id,
    'dispatch_id', v_claim.dispatch_id,
    'cycle_id', p_cycle_id,
    'cancel_risk_receipt_id', v_claim.cancel_risk_receipt_id,
    'runtime_version', v_runtime_version,
    'head_state_id', v_head_state_id,
    'fencing_token', p_fencing_token,
    'claim_fencing_token', p_claim_fencing_token,
    'claim_until', v_until,
    'risk_version', v_risk_version,
    'risk_receipt_id', v_risk_receipt_id,
    'risk_state', v_risk_state
  );
end;
$$;

revoke all on function public.brian_claim_shadow_cancel_recovery(
  text,text,bigint,text,text,integer
) from public, anon, authenticated;
revoke all on function public.brian_renew_shadow_cancel_recovery_claim(
  text,text,bigint,text,text,bigint,integer
) from public, anon, authenticated;

grant execute on function public.brian_claim_shadow_cancel_recovery(
  text,text,bigint,text,text,integer
) to service_role;
grant execute on function public.brian_renew_shadow_cancel_recovery_claim(
  text,text,bigint,text,text,bigint,integer
) to service_role;

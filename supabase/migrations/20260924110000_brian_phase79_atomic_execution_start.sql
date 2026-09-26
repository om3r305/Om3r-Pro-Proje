-- Brian Phase79 official migration: atomic execution-start point-of-no-return.
-- Promoted from the reviewed Phase79 draft contract into official
-- migration lineage on 2026-09-24. This repository change does NOT deploy the
-- migration to any live Supabase project. Runtime remains shadow/paper-only.

create table if not exists public.brian_shadow_execution_starts (
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  worker_token text not null,
  claim_fencing_token bigint not null check (claim_fencing_token > 0),
  runtime_fencing_token bigint not null check (runtime_fencing_token > 0),
  runtime_version_at_start bigint not null check (runtime_version_at_start > 0),
  risk_version_at_start bigint not null check (risk_version_at_start > 0),
  risk_receipt_id_at_start text not null,
  risk_state_at_start text not null check (
    risk_state_at_start in ('ACTIVE','REDUCING','HALTED')
  ),
  journal_stage_at_start text not null,
  phase78_status_at_start text not null check (
    phase78_status_at_start in ('PROCEED','RESUME_ONLY','CANCEL_REQUESTED')
  ),
  cancel_requested_at_start boolean not null default false,
  cancel_reason_at_start text,
  resume_only boolean not null default false,
  started_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, dispatch_id),
  unique (runtime_id, cycle_id),
  check (length(risk_receipt_id_at_start) = 64),
  check (
    (cancel_requested_at_start and cancel_reason_at_start is not null)
    or
    (not cancel_requested_at_start)
  )
);

create index if not exists brian_shadow_execution_starts_runtime_idx
  on public.brian_shadow_execution_starts(
    runtime_id, started_at desc, cycle_id
  );

alter table public.brian_shadow_execution_starts enable row level security;
revoke all on public.brian_shadow_execution_starts
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_execution_starts_append_only
  on public.brian_shadow_execution_starts;
create trigger brian_shadow_execution_starts_append_only
  before update or delete on public.brian_shadow_execution_starts
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_shadow_execution_start_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  worker_token text not null,
  claim_fencing_token bigint not null,
  runtime_fencing_token bigint not null,
  runtime_version bigint,
  risk_version bigint,
  risk_receipt_id text,
  journal_stage text,
  phase78_status text,
  event text not null check (
    event in (
      'STARTED',
      'STARTED_RESUME',
      'STARTED_ALREADY',
      'CANCELLED_BEFORE_START',
      'COMPLETED',
      'ABORTED',
      'LEASE_LOST',
      'CLAIM_LOST',
      'RISK_STATE_UNAVAILABLE'
    )
  ),
  observed_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_shadow_execution_start_events_runtime_idx
  on public.brian_shadow_execution_start_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_shadow_execution_start_events enable row level security;
revoke all on public.brian_shadow_execution_start_events
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_execution_start_events_append_only
  on public.brian_shadow_execution_start_events;
create trigger brian_shadow_execution_start_events_append_only
  before update or delete on public.brian_shadow_execution_start_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_mark_shadow_execution_started(
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
  v_runtime_version bigint;
  v_lease_until timestamptz;
  v_dispatch_id text;
  v_claim public.brian_shadow_execution_claims%rowtype;
  v_existing public.brian_shadow_execution_starts%rowtype;
  v_decision jsonb;
  v_status text;
  v_proceed boolean;
  v_cancel_requested boolean;
  v_terminal boolean;
  v_reason text;
  v_journal_stage text;
  v_risk_version bigint;
  v_risk_receipt_id text;
  v_risk_head_entry_id text;
  v_risk_receipt jsonb;
  v_risk_state text;
  v_resume_only boolean;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_cycle_id is null or length(p_cycle_id) <> 64
     or nullif(trim(p_worker_token), '') is null
     or p_claim_fencing_token is null or p_claim_fencing_token <= 0 then
    raise exception 'PHASE79_START: valid runtime/lease/cycle/worker/claim fence required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, lease_until
    into v_owner, v_runtime_fence, v_runtime_version, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  select dispatch_id
    into v_dispatch_id
  from public.brian_shadow_execution_dispatches
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if v_dispatch_id is null then
    raise exception 'PHASE79_START: dispatch missing for %', p_cycle_id;
  end if;

  if v_owner is null
     or v_owner <> p_owner_token
     or v_runtime_fence <> p_fencing_token
     or v_lease_until <= v_now then
    insert into public.brian_shadow_execution_start_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token,
      runtime_version, event, observed_at
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token,
      v_runtime_version, 'LEASE_LOST', v_now
    );
    return jsonb_build_object(
      'started', false,
      'duplicate', false,
      'terminal', false,
      'cancel_requested', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', v_runtime_fence,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  select *
    into v_claim
  from public.brian_shadow_execution_claims
  where runtime_id = p_runtime_id
    and dispatch_id = v_dispatch_id
  for update;

  if not found
     or v_claim.status <> 'CLAIMED'
     or v_claim.worker_token <> p_worker_token
     or v_claim.claim_fencing_token <> p_claim_fencing_token
     or v_claim.claim_until <= v_now then
    insert into public.brian_shadow_execution_start_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token,
      runtime_version, event, observed_at
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token,
      v_runtime_version, 'CLAIM_LOST', v_now
    );
    return jsonb_build_object(
      'started', false,
      'duplicate', false,
      'terminal', false,
      'cancel_requested', false,
      'status', 'CLAIM_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  select *
    into v_existing
  from public.brian_shadow_execution_starts
  where runtime_id = p_runtime_id
    and dispatch_id = v_dispatch_id;

  if found then
    insert into public.brian_shadow_execution_start_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token,
      runtime_version, risk_version, risk_receipt_id,
      journal_stage, phase78_status, event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token,
      v_runtime_version, v_existing.risk_version_at_start,
      v_existing.risk_receipt_id_at_start,
      v_existing.journal_stage_at_start,
      v_existing.phase78_status_at_start,
      'STARTED_ALREADY', v_now,
      jsonb_build_object(
        'original_worker_token', v_existing.worker_token,
        'original_claim_fencing_token', v_existing.claim_fencing_token,
        'cancel_requested_at_start', v_existing.cancel_requested_at_start,
        'resume_only', v_existing.resume_only
      )
    );

    return jsonb_build_object(
      'started', true,
      'duplicate', true,
      'terminal', false,
      'cancel_requested', v_existing.cancel_requested_at_start,
      'status', 'STARTED_ALREADY',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token,
      'risk_version', v_existing.risk_version_at_start,
      'risk_receipt_id', v_existing.risk_receipt_id_at_start,
      'risk_state', v_existing.risk_state_at_start,
      'journal_stage', v_existing.journal_stage_at_start,
      'phase78_status', v_existing.phase78_status_at_start,
      'reason', v_existing.cancel_reason_at_start,
      'resume_only', v_existing.resume_only
    );
  end if;

  -- Phase78 owns the policy decision. Because this call runs inside the same
  -- advisory-lock transaction, no Phase73 risk commit can slip between the
  -- decision and the immutable STARTED insert below.
  select public.brian_check_shadow_execution_kill_switch(
    p_runtime_id,
    p_owner_token,
    p_fencing_token,
    p_cycle_id,
    p_worker_token,
    p_claim_fencing_token
  ) into v_decision;

  v_status := nullif(trim(v_decision->>'status'), '');
  v_proceed := coalesce((v_decision->>'proceed')::boolean, false);
  v_cancel_requested := coalesce(
    (v_decision->>'cancel_requested')::boolean,
    false
  );
  v_terminal := coalesce((v_decision->>'terminal')::boolean, false);
  v_reason := nullif(trim(v_decision->>'reason'), '');
  v_journal_stage := nullif(trim(v_decision->>'journal_stage'), '');
  v_risk_version := nullif(v_decision->>'risk_version', '')::bigint;
  v_risk_receipt_id := nullif(trim(v_decision->>'risk_receipt_id'), '');

  if v_status in ('LEASE_LOST','CLAIM_LOST','RISK_STATE_UNAVAILABLE') then
    insert into public.brian_shadow_execution_start_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token,
      runtime_version, risk_version, risk_receipt_id,
      journal_stage, phase78_status, event, observed_at
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token,
      v_runtime_version, v_risk_version, v_risk_receipt_id,
      v_journal_stage, v_status, v_status, v_now
    );
    return jsonb_build_object(
      'started', false,
      'duplicate', false,
      'terminal', false,
      'cancel_requested', false,
      'status', v_status,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token,
      'journal_stage', v_journal_stage
    );
  end if;

  if v_status in ('CANCELLED_BEFORE_EXECUTION','ABORTED','COMPLETED') then
    insert into public.brian_shadow_execution_start_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token,
      runtime_version, risk_version, risk_receipt_id,
      journal_stage, phase78_status, event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token,
      v_runtime_version, v_risk_version, v_risk_receipt_id,
      v_journal_stage, v_status,
      case
        when v_status = 'COMPLETED' then 'COMPLETED'
        when v_status = 'ABORTED' then 'ABORTED'
        else 'CANCELLED_BEFORE_START'
      end,
      v_now,
      jsonb_build_object('reason', v_reason)
    );

    return jsonb_build_object(
      'started', false,
      'duplicate', false,
      'terminal', true,
      'cancel_requested', v_cancel_requested,
      'status', v_status,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token,
      'risk_version', v_risk_version,
      'risk_receipt_id', v_risk_receipt_id,
      'journal_stage', v_journal_stage,
      'reason', v_reason
    );
  end if;

  if v_status not in ('PROCEED','RESUME_ONLY','CANCEL_REQUESTED')
     or not v_proceed then
    raise exception 'PHASE79_START: unexpected Phase78 decision %', v_decision;
  end if;

  select head_entry_id
    into v_risk_head_entry_id
  from public.brian_operational_risk_heads
  where runtime_id = p_runtime_id
  for update;

  select entry_payload->'receipt'
    into v_risk_receipt
  from public.brian_operational_risk_entries
  where runtime_id = p_runtime_id
    and entry_id = v_risk_head_entry_id;

  v_risk_state := nullif(trim(v_risk_receipt->>'trading_state'), '');

  if v_risk_version is null
     or v_risk_version <= 0
     or v_risk_receipt_id is null
     or length(v_risk_receipt_id) <> 64
     or v_risk_state not in ('ACTIVE','REDUCING','HALTED')
     or v_journal_stage is null then
    raise exception 'PHASE79_START: incomplete Phase78/risk evidence';
  end if;

  v_resume_only := v_status in ('RESUME_ONLY','CANCEL_REQUESTED');

  insert into public.brian_shadow_execution_starts(
    runtime_id, dispatch_id, cycle_id, worker_token,
    claim_fencing_token, runtime_fencing_token,
    runtime_version_at_start, risk_version_at_start,
    risk_receipt_id_at_start, risk_state_at_start,
    journal_stage_at_start, phase78_status_at_start,
    cancel_requested_at_start, cancel_reason_at_start,
    resume_only, started_at
  ) values (
    p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
    p_claim_fencing_token, p_fencing_token,
    v_runtime_version, v_risk_version,
    v_risk_receipt_id, v_risk_state,
    v_journal_stage, v_status,
    v_cancel_requested, v_reason,
    v_resume_only, v_now
  );

  insert into public.brian_shadow_execution_start_events(
    runtime_id, dispatch_id, cycle_id, worker_token,
    claim_fencing_token, runtime_fencing_token,
    runtime_version, risk_version, risk_receipt_id,
    journal_stage, phase78_status, event, observed_at,
    metadata
  ) values (
    p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
    p_claim_fencing_token, p_fencing_token,
    v_runtime_version, v_risk_version, v_risk_receipt_id,
    v_journal_stage, v_status,
    case when v_resume_only then 'STARTED_RESUME' else 'STARTED' end,
    v_now,
    jsonb_build_object(
      'cancel_requested_at_start', v_cancel_requested,
      'reason', v_reason,
      'risk_state', v_risk_state,
      'resume_only', v_resume_only
    )
  );

  return jsonb_build_object(
    'started', true,
    'duplicate', false,
    'terminal', false,
    'cancel_requested', v_cancel_requested,
    'status', case when v_resume_only then 'STARTED_RESUME' else 'STARTED' end,
    'runtime_id', p_runtime_id,
    'dispatch_id', v_dispatch_id,
    'cycle_id', p_cycle_id,
    'runtime_version', v_runtime_version,
    'fencing_token', p_fencing_token,
    'claim_fencing_token', p_claim_fencing_token,
    'risk_version', v_risk_version,
    'risk_receipt_id', v_risk_receipt_id,
    'risk_state', v_risk_state,
    'journal_stage', v_journal_stage,
    'phase78_status', v_status,
    'reason', v_reason,
    'resume_only', v_resume_only
  );
end;
$$;

create or replace function public.brian_read_shadow_execution_start(
  p_runtime_id text,
  p_cycle_id text
) returns jsonb
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  select case when s.runtime_id is null then null else jsonb_build_object(
    'runtime_id', s.runtime_id,
    'dispatch_id', s.dispatch_id,
    'cycle_id', s.cycle_id,
    'worker_token', s.worker_token,
    'claim_fencing_token', s.claim_fencing_token,
    'runtime_fencing_token', s.runtime_fencing_token,
    'runtime_version_at_start', s.runtime_version_at_start,
    'risk_version_at_start', s.risk_version_at_start,
    'risk_receipt_id_at_start', s.risk_receipt_id_at_start,
    'risk_state_at_start', s.risk_state_at_start,
    'journal_stage_at_start', s.journal_stage_at_start,
    'phase78_status_at_start', s.phase78_status_at_start,
    'cancel_requested_at_start', s.cancel_requested_at_start,
    'cancel_reason_at_start', s.cancel_reason_at_start,
    'resume_only', s.resume_only,
    'started_at', s.started_at,
    'shadow_only', s.shadow_only,
    'live_execution', s.live_execution
  ) end
  from (select p_runtime_id runtime_id, p_cycle_id cycle_id) q
  left join public.brian_shadow_execution_starts s
    on s.runtime_id=q.runtime_id and s.cycle_id=q.cycle_id;
$$;

revoke all on function public.brian_mark_shadow_execution_started(
  text,text,bigint,text,text,bigint
) from public, anon, authenticated;
revoke all on function public.brian_read_shadow_execution_start(text,text)
  from public, anon, authenticated;

grant execute on function public.brian_mark_shadow_execution_started(
  text,text,bigint,text,text,bigint
) to service_role;
grant execute on function public.brian_read_shadow_execution_start(text,text)
  to service_role;

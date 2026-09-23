-- Brian Phase 78 DRAFT SQL: execution-start point-of-no-return + post-start kill requests.
--
-- IMPORTANT: This is intentionally NOT a Supabase migration file yet.
-- The PR is still draft/undeployed. At rollout freeze this SQL must be converted
-- into an official migration created with `supabase migration new`, then
-- re-verified locally/CI before deployment.
--
-- Semantics:
--   * Phase77 CLAIMED is ownership, not permission to begin side effects.
--   * Phase78 STARTED is the durable point-of-no-return. Current risk is checked
--     one final time under the same runtime advisory lock before STARTED exists.
--   * Before STARTED, a new HALT / REDUCING-new-risk / asset cooldown cancels.
--   * After STARTED, later risk deterioration never pretends prior side effects
--     did not happen. It emits an immutable kill request for recovery/reduction.
--   * A journal already beyond CYCLE_CREATED is recovered as STARTED_RESUME,
--     because persisted paper progress proves the point-of-no-return was crossed.

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
  resume_only boolean not null default false,
  started_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, dispatch_id),
  unique (runtime_id, cycle_id),
  check (length(risk_receipt_id_at_start) = 64)
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
  worker_token text,
  claim_fencing_token bigint,
  runtime_fencing_token bigint,
  runtime_version bigint,
  risk_version bigint,
  risk_receipt_id text,
  journal_stage text,
  event text not null check (
    event in (
      'STARTED',
      'STARTED_ALREADY',
      'STARTED_RESUME',
      'CANCELLED_BEFORE_START',
      'LEASE_LOST',
      'DISPATCH_MISSING',
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

create table if not exists public.brian_shadow_execution_kill_requests (
  request_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  risk_version bigint not null check (risk_version > 0),
  risk_receipt_id text not null,
  trading_state text not null check (
    trading_state in ('ACTIVE','REDUCING','HALTED')
  ),
  reason text not null check (
    reason in ('HALTED','REDUCING_NEW_RISK','ASSET_COOLDOWN')
  ),
  journal_stage text not null,
  observed_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  unique (
    runtime_id, dispatch_id, risk_version, risk_receipt_id, reason
  ),
  check (length(risk_receipt_id) = 64)
);

create index if not exists brian_shadow_execution_kill_requests_runtime_idx
  on public.brian_shadow_execution_kill_requests(
    runtime_id, observed_at desc, request_sequence desc
  );

alter table public.brian_shadow_execution_kill_requests enable row level security;
revoke all on public.brian_shadow_execution_kill_requests
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_execution_kill_requests_append_only
  on public.brian_shadow_execution_kill_requests;
create trigger brian_shadow_execution_kill_requests_append_only
  before update or delete on public.brian_shadow_execution_kill_requests
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_start_shadow_execution_claim(
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
  v_checkpoint jsonb;
  v_lease_until timestamptz;
  v_dispatch public.brian_shadow_execution_dispatches%rowtype;
  v_claim public.brian_shadow_execution_claims%rowtype;
  v_start public.brian_shadow_execution_starts%rowtype;
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
  v_resume_only boolean := false;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_cycle_id is null or length(p_cycle_id) <> 64
     or nullif(trim(p_worker_token), '') is null
     or p_claim_fencing_token is null or p_claim_fencing_token <= 0 then
    raise exception 'PHASE78_START: valid runtime/owner/fence/cycle/worker/claim-fence required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, checkpoint_payload, lease_until
    into v_owner, v_runtime_fence, v_runtime_version, v_checkpoint, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if not found
     or v_owner <> p_owner_token
     or v_runtime_fence <> p_fencing_token
     or v_lease_until <= v_now then
    insert into public.brian_shadow_execution_start_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token, runtime_version,
      event, observed_at
    ) values (
      p_runtime_id, '<unknown>', p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token, v_runtime_version,
      'LEASE_LOST', v_now
    );
    return jsonb_build_object(
      'started', false,
      'cancelled', false,
      'duplicate', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', v_runtime_fence,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  select *
    into v_dispatch
  from public.brian_shadow_execution_dispatches
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if not found then
    insert into public.brian_shadow_execution_start_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token, runtime_version,
      event, observed_at
    ) values (
      p_runtime_id, '<missing>', p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token, v_runtime_version,
      'DISPATCH_MISSING', v_now
    );
    return jsonb_build_object(
      'started', false,
      'cancelled', false,
      'duplicate', false,
      'status', 'DISPATCH_MISSING',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  select *
    into v_claim
  from public.brian_shadow_execution_claims
  where runtime_id = p_runtime_id
    and dispatch_id = v_dispatch.dispatch_id
  for update;

  if not found
     or v_claim.status <> 'CLAIMED'
     or v_claim.worker_token <> p_worker_token
     or v_claim.claim_fencing_token <> p_claim_fencing_token
     or v_claim.claim_until <= v_now then
    insert into public.brian_shadow_execution_start_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token, runtime_version,
      event, observed_at
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token, v_runtime_version,
      'CLAIM_LOST', v_now
    );
    return jsonb_build_object(
      'started', false,
      'cancelled', false,
      'duplicate', false,
      'status', 'CLAIM_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  select *
    into v_start
  from public.brian_shadow_execution_starts
  where runtime_id = p_runtime_id
    and dispatch_id = v_dispatch.dispatch_id;

  if found then
    insert into public.brian_shadow_execution_start_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token, runtime_version,
      risk_version, risk_receipt_id, journal_stage,
      event, observed_at, metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token, v_runtime_version,
      v_start.risk_version_at_start, v_start.risk_receipt_id_at_start,
      v_start.journal_stage_at_start,
      'STARTED_ALREADY', v_now,
      jsonb_build_object(
        'original_worker_token', v_start.worker_token,
        'original_claim_fencing_token', v_start.claim_fencing_token,
        'resume_only', v_start.resume_only
      )
    );

    return jsonb_build_object(
      'started', true,
      'cancelled', false,
      'duplicate', true,
      'status', 'STARTED_ALREADY',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token,
      'risk_version', v_start.risk_version_at_start,
      'risk_receipt_id', v_start.risk_receipt_id_at_start,
      'risk_state', v_start.risk_state_at_start,
      'journal_stage', v_start.journal_stage_at_start,
      'resume_only', v_start.resume_only
    );
  end if;

  select cycle_payload
    into v_cycle_payload
  from public.brian_shadow_runtime_cycles
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if v_cycle_payload is null then
    raise exception 'PHASE78_START: durable cycle body missing for %', p_cycle_id;
  end if;

  select e.value->>'stage'
    into v_journal_stage
  from jsonb_array_elements(
    coalesce(v_checkpoint->'journal_manifest'->'entries', '[]'::jsonb)
  ) with ordinality e(value, ord)
  where e.value->>'cycle_id' = p_cycle_id
  order by e.ord desc
  limit 1;

  if v_journal_stage is null then
    raise exception 'PHASE78_START: cycle % missing from current journal', p_cycle_id;
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
    insert into public.brian_shadow_execution_start_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token, runtime_version,
      event, observed_at, journal_stage
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token, v_runtime_version,
      'RISK_STATE_UNAVAILABLE', v_now, v_journal_stage
    );
    return jsonb_build_object(
      'started', false,
      'cancelled', false,
      'duplicate', false,
      'status', 'RISK_STATE_UNAVAILABLE',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token,
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
    raise exception 'PHASE78_START: malformed current risk receipt';
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

  v_resume_only := v_journal_stage <> 'CYCLE_CREATED';

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
    update public.brian_shadow_execution_claims
      set status = 'CANCELLED_BEFORE_EXECUTION',
          worker_token = null,
          claim_until = null,
          risk_version_at_decision = v_risk_version,
          risk_receipt_id_at_decision = v_risk_receipt_id,
          journal_stage_at_decision = v_journal_stage,
          resume_only = false,
          cancel_reason = v_cancel_reason,
          completed_at = null,
          completion_checkpoint_id = null,
          updated_at = v_now
      where runtime_id = p_runtime_id
        and dispatch_id = v_dispatch.dispatch_id;

    insert into public.brian_shadow_execution_start_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token, runtime_version,
      risk_version, risk_receipt_id, journal_stage,
      event, observed_at, metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token, v_runtime_version,
      v_risk_version, v_risk_receipt_id, v_journal_stage,
      'CANCELLED_BEFORE_START', v_now,
      jsonb_build_object(
        'reason', v_cancel_reason,
        'trading_state', v_risk_state,
        'blocked_assets', v_blocked_assets
      )
    );

    return jsonb_build_object(
      'started', false,
      'cancelled', true,
      'duplicate', false,
      'status', 'CANCELLED_BEFORE_START',
      'cancel_reason', v_cancel_reason,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token,
      'risk_version', v_risk_version,
      'risk_receipt_id', v_risk_receipt_id,
      'risk_state', v_risk_state,
      'journal_stage', v_journal_stage,
      'resume_only', false
    );
  end if;

  insert into public.brian_shadow_execution_starts(
    runtime_id, dispatch_id, cycle_id, worker_token,
    claim_fencing_token, runtime_fencing_token, runtime_version_at_start,
    risk_version_at_start, risk_receipt_id_at_start, risk_state_at_start,
    journal_stage_at_start, resume_only, started_at
  ) values (
    p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
    p_claim_fencing_token, p_fencing_token, v_runtime_version,
    v_risk_version, v_risk_receipt_id, v_risk_state,
    v_journal_stage, v_resume_only, v_now
  );

  insert into public.brian_shadow_execution_start_events(
    runtime_id, dispatch_id, cycle_id, worker_token,
    claim_fencing_token, runtime_fencing_token, runtime_version,
    risk_version, risk_receipt_id, journal_stage,
    event, observed_at, metadata
  ) values (
    p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
    p_claim_fencing_token, p_fencing_token, v_runtime_version,
    v_risk_version, v_risk_receipt_id, v_journal_stage,
    case when v_resume_only then 'STARTED_RESUME' else 'STARTED' end,
    v_now,
    jsonb_build_object('resume_only', v_resume_only)
  );

  return jsonb_build_object(
    'started', true,
    'cancelled', false,
    'duplicate', false,
    'status', case when v_resume_only then 'STARTED_RESUME' else 'STARTED' end,
    'runtime_id', p_runtime_id,
    'dispatch_id', v_dispatch.dispatch_id,
    'cycle_id', p_cycle_id,
    'runtime_version', v_runtime_version,
    'fencing_token', p_fencing_token,
    'claim_fencing_token', p_claim_fencing_token,
    'risk_version', v_risk_version,
    'risk_receipt_id', v_risk_receipt_id,
    'risk_state', v_risk_state,
    'journal_stage', v_journal_stage,
    'resume_only', v_resume_only
  );
end;
$$;

create or replace function public.brian_check_shadow_execution_kill_switch(
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
  v_runtime_ok boolean;
  v_runtime_version bigint;
  v_checkpoint jsonb;
  v_dispatch public.brian_shadow_execution_dispatches%rowtype;
  v_claim public.brian_shadow_execution_claims%rowtype;
  v_start public.brian_shadow_execution_starts%rowtype;
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
  v_reason text;
  v_request_sequence bigint;
begin
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select
    owner_token = p_owner_token
    and fencing_token = p_fencing_token
    and lease_until > v_now,
    version,
    checkpoint_payload
    into v_runtime_ok, v_runtime_version, v_checkpoint
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if coalesce(v_runtime_ok, false) is not true then
    return jsonb_build_object(
      'kill_requested', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  select *
    into v_dispatch
  from public.brian_shadow_execution_dispatches
  where runtime_id = p_runtime_id and cycle_id = p_cycle_id;

  if not found then
    return jsonb_build_object(
      'kill_requested', false,
      'status', 'DISPATCH_MISSING',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  select *
    into v_claim
  from public.brian_shadow_execution_claims
  where runtime_id = p_runtime_id
    and dispatch_id = v_dispatch.dispatch_id;

  if not found
     or v_claim.status <> 'CLAIMED'
     or v_claim.worker_token <> p_worker_token
     or v_claim.claim_fencing_token <> p_claim_fencing_token
     or v_claim.claim_until <= v_now then
    return jsonb_build_object(
      'kill_requested', false,
      'status', 'CLAIM_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  select *
    into v_start
  from public.brian_shadow_execution_starts
  where runtime_id = p_runtime_id
    and dispatch_id = v_dispatch.dispatch_id;

  if not found then
    return jsonb_build_object(
      'kill_requested', false,
      'status', 'START_MISSING',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  select cycle_payload
    into v_cycle_payload
  from public.brian_shadow_runtime_cycles
  where runtime_id = p_runtime_id and cycle_id = p_cycle_id;

  select e.value->>'stage'
    into v_journal_stage
  from jsonb_array_elements(
    coalesce(v_checkpoint->'journal_manifest'->'entries', '[]'::jsonb)
  ) with ordinality e(value, ord)
  where e.value->>'cycle_id' = p_cycle_id
  order by e.ord desc
  limit 1;

  if v_journal_stage = 'COMMITTED' then
    return jsonb_build_object(
      'kill_requested', false,
      'status', 'COMMITTED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token,
      'journal_stage', 'COMMITTED',
      'start_risk_version', v_start.risk_version_at_start,
      'start_risk_receipt_id', v_start.risk_receipt_id_at_start
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
      'kill_requested', true,
      'status', 'RISK_STATE_UNAVAILABLE',
      'reason', 'HALTED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token,
      'journal_stage', v_journal_stage,
      'start_risk_version', v_start.risk_version_at_start,
      'start_risk_receipt_id', v_start.risk_receipt_id_at_start
    );
  end if;

  v_risk_receipt_id := nullif(trim(v_risk_receipt->>'receipt_id'), '');
  v_risk_state := nullif(trim(v_risk_receipt->>'trading_state'), '');
  v_blocked_assets := coalesce(v_risk_receipt->'blocked_assets', '[]'::jsonb);

  if v_risk_receipt_id is null
     or length(v_risk_receipt_id) <> 64
     or v_risk_state not in ('ACTIVE','REDUCING','HALTED')
     or jsonb_typeof(v_blocked_assets) <> 'array' then
    raise exception 'PHASE78_KILL: malformed current risk receipt';
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

  if v_risk_state = 'HALTED' then
    v_reason := 'HALTED';
  elsif v_risk_state = 'REDUCING' and v_has_allowed_new_risk then
    v_reason := 'REDUCING_NEW_RISK';
  elsif v_risk_state = 'ACTIVE' and v_has_blocked_new_risk then
    v_reason := 'ASSET_COOLDOWN';
  end if;

  if v_reason is null then
    return jsonb_build_object(
      'kill_requested', false,
      'status', 'CONTINUE',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token,
      'risk_version', v_risk_version,
      'risk_receipt_id', v_risk_receipt_id,
      'risk_state', v_risk_state,
      'journal_stage', v_journal_stage,
      'start_risk_version', v_start.risk_version_at_start,
      'start_risk_receipt_id', v_start.risk_receipt_id_at_start
    );
  end if;

  insert into public.brian_shadow_execution_kill_requests(
    runtime_id, dispatch_id, cycle_id,
    risk_version, risk_receipt_id, trading_state,
    reason, journal_stage, observed_at,
    metadata
  ) values (
    p_runtime_id, v_dispatch.dispatch_id, p_cycle_id,
    v_risk_version, v_risk_receipt_id, v_risk_state,
    v_reason, coalesce(v_journal_stage, '<unknown>'), v_now,
    jsonb_build_object(
      'start_risk_version', v_start.risk_version_at_start,
      'start_risk_receipt_id', v_start.risk_receipt_id_at_start,
      'start_risk_state', v_start.risk_state_at_start,
      'start_journal_stage', v_start.journal_stage_at_start,
      'blocked_assets', v_blocked_assets
    )
  )
  on conflict (
    runtime_id, dispatch_id, risk_version, risk_receipt_id, reason
  ) do nothing
  returning request_sequence into v_request_sequence;

  if v_request_sequence is null then
    select request_sequence
      into v_request_sequence
    from public.brian_shadow_execution_kill_requests
    where runtime_id = p_runtime_id
      and dispatch_id = v_dispatch.dispatch_id
      and risk_version = v_risk_version
      and risk_receipt_id = v_risk_receipt_id
      and reason = v_reason;
  end if;

  return jsonb_build_object(
    'kill_requested', true,
    'status', 'KILL_REQUESTED',
    'reason', v_reason,
    'request_sequence', v_request_sequence,
    'runtime_id', p_runtime_id,
    'dispatch_id', v_dispatch.dispatch_id,
    'cycle_id', p_cycle_id,
    'runtime_version', v_runtime_version,
    'fencing_token', p_fencing_token,
    'claim_fencing_token', p_claim_fencing_token,
    'risk_version', v_risk_version,
    'risk_receipt_id', v_risk_receipt_id,
    'risk_state', v_risk_state,
    'journal_stage', v_journal_stage,
    'start_risk_version', v_start.risk_version_at_start,
    'start_risk_receipt_id', v_start.risk_receipt_id_at_start
  );
end;
$$;

revoke all on function public.brian_start_shadow_execution_claim(
  text,text,bigint,text,text,bigint
) from public, anon, authenticated;
revoke all on function public.brian_check_shadow_execution_kill_switch(
  text,text,bigint,text,text,bigint
) from public, anon, authenticated;

grant execute on function public.brian_start_shadow_execution_claim(
  text,text,bigint,text,text,bigint
) to service_role;
grant execute on function public.brian_check_shadow_execution_kill_switch(
  text,text,bigint,text,text,bigint
) to service_role;

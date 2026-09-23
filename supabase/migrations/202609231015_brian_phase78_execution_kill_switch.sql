-- Brian Phase 78: post-claim execution kill-switch / cancel-request lifecycle.
--
-- GitHub-only until explicit rollout.
--
-- Phase77 proves worker ownership. Phase78 rechecks the current persisted risk
-- head immediately before execution/resume and again after a potentially long
-- advance. Fresh CYCLE_CREATED work can still be cancelled with zero paper side
-- effect. Once the journal proves paper work started, an incompatible risk head
-- becomes a durable CANCEL_REQUESTED signal while Phase67 recovery/reconciliation
-- is allowed to finish the already-started state transition.

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

create table if not exists public.brian_shadow_execution_cancel_requests (
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  risk_version bigint not null check (risk_version > 0),
  risk_receipt_id text not null,
  journal_stage text not null,
  phase text not null check (phase in ('BEFORE_EXECUTION','AFTER_START')),
  reason text not null,
  requested_at timestamptz not null default now(),
  worker_token text not null,
  claim_fencing_token bigint not null check (claim_fencing_token > 0),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, dispatch_id, risk_receipt_id, phase),
  check (length(risk_receipt_id) = 64),
  check (length(trim(reason)) > 0)
);

create index if not exists brian_shadow_execution_cancel_requests_cycle_idx
  on public.brian_shadow_execution_cancel_requests(
    runtime_id, cycle_id, requested_at desc
  );

alter table public.brian_shadow_execution_cancel_requests enable row level security;
revoke all on public.brian_shadow_execution_cancel_requests
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_execution_cancel_requests_append_only
  on public.brian_shadow_execution_cancel_requests;
create trigger brian_shadow_execution_cancel_requests_append_only
  before update or delete on public.brian_shadow_execution_cancel_requests
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_shadow_execution_kill_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  worker_token text not null,
  claim_fencing_token bigint not null,
  event text not null check (
    event in (
      'PROCEED',
      'RESUME_ONLY',
      'CANCELLED_BEFORE_EXECUTION',
      'CANCEL_REQUESTED',
      'COMPLETED',
      'ABORTED',
      'CLAIM_LOST',
      'LEASE_LOST',
      'RISK_STATE_UNAVAILABLE'
    )
  ),
  observed_at timestamptz not null default now(),
  risk_version bigint,
  risk_receipt_id text,
  journal_stage text,
  reason text,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_shadow_execution_kill_events_runtime_idx
  on public.brian_shadow_execution_kill_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_shadow_execution_kill_events enable row level security;
revoke all on public.brian_shadow_execution_kill_events
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_execution_kill_events_append_only
  on public.brian_shadow_execution_kill_events;
create trigger brian_shadow_execution_kill_events_append_only
  before update or delete on public.brian_shadow_execution_kill_events
  for each row execute function public.brian_reject_mutation();

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
  v_owner text;
  v_runtime_fence bigint;
  v_runtime_version bigint;
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
  v_reason text;
  v_status text;
  v_phase text;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_cycle_id is null or length(p_cycle_id) <> 64
     or nullif(trim(p_worker_token), '') is null
     or p_claim_fencing_token is null or p_claim_fencing_token <= 0 then
    raise exception 'PHASE78_KILL: valid runtime/lease/cycle/worker/claim fence required';
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

  select *
    into v_dispatch
  from public.brian_shadow_execution_dispatches
  where runtime_id = p_runtime_id and cycle_id = p_cycle_id;

  if v_dispatch.dispatch_id is null then
    raise exception 'PHASE78_KILL: dispatch missing for %', p_cycle_id;
  end if;

  if v_owner is null
     or v_owner <> p_owner_token
     or v_runtime_fence <> p_fencing_token
     or v_lease_until <= v_now then
    v_status := 'LEASE_LOST';
    insert into public.brian_shadow_execution_kill_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, v_status, v_now
    );
    return jsonb_build_object(
      'status', v_status,
      'proceed', false,
      'cancel_requested', false,
      'terminal', false,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version
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
    v_status := 'CLAIM_LOST';
    insert into public.brian_shadow_execution_kill_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, v_status, v_now
    );
    return jsonb_build_object(
      'status', v_status,
      'proceed', false,
      'cancel_requested', false,
      'terminal', false,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version
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

  if v_journal_stage is null then
    raise exception 'PHASE78_KILL: cycle % missing from runtime journal', p_cycle_id;
  end if;

  if v_journal_stage = 'COMMITTED' then
    insert into public.brian_shadow_execution_kill_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at, journal_stage
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, 'COMPLETED', v_now, v_journal_stage
    );
    return jsonb_build_object(
      'status', 'COMPLETED',
      'proceed', false,
      'cancel_requested', false,
      'terminal', true,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'journal_stage', v_journal_stage
    );
  end if;

  if v_journal_stage = 'ABORTED' then
    insert into public.brian_shadow_execution_kill_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at, journal_stage
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, 'ABORTED', v_now, v_journal_stage
    );
    return jsonb_build_object(
      'status', 'ABORTED',
      'proceed', false,
      'cancel_requested', true,
      'terminal', true,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'journal_stage', v_journal_stage
    );
  end if;

  select cycle_payload
    into v_cycle_payload
  from public.brian_shadow_runtime_cycles
  where runtime_id = p_runtime_id and cycle_id = p_cycle_id;

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

  if v_cycle_payload is null
     or v_risk_version is null
     or v_risk_receipt is null
     or jsonb_typeof(v_risk_receipt) <> 'object' then
    v_status := 'RISK_STATE_UNAVAILABLE';
    insert into public.brian_shadow_execution_kill_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at, journal_stage
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, v_status, v_now, v_journal_stage
    );
    return jsonb_build_object(
      'status', v_status,
      'proceed', false,
      'cancel_requested', false,
      'terminal', false,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'journal_stage', v_journal_stage
    );
  end if;

  v_risk_receipt_id := nullif(trim(v_risk_receipt->>'receipt_id'), '');
  v_risk_state := nullif(trim(v_risk_receipt->>'trading_state'), '');
  v_blocked_assets := coalesce(v_risk_receipt->'blocked_assets', '[]'::jsonb);

  if v_risk_receipt_id is null or length(v_risk_receipt_id) <> 64
     or v_risk_state not in ('ACTIVE','REDUCING','HALTED')
     or jsonb_typeof(v_blocked_assets) <> 'array' then
    raise exception 'PHASE78_KILL: malformed current risk receipt';
  end if;

  select exists(
    select 1
    from jsonb_array_elements(coalesce(v_cycle_payload->'items','[]'::jsonb)) item
    where coalesce((item->'risk_receipt'->>'allowed')::boolean, false)
      and not coalesce((item->'risk_receipt'->>'reduce_only')::boolean, false)
  ) into v_has_allowed_new_risk;

  select exists(
    select 1
    from jsonb_array_elements(coalesce(v_cycle_payload->'items','[]'::jsonb)) item
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
    v_status := case
      when v_journal_stage = 'CYCLE_CREATED' then 'PROCEED'
      else 'RESUME_ONLY'
    end;

    insert into public.brian_shadow_execution_kill_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, event, observed_at,
      risk_version, risk_receipt_id, journal_stage,
      metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, v_status, v_now,
      v_risk_version, v_risk_receipt_id, v_journal_stage,
      jsonb_build_object('trading_state', v_risk_state)
    );

    return jsonb_build_object(
      'status', v_status,
      'proceed', true,
      'cancel_requested', false,
      'terminal', false,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'risk_version', v_risk_version,
      'risk_receipt_id', v_risk_receipt_id,
      'journal_stage', v_journal_stage,
      'reason', null
    );
  end if;

  v_phase := case
    when v_journal_stage = 'CYCLE_CREATED' then 'BEFORE_EXECUTION'
    else 'AFTER_START'
  end;

  insert into public.brian_shadow_execution_cancel_requests(
    runtime_id, dispatch_id, cycle_id,
    risk_version, risk_receipt_id, journal_stage,
    phase, reason, requested_at,
    worker_token, claim_fencing_token
  ) values (
    p_runtime_id, v_dispatch.dispatch_id, p_cycle_id,
    v_risk_version, v_risk_receipt_id, v_journal_stage,
    v_phase, v_reason, v_now,
    p_worker_token, p_claim_fencing_token
  )
  on conflict (runtime_id, dispatch_id, risk_receipt_id, phase) do nothing;

  if v_phase = 'BEFORE_EXECUTION' then
    update public.brian_shadow_execution_claims
      set status = 'CANCELLED_BEFORE_EXECUTION',
          claim_until = null,
          risk_version_at_decision = v_risk_version,
          risk_receipt_id_at_decision = v_risk_receipt_id,
          journal_stage_at_decision = v_journal_stage,
          resume_only = false,
          cancel_reason = 'phase78:' || v_reason,
          updated_at = v_now
      where runtime_id = p_runtime_id
        and dispatch_id = v_dispatch.dispatch_id
        and status = 'CLAIMED'
        and worker_token = p_worker_token
        and claim_fencing_token = p_claim_fencing_token;

    v_status := 'CANCELLED_BEFORE_EXECUTION';
  else
    v_status := 'CANCEL_REQUESTED';
  end if;

  insert into public.brian_shadow_execution_kill_events(
    runtime_id, dispatch_id, cycle_id, worker_token,
    claim_fencing_token, event, observed_at,
    risk_version, risk_receipt_id, journal_stage,
    reason, metadata
  ) values (
    p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, p_worker_token,
    p_claim_fencing_token, v_status, v_now,
    v_risk_version, v_risk_receipt_id, v_journal_stage,
    v_reason,
    jsonb_build_object(
      'phase', v_phase,
      'trading_state', v_risk_state,
      'blocked_assets', v_blocked_assets,
      'has_allowed_new_risk', v_has_allowed_new_risk
    )
  );

  return jsonb_build_object(
    'status', v_status,
    'proceed', v_phase = 'AFTER_START',
    'cancel_requested', true,
    'terminal', v_phase = 'BEFORE_EXECUTION',
    'runtime_id', p_runtime_id,
    'dispatch_id', v_dispatch.dispatch_id,
    'cycle_id', p_cycle_id,
    'runtime_version', v_runtime_version,
    'risk_version', v_risk_version,
    'risk_receipt_id', v_risk_receipt_id,
    'journal_stage', v_journal_stage,
    'reason', v_reason,
    'phase', v_phase
  );
end;
$$;

create or replace function public.brian_read_shadow_execution_cancel_requests(
  p_runtime_id text,
  p_cycle_id text
) returns setof public.brian_shadow_execution_cancel_requests
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  select *
  from public.brian_shadow_execution_cancel_requests
  where runtime_id = p_runtime_id and cycle_id = p_cycle_id
  order by requested_at asc, risk_version asc;
$$;

revoke all on function public.brian_check_shadow_execution_kill_switch(
  text,text,bigint,text,text,bigint
) from public, anon, authenticated;
revoke all on function public.brian_read_shadow_execution_cancel_requests(text,text)
  from public, anon, authenticated;

grant execute on function public.brian_check_shadow_execution_kill_switch(
  text,text,bigint,text,text,bigint
) to service_role;
grant execute on function public.brian_read_shadow_execution_cancel_requests(text,text)
  to service_role;

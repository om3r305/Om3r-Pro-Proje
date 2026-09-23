-- Brian Phase 81 DRAFT SQL: durable post-cancel recovery directive.
--
-- IMPORTANT: This is intentionally NOT an official Supabase migration yet.
-- The PR remains draft/undeployed. At rollout freeze this SQL must be converted
-- with `supabase migration new`, then rerun through the full real-Postgres
-- suite before any deployment.
--
-- Phase78 makes an AFTER_START cancel request durable. Phase79/80 ensure the
-- already-started cycle is recovered and made authoritative without pretending
-- side effects never happened. Phase81 converts that durable cancel obligation
-- into an immutable, evidence-backed recovery directive.
--
-- Critical invariant:
--   * HALTED never auto-submits reductions (Phase56 forbids every submission).
--   * REDUCING/ACTIVE may prepare a REDUCE_ONLY rollback target.
--   * recovery targets the original cycle's authoritative pre-cycle position,
--     never blindly flattens pre-existing exposure.
--   * if the current head is no longer the original cycle's reconciled commit,
--     automatic rollback is refused as HEAD_MOVED.

create table if not exists public.brian_shadow_cancel_recovery_directives (
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  cancel_risk_version bigint not null check (cancel_risk_version > 0),
  cancel_risk_receipt_id text not null,
  cancel_reason text not null check (
    cancel_reason in ('HALTED','REDUCING_NEW_RISK','ASSET_COOLDOWN')
  ),
  source_runtime_version bigint not null check (source_runtime_version > 0),
  pre_state_id text not null,
  current_state_id text not null,
  current_risk_version bigint not null check (current_risk_version > 0),
  current_risk_receipt_id text not null,
  current_risk_state text not null check (
    current_risk_state in ('ACTIVE','REDUCING','HALTED')
  ),
  recovery_status text not null check (
    recovery_status in (
      'READY_REDUCE_ONLY',
      'WAIT_RISK_RELEASE',
      'NO_RECOVERY_REQUIRED',
      'MANUAL_REVIEW'
    )
  ),
  recovery_legs jsonb not null default '[]'::jsonb
    check (jsonb_typeof(recovery_legs) = 'array'),
  unsafe_assets jsonb not null default '[]'::jsonb
    check (jsonb_typeof(unsafe_assets) = 'array'),
  prepared_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, dispatch_id, cancel_risk_receipt_id),
  check (length(cancel_risk_receipt_id) = 64),
  check (length(pre_state_id) = 64),
  check (length(current_state_id) = 64),
  check (length(current_risk_receipt_id) = 64)
);

create index if not exists brian_shadow_cancel_recovery_cycle_idx
  on public.brian_shadow_cancel_recovery_directives(
    runtime_id, cycle_id, prepared_at desc
  );

alter table public.brian_shadow_cancel_recovery_directives enable row level security;
revoke all on public.brian_shadow_cancel_recovery_directives
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_cancel_recovery_directives_append_only
  on public.brian_shadow_cancel_recovery_directives;
create trigger brian_shadow_cancel_recovery_directives_append_only
  before update or delete on public.brian_shadow_cancel_recovery_directives
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_shadow_cancel_recovery_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  dispatch_id text,
  cycle_id text not null,
  runtime_version bigint,
  cancel_risk_version bigint,
  cancel_risk_receipt_id text,
  event text not null check (
    event in (
      'PREPARED',
      'DUPLICATE',
      'NO_CANCEL_REQUEST',
      'WAIT_ORIGINAL_COMMIT',
      'FOREIGN_CYCLE_ACTIVE',
      'HEAD_MOVED',
      'LEASE_LOST',
      'RUNTIME_VERSION_CONFLICT',
      'RISK_STATE_UNAVAILABLE',
      'EVIDENCE_INVALID'
    )
  ),
  observed_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_shadow_cancel_recovery_events_runtime_idx
  on public.brian_shadow_cancel_recovery_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_shadow_cancel_recovery_events enable row level security;
revoke all on public.brian_shadow_cancel_recovery_events
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_cancel_recovery_events_append_only
  on public.brian_shadow_cancel_recovery_events;
create trigger brian_shadow_cancel_recovery_events_append_only
  before update or delete on public.brian_shadow_cancel_recovery_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_prepare_shadow_cancel_recovery(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_cycle_id text,
  p_expected_runtime_version bigint
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
  v_cancel public.brian_shadow_execution_cancel_requests%rowtype;
  v_existing public.brian_shadow_cancel_recovery_directives%rowtype;
  v_start public.brian_shadow_execution_starts%rowtype;
  v_cycle_payload jsonb;
  v_journal_stage text;
  v_foreign_cycle_id text;
  v_foreign_cycle_stage text;
  v_ledger jsonb;
  v_head_state_id text;
  v_pre_state_id text;
  v_commit_state_id text;
  v_pre_state jsonb;
  v_current_state jsonb;
  v_cancel_receipt jsonb;
  v_cancel_blocked_assets jsonb;
  v_current_risk_version bigint;
  v_current_risk_head_entry_id text;
  v_current_risk_receipt jsonb;
  v_current_risk_receipt_id text;
  v_current_risk_state text;
  v_asset text;
  v_before_weight double precision;
  v_current_weight double precision;
  v_legs jsonb := '[]'::jsonb;
  v_unsafe jsonb := '[]'::jsonb;
  v_candidate_count integer := 0;
  v_status text;
  v_leg jsonb;
  v_epsilon double precision := 1e-12;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_cycle_id is null or length(p_cycle_id) <> 64
     or p_expected_runtime_version is null or p_expected_runtime_version <= 0 then
    raise exception 'PHASE81_RECOVERY: valid runtime/lease/cycle/version required';
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
    insert into public.brian_shadow_cancel_recovery_events(
      runtime_id, cycle_id, runtime_version, event, observed_at,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, v_runtime_version, 'LEASE_LOST', v_now,
      jsonb_build_object(
        'requested_fence', p_fencing_token,
        'current_fence', v_runtime_fence
      )
    );
    return jsonb_build_object(
      'prepared', false,
      'duplicate', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', coalesce(v_runtime_version,0),
      'fencing_token', coalesce(v_runtime_fence,p_fencing_token)
    );
  end if;

  if v_runtime_version <> p_expected_runtime_version then
    insert into public.brian_shadow_cancel_recovery_events(
      runtime_id, cycle_id, runtime_version, event, observed_at,
      metadata
    ) values (
      p_runtime_id, p_cycle_id, v_runtime_version,
      'RUNTIME_VERSION_CONFLICT', v_now,
      jsonb_build_object('expected_runtime_version', p_expected_runtime_version)
    );
    return jsonb_build_object(
      'prepared', false,
      'duplicate', false,
      'status', 'RUNTIME_VERSION_CONFLICT',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  select *
    into v_dispatch
  from public.brian_shadow_execution_dispatches
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if not found then
    raise exception 'PHASE81_RECOVERY: dispatch missing for %', p_cycle_id;
  end if;

  select *
    into v_cancel
  from public.brian_shadow_execution_cancel_requests
  where runtime_id = p_runtime_id
    and dispatch_id = v_dispatch.dispatch_id
    and phase = 'AFTER_START'
  order by requested_at asc, risk_version asc, risk_receipt_id asc
  limit 1;

  if not found then
    insert into public.brian_shadow_cancel_recovery_events(
      runtime_id, dispatch_id, cycle_id, runtime_version,
      event, observed_at
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, v_runtime_version,
      'NO_CANCEL_REQUEST', v_now
    );
    return jsonb_build_object(
      'prepared', false,
      'duplicate', false,
      'status', 'NO_CANCEL_REQUEST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  select *
    into v_existing
  from public.brian_shadow_cancel_recovery_directives
  where runtime_id = p_runtime_id
    and dispatch_id = v_dispatch.dispatch_id
    and cancel_risk_receipt_id = v_cancel.risk_receipt_id;

  if found then
    insert into public.brian_shadow_cancel_recovery_events(
      runtime_id, dispatch_id, cycle_id, runtime_version,
      cancel_risk_version, cancel_risk_receipt_id,
      event, observed_at, metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, v_runtime_version,
      v_existing.cancel_risk_version, v_existing.cancel_risk_receipt_id,
      'DUPLICATE', v_now,
      jsonb_build_object('recovery_status', v_existing.recovery_status)
    );
    return jsonb_build_object(
      'prepared', true,
      'duplicate', true,
      'status', 'DUPLICATE',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_existing.source_runtime_version,
      'fencing_token', p_fencing_token,
      'cancel_risk_version', v_existing.cancel_risk_version,
      'cancel_risk_receipt_id', v_existing.cancel_risk_receipt_id,
      'cancel_reason', v_existing.cancel_reason,
      'pre_state_id', v_existing.pre_state_id,
      'current_state_id', v_existing.current_state_id,
      'current_risk_version', v_existing.current_risk_version,
      'current_risk_receipt_id', v_existing.current_risk_receipt_id,
      'current_risk_state', v_existing.current_risk_state,
      'recovery_status', v_existing.recovery_status,
      'recovery_legs', v_existing.recovery_legs,
      'unsafe_assets', v_existing.unsafe_assets
    );
  end if;

  select *
    into v_start
  from public.brian_shadow_execution_starts
  where runtime_id = p_runtime_id
    and dispatch_id = v_dispatch.dispatch_id;

  if not found then
    insert into public.brian_shadow_cancel_recovery_events(
      runtime_id, dispatch_id, cycle_id, runtime_version,
      cancel_risk_version, cancel_risk_receipt_id,
      event, observed_at, metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, v_runtime_version,
      v_cancel.risk_version, v_cancel.risk_receipt_id,
      'EVIDENCE_INVALID', v_now,
      jsonb_build_object('reason', 'START_MISSING')
    );
    return jsonb_build_object(
      'prepared', false,
      'duplicate', false,
      'status', 'EVIDENCE_INVALID',
      'reason', 'START_MISSING',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
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
    insert into public.brian_shadow_cancel_recovery_events(
      runtime_id, dispatch_id, cycle_id, runtime_version,
      cancel_risk_version, cancel_risk_receipt_id,
      event, observed_at, metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, v_runtime_version,
      v_cancel.risk_version, v_cancel.risk_receipt_id,
      'WAIT_ORIGINAL_COMMIT', v_now,
      jsonb_build_object('journal_stage', v_journal_stage)
    );
    return jsonb_build_object(
      'prepared', false,
      'duplicate', false,
      'status', 'WAIT_ORIGINAL_COMMIT',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'cancel_risk_version', v_cancel.risk_version,
      'cancel_risk_receipt_id', v_cancel.risk_receipt_id,
      'cancel_reason', v_cancel.reason,
      'journal_stage', v_journal_stage
    );
  end if;

  -- Phase86 admission interlock: do not freeze an immutable recovery
  -- directive while another journal cycle is still active. A pre-authorized
  -- CYCLE_CREATED candidate must be quarantined/aborted first, then Phase81 is
  -- retried against the new authoritative runtime version.
  select latest.cycle_id, latest.stage
    into v_foreign_cycle_id, v_foreign_cycle_stage
  from (
    select distinct on (e.value->>'cycle_id')
      e.value->>'cycle_id' as cycle_id,
      e.value->>'stage' as stage,
      e.ord
    from jsonb_array_elements(
      coalesce(v_checkpoint->'journal_manifest'->'entries','[]'::jsonb)
    ) with ordinality e(value,ord)
    where nullif(trim(e.value->>'cycle_id'),'') is not null
    order by e.value->>'cycle_id', e.ord desc
  ) latest
  where latest.cycle_id <> p_cycle_id
    and latest.stage not in ('COMMITTED','ABORTED')
  order by latest.ord asc
  limit 1;

  if v_foreign_cycle_id is not null then
    insert into public.brian_shadow_cancel_recovery_events(
      runtime_id, dispatch_id, cycle_id, runtime_version,
      cancel_risk_version, cancel_risk_receipt_id,
      event, observed_at, metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, v_runtime_version,
      v_cancel.risk_version, v_cancel.risk_receipt_id,
      'FOREIGN_CYCLE_ACTIVE', v_now,
      jsonb_build_object(
        'foreign_cycle_id',v_foreign_cycle_id,
        'foreign_cycle_stage',v_foreign_cycle_stage
      )
    );
    return jsonb_build_object(
      'prepared',false,
      'duplicate',false,
      'status','FOREIGN_CYCLE_ACTIVE',
      'runtime_id',p_runtime_id,
      'dispatch_id',v_dispatch.dispatch_id,
      'cycle_id',p_cycle_id,
      'runtime_version',v_runtime_version,
      'fencing_token',p_fencing_token,
      'cancel_risk_version',v_cancel.risk_version,
      'cancel_risk_receipt_id',v_cancel.risk_receipt_id,
      'cancel_reason',v_cancel.reason,
      'foreign_cycle_id',v_foreign_cycle_id,
      'foreign_cycle_stage',v_foreign_cycle_stage
    );
  end if;

  v_ledger := v_checkpoint->'runtime_checkpoint'->'shadow_ledger_manifest';
  if jsonb_typeof(v_ledger) <> 'object' then
    raise exception 'PHASE81_RECOVERY: runtime checkpoint missing Phase60 ledger manifest';
  end if;

  v_head_state_id := nullif(trim(v_ledger->>'head_state_id'), '');
  if v_head_state_id is null or length(v_head_state_id) <> 64 then
    raise exception 'PHASE81_RECOVERY: invalid Phase60 head_state_id';
  end if;

  select t.value->>'before_state_id'
    into v_pre_state_id
  from jsonb_array_elements(
    coalesce(v_ledger->'transitions', '[]'::jsonb)
  ) with ordinality t(value, ord)
  where t.value->>'cycle_id' = p_cycle_id
    and t.value->>'kind' = 'CYCLE_PROPOSED'
  order by t.ord asc
  limit 1;

  select t.value->>'after_state_id'
    into v_commit_state_id
  from jsonb_array_elements(
    coalesce(v_ledger->'transitions', '[]'::jsonb)
  ) with ordinality t(value, ord)
  where t.value->>'cycle_id' = p_cycle_id
    and t.value->>'kind' = 'RECONCILED_COMMIT'
  order by t.ord desc
  limit 1;

  if v_pre_state_id is null or length(v_pre_state_id) <> 64
     or v_commit_state_id is null or length(v_commit_state_id) <> 64 then
    insert into public.brian_shadow_cancel_recovery_events(
      runtime_id, dispatch_id, cycle_id, runtime_version,
      cancel_risk_version, cancel_risk_receipt_id,
      event, observed_at, metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, v_runtime_version,
      v_cancel.risk_version, v_cancel.risk_receipt_id,
      'EVIDENCE_INVALID', v_now,
      jsonb_build_object(
        'reason', 'MISSING_PHASE60_TRANSITIONS',
        'pre_state_id', v_pre_state_id,
        'commit_state_id', v_commit_state_id
      )
    );
    return jsonb_build_object(
      'prepared', false,
      'duplicate', false,
      'status', 'EVIDENCE_INVALID',
      'reason', 'MISSING_PHASE60_TRANSITIONS',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  if v_commit_state_id <> v_head_state_id then
    insert into public.brian_shadow_cancel_recovery_events(
      runtime_id, dispatch_id, cycle_id, runtime_version,
      cancel_risk_version, cancel_risk_receipt_id,
      event, observed_at, metadata
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, v_runtime_version,
      v_cancel.risk_version, v_cancel.risk_receipt_id,
      'HEAD_MOVED', v_now,
      jsonb_build_object(
        'cycle_commit_state_id', v_commit_state_id,
        'current_head_state_id', v_head_state_id
      )
    );
    return jsonb_build_object(
      'prepared', false,
      'duplicate', false,
      'status', 'HEAD_MOVED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'pre_state_id', v_pre_state_id,
      'cycle_commit_state_id', v_commit_state_id,
      'current_state_id', v_head_state_id
    );
  end if;

  v_pre_state := v_ledger->'states'->v_pre_state_id;
  v_current_state := v_ledger->'states'->v_head_state_id;
  if jsonb_typeof(v_pre_state) <> 'object'
     or jsonb_typeof(v_current_state) <> 'object' then
    raise exception 'PHASE81_RECOVERY: Phase60 state payload missing';
  end if;

  select entry_payload->'receipt'
    into v_cancel_receipt
  from public.brian_operational_risk_entries
  where runtime_id = p_runtime_id
    and receipt_id = v_cancel.risk_receipt_id
  order by sequence desc
  limit 1;

  if v_cancel_receipt is null or jsonb_typeof(v_cancel_receipt) <> 'object' then
    raise exception 'PHASE81_RECOVERY: cancel risk receipt evidence missing';
  end if;
  v_cancel_blocked_assets := coalesce(v_cancel_receipt->'blocked_assets', '[]'::jsonb);
  if jsonb_typeof(v_cancel_blocked_assets) <> 'array' then
    raise exception 'PHASE81_RECOVERY: cancel blocked_assets evidence malformed';
  end if;

  select version, head_entry_id
    into v_current_risk_version, v_current_risk_head_entry_id
  from public.brian_operational_risk_heads
  where runtime_id = p_runtime_id
  for update;

  select entry_payload->'receipt'
    into v_current_risk_receipt
  from public.brian_operational_risk_entries
  where runtime_id = p_runtime_id
    and entry_id = v_current_risk_head_entry_id;

  if v_current_risk_version is null
     or v_current_risk_receipt is null
     or jsonb_typeof(v_current_risk_receipt) <> 'object' then
    insert into public.brian_shadow_cancel_recovery_events(
      runtime_id, dispatch_id, cycle_id, runtime_version,
      cancel_risk_version, cancel_risk_receipt_id,
      event, observed_at
    ) values (
      p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, v_runtime_version,
      v_cancel.risk_version, v_cancel.risk_receipt_id,
      'RISK_STATE_UNAVAILABLE', v_now
    );
    return jsonb_build_object(
      'prepared', false,
      'duplicate', false,
      'status', 'RISK_STATE_UNAVAILABLE',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch.dispatch_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'fencing_token', p_fencing_token
    );
  end if;

  v_current_risk_receipt_id := nullif(trim(v_current_risk_receipt->>'receipt_id'), '');
  v_current_risk_state := nullif(trim(v_current_risk_receipt->>'trading_state'), '');
  if v_current_risk_receipt_id is null
     or length(v_current_risk_receipt_id) <> 64
     or v_current_risk_state not in ('ACTIVE','REDUCING','HALTED') then
    raise exception 'PHASE81_RECOVERY: malformed current risk receipt';
  end if;

  select cycle_payload
    into v_cycle_payload
  from public.brian_shadow_runtime_cycles
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if v_cycle_payload is null or jsonb_typeof(v_cycle_payload) <> 'object' then
    raise exception 'PHASE81_RECOVERY: durable cycle body missing';
  end if;

  for v_asset in
    select distinct item->>'asset_id'
    from jsonb_array_elements(
      coalesce(v_cycle_payload->'items', '[]'::jsonb)
    ) item
    where nullif(trim(item->>'asset_id'), '') is not null
      and coalesce((item->'risk_receipt'->>'allowed')::boolean, false)
      and not coalesce((item->'risk_receipt'->>'reduce_only')::boolean, false)
      and (
        v_cancel.reason <> 'ASSET_COOLDOWN'
        or exists(
          select 1
          from jsonb_array_elements_text(v_cancel_blocked_assets) blocked(asset)
          where blocked.asset = item->>'asset_id'
        )
      )
    order by 1
  loop
    v_candidate_count := v_candidate_count + 1;

    select coalesce((p.value->>1)::double precision, 0.0)
      into v_before_weight
    from jsonb_array_elements(
      coalesce(v_pre_state->'position_weights', '[]'::jsonb)
    ) p(value)
    where p.value->>0 = v_asset
    limit 1;
    v_before_weight := coalesce(v_before_weight, 0.0);

    select coalesce((p.value->>1)::double precision, 0.0)
      into v_current_weight
    from jsonb_array_elements(
      coalesce(v_current_state->'position_weights', '[]'::jsonb)
    ) p(value)
    where p.value->>0 = v_asset
    limit 1;
    v_current_weight := coalesce(v_current_weight, 0.0);

    if abs(v_current_weight - v_before_weight) <= v_epsilon
       or abs(v_current_weight) <= abs(v_before_weight) + v_epsilon then
      continue;
    end if;

    if abs(v_before_weight) > v_epsilon
       and (
         sign(v_before_weight) <> sign(v_current_weight)
         or abs(v_before_weight) > abs(v_current_weight) + v_epsilon
       ) then
      v_unsafe := v_unsafe || jsonb_build_array(jsonb_build_object(
        'asset_id', v_asset,
        'before_weight', v_before_weight,
        'current_weight', v_current_weight,
        'reason', 'ROLLBACK_NOT_REDUCE_ONLY'
      ));
      continue;
    end if;

    v_leg := jsonb_build_object(
      'asset_id', v_asset,
      'before_weight', v_before_weight,
      'current_weight', v_current_weight,
      'target_weight', v_before_weight,
      'reduce_weight', abs(v_current_weight) - abs(v_before_weight),
      'current_direction', case when v_current_weight > 0 then 1 else -1 end,
      'order_direction', case when v_current_weight > 0 then -1 else 1 end,
      'reduce_only', true
    );
    v_legs := v_legs || jsonb_build_array(v_leg);
  end loop;

  if jsonb_array_length(v_unsafe) > 0 then
    v_status := 'MANUAL_REVIEW';
  elsif jsonb_array_length(v_legs) = 0 then
    v_status := 'NO_RECOVERY_REQUIRED';
  elsif v_current_risk_state = 'HALTED' then
    v_status := 'WAIT_RISK_RELEASE';
  else
    v_status := 'READY_REDUCE_ONLY';
  end if;

  insert into public.brian_shadow_cancel_recovery_directives(
    runtime_id, dispatch_id, cycle_id,
    cancel_risk_version, cancel_risk_receipt_id, cancel_reason,
    source_runtime_version, pre_state_id, current_state_id,
    current_risk_version, current_risk_receipt_id, current_risk_state,
    recovery_status, recovery_legs, unsafe_assets, prepared_at
  ) values (
    p_runtime_id, v_dispatch.dispatch_id, p_cycle_id,
    v_cancel.risk_version, v_cancel.risk_receipt_id, v_cancel.reason,
    v_runtime_version, v_pre_state_id, v_head_state_id,
    v_current_risk_version, v_current_risk_receipt_id, v_current_risk_state,
    v_status, v_legs, v_unsafe, v_now
  );

  insert into public.brian_shadow_cancel_recovery_events(
    runtime_id, dispatch_id, cycle_id, runtime_version,
    cancel_risk_version, cancel_risk_receipt_id,
    event, observed_at, metadata
  ) values (
    p_runtime_id, v_dispatch.dispatch_id, p_cycle_id, v_runtime_version,
    v_cancel.risk_version, v_cancel.risk_receipt_id,
    'PREPARED', v_now,
    jsonb_build_object(
      'cancel_reason', v_cancel.reason,
      'current_risk_version', v_current_risk_version,
      'current_risk_receipt_id', v_current_risk_receipt_id,
      'current_risk_state', v_current_risk_state,
      'candidate_count', v_candidate_count,
      'recovery_status', v_status,
      'recovery_leg_count', jsonb_array_length(v_legs),
      'unsafe_asset_count', jsonb_array_length(v_unsafe)
    )
  );

  return jsonb_build_object(
    'prepared', true,
    'duplicate', false,
    'status', 'PREPARED',
    'runtime_id', p_runtime_id,
    'dispatch_id', v_dispatch.dispatch_id,
    'cycle_id', p_cycle_id,
    'runtime_version', v_runtime_version,
    'fencing_token', p_fencing_token,
    'cancel_risk_version', v_cancel.risk_version,
    'cancel_risk_receipt_id', v_cancel.risk_receipt_id,
    'cancel_reason', v_cancel.reason,
    'pre_state_id', v_pre_state_id,
    'current_state_id', v_head_state_id,
    'current_risk_version', v_current_risk_version,
    'current_risk_receipt_id', v_current_risk_receipt_id,
    'current_risk_state', v_current_risk_state,
    'recovery_status', v_status,
    'recovery_legs', v_legs,
    'unsafe_assets', v_unsafe
  );
end;
$$;

create or replace function public.brian_read_shadow_cancel_recovery(
  p_runtime_id text,
  p_cycle_id text
) returns setof public.brian_shadow_cancel_recovery_directives
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  select *
  from public.brian_shadow_cancel_recovery_directives
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id
  order by prepared_at asc, cancel_risk_version asc;
$$;

revoke all on function public.brian_prepare_shadow_cancel_recovery(
  text,text,bigint,text,bigint
) from public, anon, authenticated;
revoke all on function public.brian_read_shadow_cancel_recovery(text,text)
  from public, anon, authenticated;

grant execute on function public.brian_prepare_shadow_cancel_recovery(
  text,text,bigint,text,bigint
) to service_role;
grant execute on function public.brian_read_shadow_cancel_recovery(text,text)
  to service_role;

-- Brian Phase 84 DRAFT SQL: recovery-claim-fenced durable checkpoint commit.
--
-- IMPORTANT: draft/undeployed SQL. At rollout freeze create the official
-- Supabase migration with `supabase migration new`, then rerun the complete
-- Postgres suite before deployment.
--
-- Phase83 proves the recovery point-of-no-return. Phase84 carries the current
-- recovery claim fence through the recovery cycle's durable write-ahead and
-- later authoritative Phase67/70 checkpoint. The recovery claim is marked
-- COMPLETED only when the recovery cycle itself is durably COMMITTED.
--
-- This remains shadow/paper only.

create table if not exists public.brian_shadow_recovery_commit_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  dispatch_id text,
  original_cycle_id text not null,
  recovery_cycle_id text not null,
  cancel_risk_receipt_id text,
  worker_token text not null,
  recovery_claim_fencing_token bigint not null,
  runtime_fencing_token bigint not null,
  expected_runtime_version bigint,
  committed_runtime_version bigint,
  checkpoint_id text,
  recovery_journal_stage text,
  event text not null check (
    event in (
      'WRITE_AHEAD_COMMITTED',
      'PROGRESS_COMMITTED',
      'RECOVERY_COMPLETED',
      'DUPLICATE_CURRENT',
      'CLAIM_LOST',
      'LEASE_LOST',
      'RUNTIME_VERSION_CONFLICT',
      'START_MISSING',
      'CHECKPOINT_REJECTED',
      'RECOVERY_CYCLE_INVALID'
    )
  ),
  observed_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_shadow_recovery_commit_events_runtime_idx
  on public.brian_shadow_recovery_commit_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_shadow_recovery_commit_events enable row level security;
revoke all on public.brian_shadow_recovery_commit_events
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_recovery_commit_events_append_only
  on public.brian_shadow_recovery_commit_events;
create trigger brian_shadow_recovery_commit_events_append_only
  before update or delete on public.brian_shadow_recovery_commit_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_commit_shadow_recovery_checkpoint(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_original_cycle_id text,
  p_recovery_cycle_id text,
  p_worker_token text,
  p_recovery_claim_fencing_token bigint,
  p_expected_version bigint,
  p_checkpoint jsonb
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
  v_current_checkpoint_id text;
  v_current_checkpoint_payload jsonb;
  v_current_head_state_id text;
  v_lease_until timestamptz;
  v_directive public.brian_shadow_cancel_recovery_directives%rowtype;
  v_claim public.brian_shadow_recovery_claims%rowtype;
  v_start public.brian_shadow_recovery_starts%rowtype;
  v_checkpoint_id text;
  v_journal_stage text;
  v_cycle_payload jsonb;
  v_item jsonb;
  v_leg jsonb;
  v_item_count integer := 0;
  v_commit jsonb;
  v_commit_status text;
  v_commit_version bigint;
  v_current_version bigint;
  v_checkpoint_head_state_id text;
  v_reconciled_after_state_id text;
  v_event text;
  v_terminal boolean := false;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_original_cycle_id is null or length(p_original_cycle_id) <> 64
     or p_recovery_cycle_id is null or length(p_recovery_cycle_id) <> 64
     or p_original_cycle_id = p_recovery_cycle_id
     or nullif(trim(p_worker_token), '') is null
     or p_recovery_claim_fencing_token is null
     or p_recovery_claim_fencing_token <= 0
     or p_expected_version is null or p_expected_version <= 0
     or p_checkpoint is null or jsonb_typeof(p_checkpoint) <> 'object' then
    raise exception 'PHASE84_COMMIT: valid runtime/lease/original-cycle/recovery-cycle/worker/claim/version/checkpoint required';
  end if;

  if coalesce((p_checkpoint->>'live_execution')::boolean, false) then
    raise exception 'PHASE84_COMMIT: live checkpoint rejected';
  end if;

  v_checkpoint_id := nullif(trim(p_checkpoint->>'checkpoint_id'), '');
  if v_checkpoint_id is null or length(v_checkpoint_id) <> 64 then
    raise exception 'PHASE84_COMMIT: checkpoint_id must be a content hash';
  end if;

  v_cycle_payload :=
    p_checkpoint->'journal_manifest'->'cycles'->p_recovery_cycle_id;
  if v_cycle_payload is null
     or jsonb_typeof(v_cycle_payload) <> 'object'
     or nullif(trim(v_cycle_payload->>'cycle_id'), '') <> p_recovery_cycle_id
     or coalesce((v_cycle_payload->>'shadow_only')::boolean, false) is not true
     or coalesce((v_cycle_payload->>'live_execution')::boolean, false) is not false
     or coalesce((v_cycle_payload->>'account_state_mutated')::boolean, false) is not false then
    raise exception 'PHASE84_COMMIT: checkpoint missing valid shadow recovery cycle';
  end if;

  select e.value->>'stage'
    into v_journal_stage
  from jsonb_array_elements(
    coalesce(p_checkpoint->'journal_manifest'->'entries', '[]'::jsonb)
  ) with ordinality e(value, ord)
  where e.value->>'cycle_id' = p_recovery_cycle_id
  order by e.ord desc
  limit 1;

  if v_journal_stage is null
     or v_journal_stage not in (
       'CYCLE_CREATED',
       'PAPER_APPLIED',
       'LOCAL_PROJECTED',
       'RECONCILIATION_REQUIRED',
       'RECONCILED',
       'COMMITTED',
       'ABORTED'
     ) then
    raise exception 'PHASE84_COMMIT: invalid recovery journal stage %',
      coalesce(v_journal_stage, '<missing>');
  end if;

  if jsonb_typeof(v_cycle_payload->'items') <> 'array'
     or jsonb_array_length(v_cycle_payload->'items') = 0 then
    raise exception 'PHASE84_COMMIT: recovery cycle requires executable items';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, checkpoint_id,
         checkpoint_payload, head_state_id, lease_until
    into v_owner, v_runtime_fence, v_runtime_version,
         v_current_checkpoint_id, v_current_checkpoint_payload,
         v_current_head_state_id, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  select *
    into v_directive
  from public.brian_shadow_cancel_recovery_directives
  where runtime_id = p_runtime_id
    and cycle_id = p_original_cycle_id
  order by prepared_at asc, cancel_risk_version asc
  limit 1;

  if not found then
    raise exception 'PHASE84_COMMIT: recovery directive missing';
  end if;

  if v_owner is null
     or v_owner <> p_owner_token
     or v_runtime_fence <> p_fencing_token
     or v_lease_until <= v_now then
    insert into public.brian_shadow_recovery_commit_events(
      runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
      cancel_risk_receipt_id, worker_token,
      recovery_claim_fencing_token, runtime_fencing_token,
      expected_runtime_version, committed_runtime_version,
      checkpoint_id, recovery_journal_stage,
      event, observed_at
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
      p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      p_expected_version, v_runtime_version,
      v_checkpoint_id, v_journal_stage,
      'LEASE_LOST', v_now
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'terminal', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', coalesce(v_runtime_version,0),
      'current_version', coalesce(v_runtime_version,0),
      'fencing_token', coalesce(v_runtime_fence,p_fencing_token),
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
    );
  end if;

  select *
    into v_claim
  from public.brian_shadow_recovery_claims
  where runtime_id = p_runtime_id
    and dispatch_id = v_directive.dispatch_id
    and cancel_risk_receipt_id = v_directive.cancel_risk_receipt_id
  for update;

  -- Exact final lost-response retry: Phase70 checkpoint and Phase82 completion
  -- were committed in the same transaction previously.
  if found and v_claim.status = 'COMPLETED' then
    if v_claim.completion_ref = v_checkpoint_id
       and v_current_checkpoint_id = v_checkpoint_id
       and v_current_checkpoint_payload is not distinct from p_checkpoint
       and v_journal_stage = 'COMMITTED' then
      return jsonb_build_object(
        'committed', true,
        'duplicate', true,
        'terminal', true,
        'status', 'DUPLICATE_CURRENT',
        'runtime_id', p_runtime_id,
        'dispatch_id', v_directive.dispatch_id,
        'original_cycle_id', p_original_cycle_id,
        'recovery_cycle_id', p_recovery_cycle_id,
        'checkpoint_id', v_checkpoint_id,
        'journal_stage', v_journal_stage,
        'version', v_runtime_version,
        'current_version', v_runtime_version,
        'fencing_token', p_fencing_token,
        'recovery_claim_fencing_token', v_claim.claim_fencing_token,
        'head_state_id', v_current_head_state_id
      );
    end if;
    raise exception 'PHASE84_COMMIT_CONFLICT: recovery already completed with different checkpoint';
  end if;

  if not found
     or v_claim.status <> 'CLAIMED'
     or v_claim.worker_token <> p_worker_token
     or v_claim.claim_fencing_token <> p_recovery_claim_fencing_token
     or v_claim.claim_until <= v_now then
    insert into public.brian_shadow_recovery_commit_events(
      runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
      cancel_risk_receipt_id, worker_token,
      recovery_claim_fencing_token, runtime_fencing_token,
      expected_runtime_version, committed_runtime_version,
      checkpoint_id, recovery_journal_stage,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
      p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      p_expected_version, v_runtime_version,
      v_checkpoint_id, v_journal_stage,
      'CLAIM_LOST', v_now,
      jsonb_build_object(
        'current_claim_status', v_claim.status,
        'current_worker_token', v_claim.worker_token,
        'current_claim_fencing_token', v_claim.claim_fencing_token,
        'current_claim_until', v_claim.claim_until
      )
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'terminal', false,
      'status', 'CLAIM_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', v_runtime_version,
      'current_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
    );
  end if;

  select *
    into v_start
  from public.brian_shadow_recovery_starts
  where runtime_id = p_runtime_id
    and dispatch_id = v_directive.dispatch_id
    and cancel_risk_receipt_id = v_directive.cancel_risk_receipt_id;

  if not found then
    insert into public.brian_shadow_recovery_commit_events(
      runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
      cancel_risk_receipt_id, worker_token,
      recovery_claim_fencing_token, runtime_fencing_token,
      expected_runtime_version, committed_runtime_version,
      checkpoint_id, recovery_journal_stage,
      event, observed_at
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
      p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      p_expected_version, v_runtime_version,
      v_checkpoint_id, v_journal_stage,
      'START_MISSING', v_now
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'terminal', false,
      'status', 'START_MISSING',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', v_runtime_version,
      'current_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
    );
  end if;

  -- Validate that every simulated recovery item is an allowed, fully-filled,
  -- reduce-only realization of exactly one immutable Phase83 leg.
  for v_item in
    select value from jsonb_array_elements(v_cycle_payload->'items')
  loop
    v_item_count := v_item_count + 1;
    select value
      into v_leg
    from jsonb_array_elements(v_start.recovery_legs)
    where value->>'asset_id' = v_item->>'asset_id'
    limit 1;

    if v_leg is null
       or coalesce((v_item->'risk_receipt'->>'allowed')::boolean, false) is not true
       or coalesce((v_item->'risk_receipt'->>'reduce_only')::boolean, false) is not true
       or nullif(trim(v_item->'execution_receipt'->>'status'), '') <> 'FILLED'
       or abs(
            (v_item->'risk_receipt'->>'projected_position_weight')::double precision
            - (v_leg->>'target_weight')::double precision
          ) > 1e-12
       or (
         case
           when (v_leg->>'order_direction')::integer > 0 then 'BUY'
           else 'SELL'
         end
       ) <> nullif(trim(v_item->'execution_receipt'->>'side'), '') then
      insert into public.brian_shadow_recovery_commit_events(
        runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
        cancel_risk_receipt_id, worker_token,
        recovery_claim_fencing_token, runtime_fencing_token,
        expected_runtime_version, committed_runtime_version,
        checkpoint_id, recovery_journal_stage,
        event, observed_at,
        metadata
      ) values (
        p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
        p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
        p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
        p_expected_version, v_runtime_version,
        v_checkpoint_id, v_journal_stage,
        'RECOVERY_CYCLE_INVALID', v_now,
        jsonb_build_object('item', v_item, 'matched_leg', v_leg)
      );
      return jsonb_build_object(
        'committed', false,
        'duplicate', false,
        'terminal', false,
        'status', 'RECOVERY_CYCLE_INVALID',
        'runtime_id', p_runtime_id,
        'dispatch_id', v_directive.dispatch_id,
        'original_cycle_id', p_original_cycle_id,
        'recovery_cycle_id', p_recovery_cycle_id,
        'checkpoint_id', v_checkpoint_id,
        'journal_stage', v_journal_stage,
        'version', v_runtime_version,
        'current_version', v_runtime_version,
        'fencing_token', p_fencing_token,
        'recovery_claim_fencing_token', p_recovery_claim_fencing_token
      );
    end if;
  end loop;

  if v_item_count <> jsonb_array_length(v_start.recovery_legs) then
    insert into public.brian_shadow_recovery_commit_events(
      runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
      cancel_risk_receipt_id, worker_token,
      recovery_claim_fencing_token, runtime_fencing_token,
      expected_runtime_version, committed_runtime_version,
      checkpoint_id, recovery_journal_stage,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
      p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      p_expected_version, v_runtime_version,
      v_checkpoint_id, v_journal_stage,
      'RECOVERY_CYCLE_INVALID', v_now,
      jsonb_build_object(
        'item_count', v_item_count,
        'leg_count', jsonb_array_length(v_start.recovery_legs)
      )
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'terminal', false,
      'status', 'RECOVERY_CYCLE_INVALID',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', v_runtime_version,
      'current_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
    );
  end if;

  -- Before the first recovery checkpoint, the database head must still be the
  -- exact state/version anchored by Phase81/83. Later progress commits may use
  -- the newer version created by this same recovery claim.
  if v_runtime_version = v_directive.source_runtime_version
     and v_current_head_state_id is distinct from v_directive.current_state_id then
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'terminal', false,
      'status', 'RUNTIME_VERSION_CONFLICT',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', v_runtime_version,
      'current_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
    );
  end if;

  -- Lost-response retry is accepted only when the current authoritative head is
  -- exactly the submitted checkpoint and the same recovery claim still owns it.
  if v_runtime_version <> p_expected_version then
    if v_current_checkpoint_id = v_checkpoint_id then
      if v_current_checkpoint_payload is distinct from p_checkpoint then
        raise exception 'PHASE84_CHECKPOINT_ID_CONFLICT: current checkpoint id reused with different payload';
      end if;

      return jsonb_build_object(
        'committed', true,
        'duplicate', true,
        'terminal', v_journal_stage = 'COMMITTED',
        'status', 'DUPLICATE_CURRENT',
        'runtime_id', p_runtime_id,
        'dispatch_id', v_directive.dispatch_id,
        'original_cycle_id', p_original_cycle_id,
        'recovery_cycle_id', p_recovery_cycle_id,
        'checkpoint_id', v_checkpoint_id,
        'journal_stage', v_journal_stage,
        'version', v_runtime_version,
        'current_version', v_runtime_version,
        'fencing_token', p_fencing_token,
        'recovery_claim_fencing_token', p_recovery_claim_fencing_token,
        'head_state_id', v_current_head_state_id
      );
    end if;

    insert into public.brian_shadow_recovery_commit_events(
      runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
      cancel_risk_receipt_id, worker_token,
      recovery_claim_fencing_token, runtime_fencing_token,
      expected_runtime_version, committed_runtime_version,
      checkpoint_id, recovery_journal_stage,
      event, observed_at
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
      p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      p_expected_version, v_runtime_version,
      v_checkpoint_id, v_journal_stage,
      'RUNTIME_VERSION_CONFLICT', v_now
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'terminal', false,
      'status', 'RUNTIME_VERSION_CONFLICT',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', v_runtime_version,
      'current_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
    );
  end if;

  if v_journal_stage = 'COMMITTED' then
    v_checkpoint_head_state_id :=
      nullif(trim(
        p_checkpoint->'runtime_checkpoint'->'shadow_ledger_manifest'->>'head_state_id'
      ), '');

    select t.value->>'after_state_id'
      into v_reconciled_after_state_id
    from jsonb_array_elements(
      coalesce(
        p_checkpoint->'runtime_checkpoint'
          ->'shadow_ledger_manifest'->'transitions',
        '[]'::jsonb
      )
    ) with ordinality t(value, ord)
    where t.value->>'cycle_id' = p_recovery_cycle_id
      and t.value->>'kind' = 'RECONCILED_COMMIT'
    order by t.ord desc
    limit 1;

    if v_checkpoint_head_state_id is null
       or length(v_checkpoint_head_state_id) <> 64
       or v_reconciled_after_state_id is null
       or length(v_reconciled_after_state_id) <> 64
       or v_checkpoint_head_state_id <> v_reconciled_after_state_id then
      raise exception 'PHASE84_COMMIT: COMMITTED recovery checkpoint must end at recovery RECONCILED_COMMIT state';
    end if;
  end if;

  select public.brian_commit_shadow_runtime_checkpoint(
    p_runtime_id,
    p_owner_token,
    p_fencing_token,
    p_expected_version,
    p_checkpoint
  ) into v_commit;

  v_commit_status := nullif(trim(v_commit->>'status'), '');
  v_commit_version := nullif(v_commit->>'version','')::bigint;
  v_current_version := coalesce(
    nullif(v_commit->>'current_version','')::bigint,
    v_commit_version
  );

  if v_commit_status not in ('COMMITTED','DUPLICATE_CURRENT') then
    insert into public.brian_shadow_recovery_commit_events(
      runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
      cancel_risk_receipt_id, worker_token,
      recovery_claim_fencing_token, runtime_fencing_token,
      expected_runtime_version, committed_runtime_version,
      checkpoint_id, recovery_journal_stage,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
      p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      p_expected_version, v_current_version,
      v_checkpoint_id, v_journal_stage,
      'CHECKPOINT_REJECTED', v_now,
      jsonb_build_object('phase70_status', v_commit_status)
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'terminal', false,
      'status', coalesce(v_commit_status,'CHECKPOINT_REJECTED'),
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', coalesce(v_commit_version,v_runtime_version),
      'current_version', coalesce(v_current_version,v_runtime_version),
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
    );
  end if;

  v_terminal := v_journal_stage = 'COMMITTED';

  update public.brian_shadow_recovery_claims
    set recovery_cycle_id=p_recovery_cycle_id,
        progress_runtime_version=v_commit_version,
        progress_head_state_id=coalesce(
          nullif(trim(
            p_checkpoint->'runtime_checkpoint'
              ->'shadow_ledger_manifest'->>'head_state_id'
          ), ''),
          v_current_head_state_id
        ),
        progress_checkpoint_id=v_checkpoint_id,
        status=case when v_terminal then 'COMPLETED' else status end,
        claim_until=case when v_terminal then null else claim_until end,
        completed_at=case when v_terminal then v_now else completed_at end,
        completion_ref=case when v_terminal then v_checkpoint_id else completion_ref end,
        updated_at=v_now
    where runtime_id=p_runtime_id
      and dispatch_id=v_directive.dispatch_id
      and cancel_risk_receipt_id=v_directive.cancel_risk_receipt_id
      and status='CLAIMED'
      and worker_token=p_worker_token
      and claim_fencing_token=p_recovery_claim_fencing_token;

  if not found then
    raise exception 'PHASE84_COMMIT: recovery claim changed during atomic progress commit';
  end if;

  if v_terminal then
    v_event := 'RECOVERY_COMPLETED';
  elsif v_journal_stage = 'CYCLE_CREATED' then
    v_event := 'WRITE_AHEAD_COMMITTED';
  else
    v_event := 'PROGRESS_COMMITTED';
  end if;

  insert into public.brian_shadow_recovery_commit_events(
    runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
    cancel_risk_receipt_id, worker_token,
    recovery_claim_fencing_token, runtime_fencing_token,
    expected_runtime_version, committed_runtime_version,
    checkpoint_id, recovery_journal_stage,
    event, observed_at,
    metadata
  ) values (
    p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
    p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
    p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
    p_expected_version, v_commit_version,
    v_checkpoint_id, v_journal_stage,
    v_event, v_now,
    jsonb_build_object(
      'phase70_status', v_commit_status,
      'start_runtime_version', v_start.runtime_version_at_start,
      'start_head_state_id', v_start.head_state_id_at_start,
      'start_risk_version', v_start.risk_version_at_start,
      'start_risk_receipt_id', v_start.risk_receipt_id_at_start
    )
  );

  return jsonb_build_object(
    'committed', true,
    'duplicate', v_commit_status = 'DUPLICATE_CURRENT',
    'terminal', v_terminal,
    'status', case
      when v_terminal then 'RECOVERY_COMPLETED'
      else v_commit_status
    end,
    'runtime_id', p_runtime_id,
    'dispatch_id', v_directive.dispatch_id,
    'original_cycle_id', p_original_cycle_id,
    'recovery_cycle_id', p_recovery_cycle_id,
    'checkpoint_id', v_checkpoint_id,
    'journal_stage', v_journal_stage,
    'version', v_commit_version,
    'current_version', v_current_version,
    'fencing_token', p_fencing_token,
    'recovery_claim_fencing_token', p_recovery_claim_fencing_token,
    'head_state_id', case
      when v_terminal then v_checkpoint_head_state_id
      else v_current_head_state_id
    end
  );
end;
$$;

revoke all on function public.brian_commit_shadow_recovery_checkpoint(
  text,text,bigint,text,text,text,bigint,bigint,jsonb
) from public, anon, authenticated;

grant execute on function public.brian_commit_shadow_recovery_checkpoint(
  text,text,bigint,text,text,text,bigint,bigint,jsonb
) to service_role;

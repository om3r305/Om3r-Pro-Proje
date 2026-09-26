-- Brian Phase 85 DRAFT SQL: authoritative recovery completion audit.
--
-- IMPORTANT: draft/undeployed SQL. At rollout freeze create the official
-- Supabase migration with `supabase migration new`, then rerun the complete
-- Postgres suite before deployment.
--
-- Phase84 deliberately stops at RECOVERY_COMMITTED_PENDING_AUDIT. Phase85 is
-- the only boundary allowed to convert that durable recovery into a COMPLETED
-- recovery claim. It independently reconstructs the pre-recovery paper
-- quantity from the final Phase63 paper checkpoint and the recovery cycle's
-- own fills, then proves that the recovery reduced real paper exposure without
-- flipping direction. It also cross-checks final Phase60 authoritative state.

create table if not exists public.brian_shadow_recovery_completion_certificates (
  runtime_id text not null,
  dispatch_id text not null,
  original_cycle_id text not null,
  recovery_cycle_id text not null,
  cancel_risk_receipt_id text not null,
  completion_runtime_version bigint not null check (completion_runtime_version > 0),
  completion_checkpoint_id text not null,
  start_head_state_id text not null,
  final_head_state_id text not null,
  paper_checkpoint_id text not null,
  recovery_claim_fencing_token bigint not null
    check (recovery_claim_fencing_token > 0),
  leg_audits jsonb not null check (jsonb_typeof(leg_audits) = 'array'),
  recovery_fill_count integer not null check (recovery_fill_count > 0),
  certified_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, dispatch_id, cancel_risk_receipt_id),
  unique (runtime_id, recovery_cycle_id),
  check (length(cancel_risk_receipt_id) = 64),
  check (length(completion_checkpoint_id) = 64),
  check (length(start_head_state_id) = 64),
  check (length(final_head_state_id) = 64),
  check (length(paper_checkpoint_id) = 64)
);

create index if not exists brian_shadow_recovery_completion_cycle_idx
  on public.brian_shadow_recovery_completion_certificates(
    runtime_id, original_cycle_id, certified_at desc
  );

alter table public.brian_shadow_recovery_completion_certificates enable row level security;
revoke all on public.brian_shadow_recovery_completion_certificates
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_recovery_completion_certificates_append_only
  on public.brian_shadow_recovery_completion_certificates;
create trigger brian_shadow_recovery_completion_certificates_append_only
  before update or delete on public.brian_shadow_recovery_completion_certificates
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_shadow_recovery_completion_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  dispatch_id text,
  original_cycle_id text not null,
  recovery_cycle_id text not null,
  cancel_risk_receipt_id text,
  completion_checkpoint_id text,
  event text not null check (
    event in (
      'CERTIFIED',
      'DUPLICATE',
      'LEASE_LOST',
      'HEAD_MOVED',
      'CLAIM_STATE_INVALID',
      'EVIDENCE_INVALID',
      'AUDIT_FAILED'
    )
  ),
  observed_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_shadow_recovery_completion_events_runtime_idx
  on public.brian_shadow_recovery_completion_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_shadow_recovery_completion_events enable row level security;
revoke all on public.brian_shadow_recovery_completion_events
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_recovery_completion_events_append_only
  on public.brian_shadow_recovery_completion_events;
create trigger brian_shadow_recovery_completion_events_append_only
  before update or delete on public.brian_shadow_recovery_completion_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_certify_shadow_recovery_completion(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_original_cycle_id text,
  p_recovery_cycle_id text,
  p_expected_checkpoint_id text
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
  v_checkpoint_id text;
  v_checkpoint jsonb;
  v_lease_until timestamptz;
  v_directive public.brian_shadow_cancel_recovery_directives%rowtype;
  v_claim public.brian_shadow_recovery_claims%rowtype;
  v_start public.brian_shadow_recovery_starts%rowtype;
  v_existing public.brian_shadow_recovery_completion_certificates%rowtype;
  v_ledger jsonb;
  v_final_state jsonb;
  v_paper jsonb;
  v_paper_checkpoint_id text;
  v_journal_stage text;
  v_phase84_terminal_exists boolean := false;
  v_leg jsonb;
  v_asset text;
  v_current_direction integer;
  v_current_weight double precision;
  v_target_weight double precision;
  v_reduce_weight double precision;
  v_final_weight double precision;
  v_final_qty double precision;
  v_recovery_delta_qty double precision;
  v_pre_qty double precision;
  v_fill_count integer;
  v_total_fill_count integer := 0;
  v_required_drop double precision;
  v_leg_audits jsonb := '[]'::jsonb;
  v_failures jsonb := '[]'::jsonb;
  v_epsilon double precision := 1e-12;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_original_cycle_id is null or length(p_original_cycle_id) <> 64
     or p_recovery_cycle_id is null or length(p_recovery_cycle_id) <> 64
     or p_original_cycle_id = p_recovery_cycle_id
     or p_expected_checkpoint_id is null or length(p_expected_checkpoint_id) <> 64 then
    raise exception 'PHASE85_AUDIT: valid runtime/lease/original/recovery/checkpoint anchors required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, head_state_id,
         checkpoint_id, checkpoint_payload, lease_until
    into v_owner, v_runtime_fence, v_runtime_version, v_head_state_id,
         v_checkpoint_id, v_checkpoint, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if not found
     or v_owner <> p_owner_token
     or v_runtime_fence <> p_fencing_token
     or v_lease_until <= v_now then
    insert into public.brian_shadow_recovery_completion_events(
      runtime_id, original_cycle_id, recovery_cycle_id,
      completion_checkpoint_id, event, observed_at
    ) values (
      p_runtime_id, p_original_cycle_id, p_recovery_cycle_id,
      p_expected_checkpoint_id, 'LEASE_LOST', v_now
    );
    return jsonb_build_object(
      'certified', false,
      'duplicate', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'runtime_version', coalesce(v_runtime_version,0),
      'completion_checkpoint_id', coalesce(v_checkpoint_id,p_expected_checkpoint_id),
      'fencing_token', coalesce(v_runtime_fence,p_fencing_token)
    );
  end if;

  select *
    into v_directive
  from public.brian_shadow_cancel_recovery_directives
  where runtime_id=p_runtime_id
    and cycle_id=p_original_cycle_id
  order by prepared_at asc, cancel_risk_version asc
  limit 1;

  if not found then
    raise exception 'PHASE85_AUDIT: recovery directive missing';
  end if;

  select *
    into v_existing
  from public.brian_shadow_recovery_completion_certificates
  where runtime_id=p_runtime_id
    and dispatch_id=v_directive.dispatch_id
    and cancel_risk_receipt_id=v_directive.cancel_risk_receipt_id;

  if found then
    if v_existing.recovery_cycle_id <> p_recovery_cycle_id
       or v_existing.completion_checkpoint_id <> p_expected_checkpoint_id then
      raise exception 'PHASE85_AUDIT_CONFLICT: existing certificate has different recovery lineage';
    end if;

    insert into public.brian_shadow_recovery_completion_events(
      runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
      cancel_risk_receipt_id, completion_checkpoint_id,
      event, observed_at
    ) values (
      p_runtime_id, v_existing.dispatch_id, p_original_cycle_id,
      p_recovery_cycle_id, v_existing.cancel_risk_receipt_id,
      v_existing.completion_checkpoint_id,
      'DUPLICATE', v_now
    );

    return jsonb_build_object(
      'certified', true,
      'duplicate', true,
      'status', 'DUPLICATE',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_existing.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'cancel_risk_receipt_id', v_existing.cancel_risk_receipt_id,
      'runtime_version', v_existing.completion_runtime_version,
      'completion_checkpoint_id', v_existing.completion_checkpoint_id,
      'start_head_state_id', v_existing.start_head_state_id,
      'final_head_state_id', v_existing.final_head_state_id,
      'paper_checkpoint_id', v_existing.paper_checkpoint_id,
      'recovery_claim_fencing_token', v_existing.recovery_claim_fencing_token,
      'recovery_fill_count', v_existing.recovery_fill_count,
      'leg_audits', v_existing.leg_audits,
      'fencing_token', p_fencing_token
    );
  end if;

  select *
    into v_claim
  from public.brian_shadow_recovery_claims
  where runtime_id=p_runtime_id
    and dispatch_id=v_directive.dispatch_id
    and cancel_risk_receipt_id=v_directive.cancel_risk_receipt_id
  for update;

  if not found
     or v_claim.status <> 'CLAIMED'
     or v_claim.recovery_cycle_id is distinct from p_recovery_cycle_id
     or v_claim.progress_runtime_version is distinct from v_runtime_version
     or v_claim.progress_head_state_id is distinct from v_head_state_id
     or v_claim.progress_checkpoint_id is distinct from v_checkpoint_id then
    insert into public.brian_shadow_recovery_completion_events(
      runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
      cancel_risk_receipt_id, completion_checkpoint_id,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
      p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
      p_expected_checkpoint_id,
      'CLAIM_STATE_INVALID', v_now,
      jsonb_build_object(
        'claim_status', v_claim.status,
        'claim_recovery_cycle_id', v_claim.recovery_cycle_id,
        'claim_progress_runtime_version', v_claim.progress_runtime_version,
        'claim_progress_head_state_id', v_claim.progress_head_state_id,
        'claim_progress_checkpoint_id', v_claim.progress_checkpoint_id,
        'current_runtime_version', v_runtime_version,
        'current_head_state_id', v_head_state_id,
        'current_checkpoint_id', v_checkpoint_id
      )
    );
    return jsonb_build_object(
      'certified', false,
      'duplicate', false,
      'status', 'CLAIM_STATE_INVALID',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'runtime_version', v_runtime_version,
      'completion_checkpoint_id', v_checkpoint_id,
      'fencing_token', p_fencing_token
    );
  end if;

  if v_checkpoint_id <> p_expected_checkpoint_id then
    insert into public.brian_shadow_recovery_completion_events(
      runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
      cancel_risk_receipt_id, completion_checkpoint_id,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
      p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
      p_expected_checkpoint_id,
      'HEAD_MOVED', v_now,
      jsonb_build_object('current_checkpoint_id', v_checkpoint_id)
    );
    return jsonb_build_object(
      'certified', false,
      'duplicate', false,
      'status', 'HEAD_MOVED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'runtime_version', v_runtime_version,
      'completion_checkpoint_id', v_checkpoint_id,
      'fencing_token', p_fencing_token
    );
  end if;

  select *
    into v_start
  from public.brian_shadow_recovery_starts
  where runtime_id=p_runtime_id
    and dispatch_id=v_directive.dispatch_id
    and cancel_risk_receipt_id=v_directive.cancel_risk_receipt_id;

  if not found then
    raise exception 'PHASE85_AUDIT: recovery STARTED evidence missing';
  end if;

  select e.value->>'stage'
    into v_journal_stage
  from jsonb_array_elements(
    coalesce(v_checkpoint->'journal_manifest'->'entries','[]'::jsonb)
  ) with ordinality e(value, ord)
  where e.value->>'cycle_id'=p_recovery_cycle_id
  order by e.ord desc
  limit 1;

  select exists(
    select 1
    from public.brian_shadow_recovery_commit_events e
    where e.runtime_id=p_runtime_id
      and e.dispatch_id=v_directive.dispatch_id
      and e.original_cycle_id=p_original_cycle_id
      and e.recovery_cycle_id=p_recovery_cycle_id
      and e.checkpoint_id=p_expected_checkpoint_id
      and e.event='RECOVERY_COMMITTED_PENDING_AUDIT'
  ) into v_phase84_terminal_exists;

  if v_journal_stage is distinct from 'COMMITTED'
     or not v_phase84_terminal_exists then
    insert into public.brian_shadow_recovery_completion_events(
      runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
      cancel_risk_receipt_id, completion_checkpoint_id,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
      p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
      p_expected_checkpoint_id,
      'EVIDENCE_INVALID', v_now,
      jsonb_build_object(
        'journal_stage', v_journal_stage,
        'phase84_terminal_event', v_phase84_terminal_exists
      )
    );
    return jsonb_build_object(
      'certified', false,
      'duplicate', false,
      'status', 'EVIDENCE_INVALID',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'runtime_version', v_runtime_version,
      'completion_checkpoint_id', v_checkpoint_id,
      'fencing_token', p_fencing_token
    );
  end if;

  v_ledger := v_checkpoint->'runtime_checkpoint'->'shadow_ledger_manifest';
  v_paper := v_checkpoint->'runtime_checkpoint'->'paper';
  if jsonb_typeof(v_ledger) <> 'object'
     or jsonb_typeof(v_paper) <> 'object' then
    raise exception 'PHASE85_AUDIT: final checkpoint missing ledger/paper evidence';
  end if;

  v_final_state := v_ledger->'states'->v_head_state_id;
  v_paper_checkpoint_id := nullif(trim(v_paper->>'checkpoint_id'),'');
  if jsonb_typeof(v_final_state) <> 'object'
     or v_paper_checkpoint_id is null
     or length(v_paper_checkpoint_id) <> 64
     or coalesce((v_paper->>'paper_only')::boolean,false) is not true
     or coalesce((v_paper->>'live_execution')::boolean,true) is not false
     or jsonb_typeof(v_paper->'fill_sequence') <> 'array'
     or jsonb_typeof(v_paper->'final_positions') <> 'array' then
    raise exception 'PHASE85_AUDIT: malformed final paper/ledger evidence';
  end if;

  for v_leg in
    select value from jsonb_array_elements(v_start.recovery_legs)
  loop
    v_asset := nullif(trim(v_leg->>'asset_id'),'');
    v_current_direction := nullif(v_leg->>'current_direction','')::integer;
    v_current_weight := nullif(v_leg->>'current_weight','')::double precision;
    v_target_weight := nullif(v_leg->>'target_weight','')::double precision;
    v_reduce_weight := nullif(v_leg->>'reduce_weight','')::double precision;

    if v_asset is null
       or v_current_direction not in (-1,1)
       or v_current_weight is null
       or v_target_weight is null
       or v_reduce_weight is null
       or v_reduce_weight <= 0 then
      raise exception 'PHASE85_AUDIT: malformed STARTED recovery leg';
    end if;

    select coalesce((p.value->>'quantity')::double precision,0.0)
      into v_final_qty
    from jsonb_array_elements(v_paper->'final_positions') p(value)
    where p.value->>'asset_id'=v_asset
    limit 1;
    v_final_qty := coalesce(v_final_qty,0.0);

    select
      count(*),
      coalesce(sum(
        case
          when f.value->>'side'='BUY'
            then (f.value->>'quantity_base')::double precision
          when f.value->>'side'='SELL'
            then -(f.value->>'quantity_base')::double precision
          else 0.0
        end
      ),0.0)
      into v_fill_count, v_recovery_delta_qty
    from jsonb_array_elements(v_paper->'fill_sequence') f(value)
    where f.value->>'cycle_id'=p_recovery_cycle_id
      and f.value->>'asset_id'=v_asset;

    v_total_fill_count := v_total_fill_count + v_fill_count;
    v_pre_qty := v_final_qty - v_recovery_delta_qty;

    select coalesce((w.value->>1)::double precision,0.0)
      into v_final_weight
    from jsonb_array_elements(
      coalesce(v_final_state->'position_weights','[]'::jsonb)
    ) w(value)
    where w.value->>0=v_asset
    limit 1;
    v_final_weight := coalesce(v_final_weight,0.0);

    v_required_drop := least(v_epsilon, v_reduce_weight / 2.0);

    if v_fill_count <= 0 then
      v_failures := v_failures || jsonb_build_array(jsonb_build_object(
        'asset_id',v_asset,'reason','RECOVERY_FILL_MISSING'
      ));
    elsif (case when v_pre_qty > v_epsilon then 1
                when v_pre_qty < -v_epsilon then -1 else 0 end)
          <> v_current_direction then
      v_failures := v_failures || jsonb_build_array(jsonb_build_object(
        'asset_id',v_asset,'reason','PRE_RECOVERY_PAPER_DIRECTION_MISMATCH',
        'pre_quantity',v_pre_qty
      ));
    elsif (v_current_direction=1 and v_recovery_delta_qty >= -v_epsilon)
       or (v_current_direction=-1 and v_recovery_delta_qty <= v_epsilon) then
      v_failures := v_failures || jsonb_build_array(jsonb_build_object(
        'asset_id',v_asset,'reason','RECOVERY_FILL_DIRECTION_INVALID',
        'recovery_delta_quantity',v_recovery_delta_qty
      ));
    elsif abs(v_recovery_delta_qty) > abs(v_pre_qty) + v_epsilon then
      v_failures := v_failures || jsonb_build_array(jsonb_build_object(
        'asset_id',v_asset,'reason','RECOVERY_OVERREDUCED_AND_FLIPPED_QUANTITY',
        'pre_quantity',v_pre_qty,
        'recovery_delta_quantity',v_recovery_delta_qty,
        'final_quantity',v_final_qty
      ));
    elsif abs(v_final_qty) > v_epsilon
       and (case when v_final_qty > 0 then 1 else -1 end)
           <> v_current_direction then
      v_failures := v_failures || jsonb_build_array(jsonb_build_object(
        'asset_id',v_asset,'reason','FINAL_PAPER_DIRECTION_FLIPPED',
        'final_quantity',v_final_qty
      ));
    elsif abs(v_final_qty) > abs(v_pre_qty) - v_required_drop then
      v_failures := v_failures || jsonb_build_array(jsonb_build_object(
        'asset_id',v_asset,'reason','FINAL_PAPER_EXPOSURE_NOT_REDUCED',
        'pre_quantity',v_pre_qty,
        'final_quantity',v_final_qty
      ));
    elsif abs(v_final_weight) > v_epsilon
       and (case when v_final_weight > 0 then 1 else -1 end)
           <> v_current_direction then
      v_failures := v_failures || jsonb_build_array(jsonb_build_object(
        'asset_id',v_asset,'reason','FINAL_AUTHORITATIVE_WEIGHT_FLIPPED',
        'final_weight',v_final_weight
      ));
    end if;

    v_leg_audits := v_leg_audits || jsonb_build_array(jsonb_build_object(
      'asset_id',v_asset,
      'current_direction',v_current_direction,
      'current_weight',v_current_weight,
      'target_weight',v_target_weight,
      'reduce_weight',v_reduce_weight,
      'pre_recovery_quantity',v_pre_qty,
      'recovery_delta_quantity',v_recovery_delta_qty,
      'final_quantity',v_final_qty,
      'final_authoritative_weight',v_final_weight,
      'fill_count',v_fill_count,
      'direction_preserved',
        abs(v_final_qty) <= v_epsilon
        or (case when v_final_qty > 0 then 1 else -1 end)=v_current_direction,
      'quantity_exposure_reduced',
        abs(v_final_qty) <= abs(v_pre_qty) - v_required_drop,
      'target_distance_before',abs(v_current_weight-v_target_weight),
      'target_distance_after',abs(v_final_weight-v_target_weight),
      'target_weight_reached_within_1e_6',
        abs(v_final_weight-v_target_weight) <= 1e-6
    ));
  end loop;

  if jsonb_array_length(v_leg_audits)=0
     or v_total_fill_count <= 0
     or jsonb_array_length(v_failures)>0 then
    insert into public.brian_shadow_recovery_completion_events(
      runtime_id, dispatch_id, original_cycle_id, recovery_cycle_id,
      cancel_risk_receipt_id, completion_checkpoint_id,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_original_cycle_id,
      p_recovery_cycle_id, v_directive.cancel_risk_receipt_id,
      p_expected_checkpoint_id,
      'AUDIT_FAILED', v_now,
      jsonb_build_object(
        'failures',v_failures,
        'leg_audits',v_leg_audits,
        'recovery_fill_count',v_total_fill_count
      )
    );
    return jsonb_build_object(
      'certified', false,
      'duplicate', false,
      'status', 'AUDIT_FAILED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'original_cycle_id', p_original_cycle_id,
      'recovery_cycle_id', p_recovery_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'completion_checkpoint_id', v_checkpoint_id,
      'final_head_state_id', v_head_state_id,
      'paper_checkpoint_id', v_paper_checkpoint_id,
      'recovery_claim_fencing_token', v_claim.claim_fencing_token,
      'recovery_fill_count', v_total_fill_count,
      'leg_audits', v_leg_audits,
      'failures', v_failures,
      'fencing_token', p_fencing_token
    );
  end if;

  insert into public.brian_shadow_recovery_completion_certificates(
    runtime_id,dispatch_id,original_cycle_id,recovery_cycle_id,
    cancel_risk_receipt_id,completion_runtime_version,
    completion_checkpoint_id,start_head_state_id,final_head_state_id,
    paper_checkpoint_id,recovery_claim_fencing_token,
    leg_audits,recovery_fill_count,certified_at
  ) values (
    p_runtime_id,v_directive.dispatch_id,p_original_cycle_id,p_recovery_cycle_id,
    v_directive.cancel_risk_receipt_id,v_runtime_version,
    v_checkpoint_id,v_start.head_state_id_at_start,v_head_state_id,
    v_paper_checkpoint_id,v_claim.claim_fencing_token,
    v_leg_audits,v_total_fill_count,v_now
  );

  update public.brian_shadow_recovery_claims
    set status='COMPLETED',
        claim_until=null,
        completed_at=v_now,
        completion_ref=v_checkpoint_id,
        updated_at=v_now
    where runtime_id=p_runtime_id
      and dispatch_id=v_directive.dispatch_id
      and cancel_risk_receipt_id=v_directive.cancel_risk_receipt_id
      and status='CLAIMED'
      and recovery_cycle_id=p_recovery_cycle_id
      and progress_checkpoint_id=v_checkpoint_id
      and progress_runtime_version=v_runtime_version
      and progress_head_state_id=v_head_state_id;

  if not found then
    raise exception 'PHASE85_AUDIT: claim changed during atomic certification';
  end if;

  insert into public.brian_shadow_recovery_completion_events(
    runtime_id,dispatch_id,original_cycle_id,recovery_cycle_id,
    cancel_risk_receipt_id,completion_checkpoint_id,
    event,observed_at,
    metadata
  ) values (
    p_runtime_id,v_directive.dispatch_id,p_original_cycle_id,p_recovery_cycle_id,
    v_directive.cancel_risk_receipt_id,v_checkpoint_id,
    'CERTIFIED',v_now,
    jsonb_build_object(
      'completion_runtime_version',v_runtime_version,
      'start_head_state_id',v_start.head_state_id_at_start,
      'final_head_state_id',v_head_state_id,
      'paper_checkpoint_id',v_paper_checkpoint_id,
      'recovery_claim_fencing_token',v_claim.claim_fencing_token,
      'recovery_fill_count',v_total_fill_count,
      'leg_audits',v_leg_audits
    )
  );

  return jsonb_build_object(
    'certified', true,
    'duplicate', false,
    'status', 'CERTIFIED',
    'runtime_id', p_runtime_id,
    'dispatch_id', v_directive.dispatch_id,
    'original_cycle_id', p_original_cycle_id,
    'recovery_cycle_id', p_recovery_cycle_id,
    'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
    'runtime_version', v_runtime_version,
    'completion_checkpoint_id', v_checkpoint_id,
    'start_head_state_id', v_start.head_state_id_at_start,
    'final_head_state_id', v_head_state_id,
    'paper_checkpoint_id', v_paper_checkpoint_id,
    'recovery_claim_fencing_token', v_claim.claim_fencing_token,
    'recovery_fill_count', v_total_fill_count,
    'leg_audits', v_leg_audits,
    'fencing_token', p_fencing_token
  );
end;
$$;

create or replace function public.brian_read_shadow_recovery_completion(
  p_runtime_id text,
  p_original_cycle_id text
) returns setof public.brian_shadow_recovery_completion_certificates
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  select *
  from public.brian_shadow_recovery_completion_certificates
  where runtime_id=p_runtime_id
    and original_cycle_id=p_original_cycle_id
  order by certified_at asc;
$$;

revoke all on function public.brian_certify_shadow_recovery_completion(
  text,text,bigint,text,text,text
) from public, anon, authenticated;
revoke all on function public.brian_read_shadow_recovery_completion(text,text)
  from public, anon, authenticated;

grant execute on function public.brian_certify_shadow_recovery_completion(
  text,text,bigint,text,text,text
) to service_role;
grant execute on function public.brian_read_shadow_recovery_completion(text,text)
  to service_role;

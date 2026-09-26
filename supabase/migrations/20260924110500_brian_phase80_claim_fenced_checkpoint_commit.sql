-- Brian Phase80 official migration: claim-fenced authoritative checkpoint commit.
-- Promoted from the reviewed Phase80 draft contract into official
-- migration lineage on 2026-09-24. This repository change does NOT deploy the
-- migration to any live Supabase project. Runtime remains shadow/paper-only.

create table if not exists public.brian_shadow_claim_commit_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  worker_token text not null,
  claim_fencing_token bigint not null,
  runtime_fencing_token bigint not null,
  expected_runtime_version bigint,
  committed_runtime_version bigint,
  checkpoint_id text,
  journal_stage text,
  event text not null check (
    event in (
      'COMMITTED',
      'DUPLICATE_CURRENT',
      'CLAIM_LOST',
      'LEASE_LOST',
      'RUNTIME_VERSION_CONFLICT',
      'START_MISSING',
      'CHECKPOINT_REJECTED'
    )
  ),
  observed_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_shadow_claim_commit_events_runtime_idx
  on public.brian_shadow_claim_commit_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_shadow_claim_commit_events enable row level security;
revoke all on public.brian_shadow_claim_commit_events
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_claim_commit_events_append_only
  on public.brian_shadow_claim_commit_events;
create trigger brian_shadow_claim_commit_events_append_only
  before update or delete on public.brian_shadow_claim_commit_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_commit_claimed_shadow_runtime_checkpoint(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_cycle_id text,
  p_worker_token text,
  p_claim_fencing_token bigint,
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
  v_lease_until timestamptz;
  v_dispatch_id text;
  v_claim public.brian_shadow_execution_claims%rowtype;
  v_start public.brian_shadow_execution_starts%rowtype;
  v_checkpoint_id text;
  v_journal_stage text;
  v_commit jsonb;
  v_commit_status text;
  v_commit_version bigint;
  v_current_version bigint;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_cycle_id is null or length(p_cycle_id) <> 64
     or nullif(trim(p_worker_token), '') is null
     or p_claim_fencing_token is null or p_claim_fencing_token <= 0
     or p_expected_version is null or p_expected_version <= 0
     or p_checkpoint is null or jsonb_typeof(p_checkpoint) <> 'object' then
    raise exception 'PHASE80_COMMIT: valid runtime/lease/cycle/worker/claim/version/checkpoint required';
  end if;

  if coalesce((p_checkpoint->>'live_execution')::boolean, false) then
    raise exception 'PHASE80_COMMIT: live checkpoint rejected';
  end if;

  v_checkpoint_id := nullif(trim(p_checkpoint->>'checkpoint_id'), '');
  if v_checkpoint_id is null or length(v_checkpoint_id) <> 64 then
    raise exception 'PHASE80_COMMIT: checkpoint_id must be a content hash';
  end if;

  select e.value->>'stage'
    into v_journal_stage
  from jsonb_array_elements(
    coalesce(p_checkpoint->'journal_manifest'->'entries', '[]'::jsonb)
  ) with ordinality e(value, ord)
  where e.value->>'cycle_id' = p_cycle_id
  order by e.ord desc
  limit 1;

  if v_journal_stage is null
     or v_journal_stage not in (
       'PAPER_APPLIED',
       'LOCAL_PROJECTED',
       'RECONCILIATION_REQUIRED',
       'RECONCILED',
       'COMMITTED',
       'ABORTED'
     ) then
    raise exception 'PHASE80_COMMIT: checkpoint must prove post-STARTED cycle progress, got %',
      coalesce(v_journal_stage, '<missing>');
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, checkpoint_id, checkpoint_payload, lease_until
    into v_owner, v_runtime_fence, v_runtime_version,
         v_current_checkpoint_id, v_current_checkpoint_payload, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  select dispatch_id
    into v_dispatch_id
  from public.brian_shadow_execution_dispatches
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id;

  if v_dispatch_id is null then
    raise exception 'PHASE80_COMMIT: dispatch missing for %', p_cycle_id;
  end if;

  if v_owner is null
     or v_owner <> p_owner_token
     or v_runtime_fence <> p_fencing_token
     or v_lease_until <= v_now then
    insert into public.brian_shadow_claim_commit_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token,
      expected_runtime_version, checkpoint_id, journal_stage,
      event, observed_at
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token,
      p_expected_version, v_checkpoint_id, v_journal_stage,
      'LEASE_LOST', v_now
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', coalesce(v_runtime_version,0),
      'current_version', coalesce(v_runtime_version,0),
      'fencing_token', coalesce(v_runtime_fence,p_fencing_token),
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
    insert into public.brian_shadow_claim_commit_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token,
      expected_runtime_version, committed_runtime_version,
      checkpoint_id, journal_stage, event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token,
      p_expected_version, v_runtime_version,
      v_checkpoint_id, v_journal_stage,
      'CLAIM_LOST', v_now,
      jsonb_build_object(
        'current_claim_status', case when v_claim.dispatch_id is null then null else v_claim.status end,
        'current_worker_token', v_claim.worker_token,
        'current_claim_fencing_token', v_claim.claim_fencing_token,
        'current_claim_until', v_claim.claim_until
      )
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'status', 'CLAIM_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', v_runtime_version,
      'current_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  select *
    into v_start
  from public.brian_shadow_execution_starts
  where runtime_id = p_runtime_id
    and dispatch_id = v_dispatch_id;

  if not found then
    insert into public.brian_shadow_claim_commit_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token,
      expected_runtime_version, committed_runtime_version,
      checkpoint_id, journal_stage, event, observed_at
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token,
      p_expected_version, v_runtime_version,
      v_checkpoint_id, v_journal_stage,
      'START_MISSING', v_now
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'status', 'START_MISSING',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', v_runtime_version,
      'current_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  -- Lost-response retry is accepted only when the current authoritative head is
  -- exactly the submitted checkpoint AND the same claim still owns execution.
  if v_runtime_version <> p_expected_version then
    if v_current_checkpoint_id = v_checkpoint_id then
      if v_current_checkpoint_payload is distinct from p_checkpoint then
        raise exception 'PHASE80_CHECKPOINT_ID_CONFLICT: current checkpoint id reused with different payload';
      end if;

      insert into public.brian_shadow_claim_commit_events(
        runtime_id, dispatch_id, cycle_id, worker_token,
        claim_fencing_token, runtime_fencing_token,
        expected_runtime_version, committed_runtime_version,
        checkpoint_id, journal_stage, event, observed_at,
        metadata
      ) values (
        p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
        p_claim_fencing_token, p_fencing_token,
        p_expected_version, v_runtime_version,
        v_checkpoint_id, v_journal_stage,
        'DUPLICATE_CURRENT', v_now,
        jsonb_build_object('lost_response_retry', true)
      );

      return jsonb_build_object(
        'committed', true,
        'duplicate', true,
        'status', 'DUPLICATE_CURRENT',
        'runtime_id', p_runtime_id,
        'dispatch_id', v_dispatch_id,
        'cycle_id', p_cycle_id,
        'checkpoint_id', v_checkpoint_id,
        'journal_stage', v_journal_stage,
        'version', v_runtime_version,
        'current_version', v_runtime_version,
        'fencing_token', p_fencing_token,
        'claim_fencing_token', p_claim_fencing_token
      );
    end if;

    insert into public.brian_shadow_claim_commit_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token,
      expected_runtime_version, committed_runtime_version,
      checkpoint_id, journal_stage, event, observed_at
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token,
      p_expected_version, v_runtime_version,
      v_checkpoint_id, v_journal_stage,
      'RUNTIME_VERSION_CONFLICT', v_now
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'status', 'RUNTIME_VERSION_CONFLICT',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', v_runtime_version,
      'current_version', v_runtime_version,
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
    );
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
    insert into public.brian_shadow_claim_commit_events(
      runtime_id, dispatch_id, cycle_id, worker_token,
      claim_fencing_token, runtime_fencing_token,
      expected_runtime_version, committed_runtime_version,
      checkpoint_id, journal_stage, event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
      p_claim_fencing_token, p_fencing_token,
      p_expected_version, v_current_version,
      v_checkpoint_id, v_journal_stage,
      'CHECKPOINT_REJECTED', v_now,
      jsonb_build_object('phase70_status', v_commit_status)
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'status', coalesce(v_commit_status,'CHECKPOINT_REJECTED'),
      'runtime_id', p_runtime_id,
      'dispatch_id', v_dispatch_id,
      'cycle_id', p_cycle_id,
      'checkpoint_id', v_checkpoint_id,
      'journal_stage', v_journal_stage,
      'version', coalesce(v_commit_version,v_runtime_version),
      'current_version', coalesce(v_current_version,v_runtime_version),
      'fencing_token', p_fencing_token,
      'claim_fencing_token', p_claim_fencing_token
    );
  end if;

  insert into public.brian_shadow_claim_commit_events(
    runtime_id, dispatch_id, cycle_id, worker_token,
    claim_fencing_token, runtime_fencing_token,
    expected_runtime_version, committed_runtime_version,
    checkpoint_id, journal_stage, event, observed_at,
    metadata
  ) values (
    p_runtime_id, v_dispatch_id, p_cycle_id, p_worker_token,
    p_claim_fencing_token, p_fencing_token,
    p_expected_version, v_commit_version,
    v_checkpoint_id, v_journal_stage,
    v_commit_status, v_now,
    jsonb_build_object(
      'phase70_status', v_commit_status,
      'start_risk_version', v_start.risk_version_at_start,
      'start_risk_receipt_id', v_start.risk_receipt_id_at_start
    )
  );

  return jsonb_build_object(
    'committed', true,
    'duplicate', v_commit_status = 'DUPLICATE_CURRENT',
    'status', v_commit_status,
    'runtime_id', p_runtime_id,
    'dispatch_id', v_dispatch_id,
    'cycle_id', p_cycle_id,
    'checkpoint_id', v_checkpoint_id,
    'journal_stage', v_journal_stage,
    'version', v_commit_version,
    'current_version', v_current_version,
    'fencing_token', p_fencing_token,
    'claim_fencing_token', p_claim_fencing_token
  );
end;
$$;

revoke all on function public.brian_commit_claimed_shadow_runtime_checkpoint(
  text,text,bigint,text,text,bigint,bigint,jsonb
) from public, anon, authenticated;

grant execute on function public.brian_commit_claimed_shadow_runtime_checkpoint(
  text,text,bigint,text,text,bigint,bigint,jsonb
) to service_role;

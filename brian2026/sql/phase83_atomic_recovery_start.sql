-- Brian Phase 83 DRAFT SQL: atomic recovery STARTED boundary.
--
-- IMPORTANT: draft/undeployed SQL. At rollout freeze create the official
-- Supabase migration with `supabase migration new`, then rerun the complete
-- Postgres suite before deployment.
--
-- Phase81 defines the immutable reduce-only recovery obligation.
-- Phase82 grants one recovery worker at a time an independent fencing token.
-- Phase83 is the final point-of-no-return before any recovery paper side effect:
-- runtime lease/head, current recovery claim, current risk state and recovery
-- leg shape are revalidated under the same runtime advisory lock, then an
-- immutable RECOVERY_STARTED record is inserted.
--
-- This phase still performs no paper or exchange side effect.

create table if not exists public.brian_shadow_recovery_starts (
  runtime_id text not null,
  dispatch_id text not null,
  cycle_id text not null,
  cancel_risk_receipt_id text not null,
  worker_token text not null,
  recovery_claim_fencing_token bigint not null
    check (recovery_claim_fencing_token > 0),
  runtime_fencing_token bigint not null check (runtime_fencing_token > 0),
  runtime_version_at_start bigint not null check (runtime_version_at_start > 0),
  head_state_id_at_start text not null,
  risk_version_at_start bigint not null check (risk_version_at_start > 0),
  risk_receipt_id_at_start text not null,
  risk_state_at_start text not null
    check (risk_state_at_start in ('ACTIVE','REDUCING')),
  directive_source_runtime_version bigint not null
    check (directive_source_runtime_version > 0),
  directive_current_state_id text not null,
  recovery_legs jsonb not null check (jsonb_typeof(recovery_legs) = 'array'),
  started_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, dispatch_id, cancel_risk_receipt_id),
  unique (runtime_id, cycle_id, cancel_risk_receipt_id),
  check (length(cancel_risk_receipt_id) = 64),
  check (length(head_state_id_at_start) = 64),
  check (length(risk_receipt_id_at_start) = 64),
  check (length(directive_current_state_id) = 64),
  check (jsonb_array_length(recovery_legs) > 0)
);

create index if not exists brian_shadow_recovery_starts_cycle_idx
  on public.brian_shadow_recovery_starts(
    runtime_id, cycle_id, started_at desc
  );

alter table public.brian_shadow_recovery_starts enable row level security;
revoke all on public.brian_shadow_recovery_starts
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_recovery_starts_append_only
  on public.brian_shadow_recovery_starts;
create trigger brian_shadow_recovery_starts_append_only
  before update or delete on public.brian_shadow_recovery_starts
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_shadow_recovery_start_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  dispatch_id text,
  cycle_id text not null,
  cancel_risk_receipt_id text,
  worker_token text,
  recovery_claim_fencing_token bigint,
  runtime_fencing_token bigint,
  runtime_version bigint,
  head_state_id text,
  risk_version bigint,
  risk_receipt_id text,
  risk_state text,
  event text not null check (
    event in (
      'STARTED',
      'STARTED_ALREADY',
      'STARTED_RESUME',
      'WAIT_RISK_RELEASE',
      'CLAIM_LOST',
      'HEAD_MOVED',
      'DIRECTIVE_MISSING',
      'EVIDENCE_INVALID',
      'LEASE_LOST',
      'RISK_STATE_UNAVAILABLE'
    )
  ),
  observed_at timestamptz not null default now(),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_shadow_recovery_start_events_runtime_idx
  on public.brian_shadow_recovery_start_events(
    runtime_id, observed_at desc, event_sequence desc
  );

alter table public.brian_shadow_recovery_start_events enable row level security;
revoke all on public.brian_shadow_recovery_start_events
  from anon, authenticated, service_role;

drop trigger if exists brian_shadow_recovery_start_events_append_only
  on public.brian_shadow_recovery_start_events;
create trigger brian_shadow_recovery_start_events_append_only
  before update or delete on public.brian_shadow_recovery_start_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_mark_shadow_recovery_started(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_cycle_id text,
  p_worker_token text,
  p_recovery_claim_fencing_token bigint
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
  v_existing public.brian_shadow_recovery_starts%rowtype;
  v_risk_version bigint;
  v_risk_head_entry_id text;
  v_risk_receipt jsonb;
  v_risk_receipt_id text;
  v_risk_state text;
  v_leg jsonb;
  v_leg_count integer;
  v_resume boolean := false;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null or p_fencing_token <= 0
     or p_cycle_id is null or length(p_cycle_id) <> 64
     or nullif(trim(p_worker_token), '') is null
     or p_recovery_claim_fencing_token is null
     or p_recovery_claim_fencing_token <= 0 then
    raise exception 'PHASE83_START: valid runtime/lease/cycle/worker/recovery-claim fence required';
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
    insert into public.brian_shadow_recovery_start_events(
      runtime_id, cycle_id, worker_token,
      recovery_claim_fencing_token, runtime_fencing_token,
      runtime_version, head_state_id,
      event, observed_at
    ) values (
      p_runtime_id, p_cycle_id, p_worker_token,
      p_recovery_claim_fencing_token, p_fencing_token,
      v_runtime_version, v_head_state_id,
      'LEASE_LOST', v_now
    );
    return jsonb_build_object(
      'started', false,
      'duplicate', false,
      'resume_only', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', coalesce(v_runtime_version,0),
      'head_state_id', v_head_state_id,
      'fencing_token', coalesce(v_runtime_fence,p_fencing_token),
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
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
    insert into public.brian_shadow_recovery_start_events(
      runtime_id, cycle_id, worker_token,
      recovery_claim_fencing_token, runtime_fencing_token,
      runtime_version, head_state_id,
      event, observed_at
    ) values (
      p_runtime_id, p_cycle_id, p_worker_token,
      p_recovery_claim_fencing_token, p_fencing_token,
      v_runtime_version, v_head_state_id,
      'DIRECTIVE_MISSING', v_now
    );
    return jsonb_build_object(
      'started', false,
      'duplicate', false,
      'resume_only', false,
      'status', 'DIRECTIVE_MISSING',
      'runtime_id', p_runtime_id,
      'cycle_id', p_cycle_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
    );
  end if;

  if v_directive.recovery_status not in ('READY_REDUCE_ONLY','WAIT_RISK_RELEASE')
     or jsonb_typeof(v_directive.recovery_legs) <> 'array'
     or jsonb_array_length(v_directive.recovery_legs) = 0 then
    insert into public.brian_shadow_recovery_start_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, recovery_claim_fencing_token, runtime_fencing_token,
      runtime_version, head_state_id,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      v_runtime_version, v_head_state_id,
      'EVIDENCE_INVALID', v_now,
      jsonb_build_object(
        'recovery_status', v_directive.recovery_status,
        'recovery_legs', v_directive.recovery_legs
      )
    );
    return jsonb_build_object(
      'started', false,
      'duplicate', false,
      'resume_only', false,
      'status', 'EVIDENCE_INVALID',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
    );
  end if;

  v_leg_count := 0;
  for v_leg in
    select value
    from jsonb_array_elements(v_directive.recovery_legs)
  loop
    v_leg_count := v_leg_count + 1;
    if jsonb_typeof(v_leg) <> 'object'
       or nullif(trim(v_leg->>'asset_id'), '') is null
       or coalesce((v_leg->>'reduce_only')::boolean, false) is not true
       or nullif(v_leg->>'reduce_weight','')::double precision is null
       or (v_leg->>'reduce_weight')::double precision <= 0
       or (v_leg->>'current_direction')::integer not in (-1,1)
       or (v_leg->>'order_direction')::integer
            <> -(v_leg->>'current_direction')::integer
       or abs((v_leg->>'target_weight')::double precision)
            > abs((v_leg->>'current_weight')::double precision) + 1e-12
       or abs(
            (v_leg->>'reduce_weight')::double precision
            - (
              abs((v_leg->>'current_weight')::double precision)
              - abs((v_leg->>'target_weight')::double precision)
            )
          ) > 1e-12 then
      insert into public.brian_shadow_recovery_start_events(
        runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
        worker_token, recovery_claim_fencing_token, runtime_fencing_token,
        runtime_version, head_state_id,
        event, observed_at,
        metadata
      ) values (
        p_runtime_id, v_directive.dispatch_id, p_cycle_id,
        v_directive.cancel_risk_receipt_id,
        p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
        v_runtime_version, v_head_state_id,
        'EVIDENCE_INVALID', v_now,
        jsonb_build_object('reason', 'INVALID_REDUCE_ONLY_LEG', 'leg', v_leg)
      );
      return jsonb_build_object(
        'started', false,
        'duplicate', false,
        'resume_only', false,
        'status', 'EVIDENCE_INVALID',
        'runtime_id', p_runtime_id,
        'dispatch_id', v_directive.dispatch_id,
        'cycle_id', p_cycle_id,
        'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
        'runtime_version', v_runtime_version,
        'head_state_id', v_head_state_id,
        'fencing_token', p_fencing_token,
        'recovery_claim_fencing_token', p_recovery_claim_fencing_token
      );
    end if;
  end loop;

  if v_leg_count <= 0 then
    raise exception 'PHASE83_START: recovery directive unexpectedly has zero legs';
  end if;

  select *
    into v_claim
  from public.brian_shadow_recovery_claims
  where runtime_id = p_runtime_id
    and dispatch_id = v_directive.dispatch_id
    and cancel_risk_receipt_id = v_directive.cancel_risk_receipt_id
  for update;

  if not found
     or v_claim.status <> 'CLAIMED'
     or v_claim.worker_token <> p_worker_token
     or v_claim.claim_fencing_token <> p_recovery_claim_fencing_token
     or v_claim.claim_until <= v_now then
    insert into public.brian_shadow_recovery_start_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, recovery_claim_fencing_token, runtime_fencing_token,
      runtime_version, head_state_id,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      v_runtime_version, v_head_state_id,
      'CLAIM_LOST', v_now,
      jsonb_build_object(
        'current_worker_token', v_claim.worker_token,
        'current_claim_fencing_token', v_claim.claim_fencing_token,
        'current_claim_until', v_claim.claim_until
      )
    );
    return jsonb_build_object(
      'started', false,
      'duplicate', false,
      'resume_only', false,
      'status', 'CLAIM_LOST',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
    );
  end if;

  -- Lost-response retry / takeover recovery is evaluated only after the current
  -- claim has been proven. A new valid claim may resume an existing STARTED
  -- recovery but can never manufacture a second STARTED row.
  select *
    into v_existing
  from public.brian_shadow_recovery_starts
  where runtime_id = p_runtime_id
    and dispatch_id = v_directive.dispatch_id
    and cancel_risk_receipt_id = v_directive.cancel_risk_receipt_id;

  if found then
    if v_existing.cycle_id <> p_cycle_id
       or v_existing.directive_source_runtime_version
            <> v_directive.source_runtime_version
       or v_existing.directive_current_state_id <> v_directive.current_state_id
       or v_existing.recovery_legs is distinct from v_directive.recovery_legs then
      raise exception 'PHASE83_START_CONFLICT: stored recovery STARTED evidence changed';
    end if;

    v_resume :=
      v_existing.worker_token <> p_worker_token
      or v_existing.recovery_claim_fencing_token
           <> p_recovery_claim_fencing_token;

    insert into public.brian_shadow_recovery_start_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, recovery_claim_fencing_token, runtime_fencing_token,
      runtime_version, head_state_id,
      risk_version, risk_receipt_id, risk_state,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      v_runtime_version, v_head_state_id,
      v_existing.risk_version_at_start,
      v_existing.risk_receipt_id_at_start,
      v_existing.risk_state_at_start,
      case when v_resume then 'STARTED_RESUME' else 'STARTED_ALREADY' end,
      v_now,
      jsonb_build_object(
        'original_worker_token', v_existing.worker_token,
        'original_recovery_claim_fencing_token',
          v_existing.recovery_claim_fencing_token
      )
    );

    return jsonb_build_object(
      'started', true,
      'duplicate', true,
      'resume_only', v_resume,
      'status', case
        when v_resume then 'STARTED_RESUME'
        else 'STARTED_ALREADY'
      end,
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token,
      'risk_version', v_existing.risk_version_at_start,
      'risk_receipt_id', v_existing.risk_receipt_id_at_start,
      'risk_state', v_existing.risk_state_at_start,
      'recovery_legs', v_existing.recovery_legs
    );
  end if;

  if v_runtime_version <> v_directive.source_runtime_version
     or v_head_state_id is distinct from v_directive.current_state_id then
    insert into public.brian_shadow_recovery_start_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, recovery_claim_fencing_token, runtime_fencing_token,
      runtime_version, head_state_id,
      event, observed_at,
      metadata
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      v_runtime_version, v_head_state_id,
      'HEAD_MOVED', v_now,
      jsonb_build_object(
        'directive_runtime_version', v_directive.source_runtime_version,
        'directive_state_id', v_directive.current_state_id
      )
    );
    return jsonb_build_object(
      'started', false,
      'duplicate', false,
      'resume_only', false,
      'status', 'HEAD_MOVED',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
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
    insert into public.brian_shadow_recovery_start_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, recovery_claim_fencing_token, runtime_fencing_token,
      runtime_version, head_state_id,
      event, observed_at
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      v_runtime_version, v_head_state_id,
      'RISK_STATE_UNAVAILABLE', v_now
    );
    return jsonb_build_object(
      'started', false,
      'duplicate', false,
      'resume_only', false,
      'status', 'RISK_STATE_UNAVAILABLE',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token
    );
  end if;

  v_risk_receipt_id := nullif(trim(v_risk_receipt->>'receipt_id'), '');
  v_risk_state := nullif(trim(v_risk_receipt->>'trading_state'), '');
  if v_risk_receipt_id is null
     or length(v_risk_receipt_id) <> 64
     or v_risk_state not in ('ACTIVE','REDUCING','HALTED') then
    raise exception 'PHASE83_START: malformed current risk receipt';
  end if;

  if v_risk_state = 'HALTED' then
    insert into public.brian_shadow_recovery_start_events(
      runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
      worker_token, recovery_claim_fencing_token, runtime_fencing_token,
      runtime_version, head_state_id,
      risk_version, risk_receipt_id, risk_state,
      event, observed_at
    ) values (
      p_runtime_id, v_directive.dispatch_id, p_cycle_id,
      v_directive.cancel_risk_receipt_id,
      p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
      v_runtime_version, v_head_state_id,
      v_risk_version, v_risk_receipt_id, v_risk_state,
      'WAIT_RISK_RELEASE', v_now
    );
    return jsonb_build_object(
      'started', false,
      'duplicate', false,
      'resume_only', false,
      'status', 'WAIT_RISK_RELEASE',
      'runtime_id', p_runtime_id,
      'dispatch_id', v_directive.dispatch_id,
      'cycle_id', p_cycle_id,
      'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
      'runtime_version', v_runtime_version,
      'head_state_id', v_head_state_id,
      'fencing_token', p_fencing_token,
      'recovery_claim_fencing_token', p_recovery_claim_fencing_token,
      'risk_version', v_risk_version,
      'risk_receipt_id', v_risk_receipt_id,
      'risk_state', v_risk_state
    );
  end if;

  insert into public.brian_shadow_recovery_starts(
    runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
    worker_token, recovery_claim_fencing_token,
    runtime_fencing_token, runtime_version_at_start,
    head_state_id_at_start,
    risk_version_at_start, risk_receipt_id_at_start, risk_state_at_start,
    directive_source_runtime_version, directive_current_state_id,
    recovery_legs, started_at
  ) values (
    p_runtime_id, v_directive.dispatch_id, p_cycle_id,
    v_directive.cancel_risk_receipt_id,
    p_worker_token, p_recovery_claim_fencing_token,
    p_fencing_token, v_runtime_version,
    v_head_state_id,
    v_risk_version, v_risk_receipt_id, v_risk_state,
    v_directive.source_runtime_version, v_directive.current_state_id,
    v_directive.recovery_legs, v_now
  );

  insert into public.brian_shadow_recovery_start_events(
    runtime_id, dispatch_id, cycle_id, cancel_risk_receipt_id,
    worker_token, recovery_claim_fencing_token, runtime_fencing_token,
    runtime_version, head_state_id,
    risk_version, risk_receipt_id, risk_state,
    event, observed_at,
    metadata
  ) values (
    p_runtime_id, v_directive.dispatch_id, p_cycle_id,
    v_directive.cancel_risk_receipt_id,
    p_worker_token, p_recovery_claim_fencing_token, p_fencing_token,
    v_runtime_version, v_head_state_id,
    v_risk_version, v_risk_receipt_id, v_risk_state,
    'STARTED', v_now,
    jsonb_build_object(
      'recovery_leg_count', jsonb_array_length(v_directive.recovery_legs)
    )
  );

  return jsonb_build_object(
    'started', true,
    'duplicate', false,
    'resume_only', false,
    'status', 'STARTED',
    'runtime_id', p_runtime_id,
    'dispatch_id', v_directive.dispatch_id,
    'cycle_id', p_cycle_id,
    'cancel_risk_receipt_id', v_directive.cancel_risk_receipt_id,
    'runtime_version', v_runtime_version,
    'head_state_id', v_head_state_id,
    'fencing_token', p_fencing_token,
    'recovery_claim_fencing_token', p_recovery_claim_fencing_token,
    'risk_version', v_risk_version,
    'risk_receipt_id', v_risk_receipt_id,
    'risk_state', v_risk_state,
    'recovery_legs', v_directive.recovery_legs
  );
end;
$$;

create or replace function public.brian_read_shadow_recovery_start(
  p_runtime_id text,
  p_cycle_id text
) returns setof public.brian_shadow_recovery_starts
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  select *
  from public.brian_shadow_recovery_starts
  where runtime_id = p_runtime_id
    and cycle_id = p_cycle_id
  order by started_at asc;
$$;

revoke all on function public.brian_mark_shadow_recovery_started(
  text,text,bigint,text,text,bigint
) from public, anon, authenticated;
revoke all on function public.brian_read_shadow_recovery_start(text,text)
  from public, anon, authenticated;

grant execute on function public.brian_mark_shadow_recovery_started(
  text,text,bigint,text,text,bigint
) to service_role;
grant execute on function public.brian_read_shadow_recovery_start(text,text)
  to service_role;

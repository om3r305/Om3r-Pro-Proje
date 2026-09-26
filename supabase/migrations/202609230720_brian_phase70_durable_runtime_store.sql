-- Brian Phase 70: transactional durable shadow-runtime persistence.
--
-- GitHub-only until explicit rollout. This migration deliberately reuses the
-- project's already-proven concurrency patterns:
--   * owner-token lease semantics from brian_collector_lease;
--   * transaction-scoped advisory serialization + CAS from Treasury/Ocean.
--
-- The database is the final concurrency boundary. A worker may calculate or
-- stage state in memory, but it cannot advance the durable runtime unless it
-- still owns the current lease/fencing token AND the expected checkpoint
-- version matches the authoritative head.
--
-- No live execution state is accepted. All persisted checkpoints/cycles remain
-- shadow/paper-only.

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

create table if not exists public.brian_shadow_runtime_heads (
  runtime_id text primary key,
  version bigint not null default 0 check (version >= 0),
  checkpoint_id text,
  checkpoint_payload jsonb,
  journal_hash text,
  head_state_id text,
  pending_cycle_id text,
  owner_token text not null,
  fencing_token bigint not null default 1 check (fencing_token > 0),
  acquired_at timestamptz not null,
  lease_until timestamptz not null,
  updated_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  check (length(trim(runtime_id)) > 0),
  check (
    (version = 0
      and checkpoint_id is null
      and checkpoint_payload is null
      and journal_hash is null
      and head_state_id is null
      and pending_cycle_id is null)
    or
    (version > 0
      and checkpoint_id is not null
      and checkpoint_payload is not null
      and journal_hash is not null
      and head_state_id is not null)
  )
);

alter table public.brian_shadow_runtime_heads enable row level security;
revoke all on public.brian_shadow_runtime_heads from anon, authenticated, service_role;

create table if not exists public.brian_shadow_runtime_checkpoints (
  runtime_id text not null,
  version bigint not null check (version > 0),
  checkpoint_id text not null,
  checkpoint_payload jsonb not null,
  journal_hash text not null,
  head_state_id text not null,
  pending_cycle_id text,
  fencing_token bigint not null check (fencing_token > 0),
  committed_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, version),
  unique (runtime_id, checkpoint_id),
  check (length(checkpoint_id) = 64),
  check (length(journal_hash) = 64),
  check (length(head_state_id) = 64),
  check (jsonb_typeof(checkpoint_payload) = 'object')
);

create index if not exists brian_shadow_runtime_checkpoints_runtime_time_idx
  on public.brian_shadow_runtime_checkpoints(runtime_id, committed_at desc);

alter table public.brian_shadow_runtime_checkpoints enable row level security;
revoke all on public.brian_shadow_runtime_checkpoints from anon, authenticated, service_role;

drop trigger if exists brian_shadow_runtime_checkpoints_append_only
  on public.brian_shadow_runtime_checkpoints;
create trigger brian_shadow_runtime_checkpoints_append_only
  before update or delete on public.brian_shadow_runtime_checkpoints
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_shadow_runtime_cycles (
  runtime_id text not null,
  cycle_id text not null,
  cycle_hash text not null,
  cycle_payload jsonb not null,
  first_seen_version bigint not null check (first_seen_version > 0),
  created_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, cycle_id),
  check (length(cycle_hash) = 64),
  check (jsonb_typeof(cycle_payload) = 'object')
);

alter table public.brian_shadow_runtime_cycles enable row level security;
revoke all on public.brian_shadow_runtime_cycles from anon, authenticated, service_role;

drop trigger if exists brian_shadow_runtime_cycles_append_only
  on public.brian_shadow_runtime_cycles;
create trigger brian_shadow_runtime_cycles_append_only
  before update or delete on public.brian_shadow_runtime_cycles
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_shadow_runtime_journal_entries (
  runtime_id text not null,
  sequence bigint not null check (sequence >= 0),
  entry_id text not null,
  previous_entry_id text,
  cycle_id text not null,
  stage text not null check (
    stage in (
      'CYCLE_CREATED',
      'PAPER_APPLIED',
      'LOCAL_PROJECTED',
      'RECONCILIATION_REQUIRED',
      'RECONCILED',
      'COMMITTED',
      'ABORTED'
    )
  ),
  cycle_hash text not null,
  artifact_hash text not null,
  artifact_ref text not null,
  entry_payload jsonb not null,
  first_seen_version bigint not null check (first_seen_version > 0),
  created_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, sequence),
  unique (runtime_id, entry_id),
  check (length(entry_id) = 64),
  check (previous_entry_id is null or length(previous_entry_id) = 64),
  check (length(cycle_hash) = 64),
  check (length(artifact_hash) = 64),
  check (jsonb_typeof(entry_payload) = 'object')
);

create index if not exists brian_shadow_runtime_journal_cycle_idx
  on public.brian_shadow_runtime_journal_entries(runtime_id, cycle_id, sequence);

alter table public.brian_shadow_runtime_journal_entries enable row level security;
revoke all on public.brian_shadow_runtime_journal_entries from anon, authenticated, service_role;

drop trigger if exists brian_shadow_runtime_journal_entries_append_only
  on public.brian_shadow_runtime_journal_entries;
create trigger brian_shadow_runtime_journal_entries_append_only
  before update or delete on public.brian_shadow_runtime_journal_entries
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_shadow_runtime_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  owner_token text not null,
  fencing_token bigint,
  runtime_version bigint,
  event text not null check (
    event in (
      'ACQUIRED',
      'ALREADY_OWNED',
      'BLOCKED_ACTIVE',
      'EXPIRED_RECOVERY',
      'RENEWED',
      'RENEWAL_LOST',
      'RELEASED',
      'CHECKPOINT_COMMITTED',
      'CHECKPOINT_DUPLICATE',
      'CAS_CONFLICT',
      'LEASE_LOST'
    )
  ),
  observed_at timestamptz not null default now(),
  lease_until timestamptz,
  checkpoint_id text,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_shadow_runtime_events_runtime_time_idx
  on public.brian_shadow_runtime_events(runtime_id, observed_at desc, event_sequence desc);

alter table public.brian_shadow_runtime_events enable row level security;
revoke all on public.brian_shadow_runtime_events from anon, authenticated, service_role;

drop trigger if exists brian_shadow_runtime_events_append_only
  on public.brian_shadow_runtime_events;
create trigger brian_shadow_runtime_events_append_only
  before update or delete on public.brian_shadow_runtime_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_acquire_shadow_runtime_lease(
  p_runtime_id text,
  p_owner_token text,
  p_lease_seconds integer
) returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_now timestamptz;
  v_owner text;
  v_fence bigint;
  v_version bigint;
  v_lease_until timestamptz;
  v_event text;
  v_acquired boolean := false;
begin
  if nullif(trim(p_runtime_id), '') is null then
    raise exception 'PHASE70_LEASE: runtime_id is required';
  end if;
  if nullif(trim(p_owner_token), '') is null then
    raise exception 'PHASE70_LEASE: owner_token is required';
  end if;
  if p_lease_seconds is null or p_lease_seconds <= 0 then
    raise exception 'PHASE70_LEASE: lease_seconds must be positive';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, lease_until
    into v_owner, v_fence, v_version, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if not found then
    v_fence := 1;
    v_version := 0;
    v_lease_until := v_now + make_interval(secs => p_lease_seconds);
    insert into public.brian_shadow_runtime_heads(
      runtime_id, version, owner_token, fencing_token,
      acquired_at, lease_until, updated_at
    ) values (
      p_runtime_id, 0, p_owner_token, v_fence,
      v_now, v_lease_until, v_now
    );
    v_event := 'ACQUIRED';
    v_acquired := true;
  elsif v_owner = p_owner_token and v_lease_until > v_now then
    v_event := 'ALREADY_OWNED';
    v_acquired := true;
  elsif v_lease_until <= v_now then
    v_fence := v_fence + 1;
    v_lease_until := v_now + make_interval(secs => p_lease_seconds);
    update public.brian_shadow_runtime_heads
      set owner_token = p_owner_token,
          fencing_token = v_fence,
          acquired_at = v_now,
          lease_until = v_lease_until,
          updated_at = v_now
      where runtime_id = p_runtime_id;
    v_event := 'EXPIRED_RECOVERY';
    v_acquired := true;
  else
    v_event := 'BLOCKED_ACTIVE';
    v_acquired := false;
  end if;

  insert into public.brian_shadow_runtime_events(
    runtime_id, owner_token, fencing_token, runtime_version,
    event, observed_at, lease_until, metadata
  ) values (
    p_runtime_id, p_owner_token, v_fence, v_version,
    v_event, v_now, v_lease_until,
    jsonb_build_object('lease_seconds', p_lease_seconds)
  );

  return jsonb_build_object(
    'acquired', v_acquired,
    'status', v_event,
    'runtime_id', p_runtime_id,
    'fencing_token', v_fence,
    'version', v_version,
    'lease_until', v_lease_until
  );
end;
$$;

create or replace function public.brian_renew_shadow_runtime_lease(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_lease_seconds integer
) returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_now timestamptz;
  v_rows integer;
  v_version bigint;
  v_current_fence bigint;
  v_lease_until timestamptz;
  v_renewed boolean;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null
     or p_fencing_token <= 0
     or p_lease_seconds is null
     or p_lease_seconds <= 0 then
    raise exception 'PHASE70_RENEW: valid runtime/owner/fence/lease_seconds required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  update public.brian_shadow_runtime_heads
    set lease_until = v_now + make_interval(secs => p_lease_seconds),
        updated_at = v_now
    where runtime_id = p_runtime_id
      and owner_token = p_owner_token
      and fencing_token = p_fencing_token
      and lease_until > v_now
    returning version, fencing_token, lease_until
    into v_version, v_current_fence, v_lease_until;

  get diagnostics v_rows = row_count;
  v_renewed := v_rows > 0;

  if not v_renewed then
    select version, fencing_token, lease_until
      into v_version, v_current_fence, v_lease_until
    from public.brian_shadow_runtime_heads
    where runtime_id = p_runtime_id;
  end if;

  insert into public.brian_shadow_runtime_events(
    runtime_id, owner_token, fencing_token, runtime_version,
    event, observed_at, lease_until, metadata
  ) values (
    p_runtime_id, p_owner_token, coalesce(v_current_fence, p_fencing_token), v_version,
    case when v_renewed then 'RENEWED' else 'RENEWAL_LOST' end,
    v_now, v_lease_until,
    jsonb_build_object('requested_fence', p_fencing_token, 'lease_seconds', p_lease_seconds)
  );

  return jsonb_build_object(
    'renewed', v_renewed,
    'status', case when v_renewed then 'RENEWED' else 'RENEWAL_LOST' end,
    'runtime_id', p_runtime_id,
    'fencing_token', coalesce(v_current_fence, p_fencing_token),
    'version', v_version,
    'lease_until', v_lease_until
  );
end;
$$;

create or replace function public.brian_release_shadow_runtime_lease(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint
) returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_now timestamptz;
  v_rows integer;
  v_version bigint;
  v_current_fence bigint;
  v_released boolean;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null
     or p_fencing_token <= 0 then
    raise exception 'PHASE70_RELEASE: valid runtime/owner/fence required';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  update public.brian_shadow_runtime_heads
    set owner_token = 'released:' || gen_random_uuid()::text,
        lease_until = v_now,
        updated_at = v_now
    where runtime_id = p_runtime_id
      and owner_token = p_owner_token
      and fencing_token = p_fencing_token
    returning version, fencing_token
    into v_version, v_current_fence;

  get diagnostics v_rows = row_count;
  v_released := v_rows > 0;

  if not v_released then
    select version, fencing_token
      into v_version, v_current_fence
    from public.brian_shadow_runtime_heads
    where runtime_id = p_runtime_id;
  end if;

  if v_released then
    insert into public.brian_shadow_runtime_events(
      runtime_id, owner_token, fencing_token, runtime_version,
      event, observed_at, lease_until
    ) values (
      p_runtime_id, p_owner_token, p_fencing_token, v_version,
      'RELEASED', v_now, v_now
    );
  end if;

  return jsonb_build_object(
    'released', v_released,
    'status', case when v_released then 'RELEASED' else 'RELEASE_LOST' end,
    'runtime_id', p_runtime_id,
    'fencing_token', coalesce(v_current_fence, p_fencing_token),
    'version', v_version
  );
end;
$$;

create or replace function public.brian_commit_shadow_runtime_checkpoint(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_expected_version bigint,
  p_checkpoint jsonb
) returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_now timestamptz;
  v_current_owner text;
  v_current_fence bigint;
  v_current_version bigint;
  v_lease_until timestamptz;
  v_checkpoint_id text;
  v_runtime jsonb;
  v_ledger jsonb;
  v_journal jsonb;
  v_cycles jsonb;
  v_entries jsonb;
  v_journal_hash text;
  v_head_state_id text;
  v_pending_cycle_id text;
  v_runtime_pending text;
  v_existing_version bigint;
  v_existing_payload jsonb;
  v_new_version bigint;
  v_entry jsonb;
  v_cycle_key text;
  v_cycle_payload jsonb;
  v_cycle_hash text;
  v_expected_sequence bigint := 0;
  v_previous_entry_id text := null;
  v_stage text;
  v_entry_id text;
  v_entry_previous text;
  v_entry_cycle_id text;
  v_entry_cycle_hash text;
  v_artifact_hash text;
  v_artifact_ref text;
  v_existing_entry jsonb;
  v_existing_entry_id text;
  v_persisted_entry_count bigint;
  v_incoming_entry_count bigint;
  v_created_count integer;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null
     or p_fencing_token <= 0
     or p_expected_version is null
     or p_expected_version < 0 then
    raise exception 'PHASE70_COMMIT: valid runtime/owner/fence/expected_version required';
  end if;
  if p_checkpoint is null or jsonb_typeof(p_checkpoint) <> 'object' then
    raise exception 'PHASE70_COMMIT: checkpoint must be a json object';
  end if;

  v_checkpoint_id := nullif(trim(p_checkpoint->>'checkpoint_id'), '');
  v_runtime := p_checkpoint->'runtime_checkpoint';
  v_journal := p_checkpoint->'journal_manifest';
  if v_checkpoint_id is null or length(v_checkpoint_id) <> 64 then
    raise exception 'PHASE70_COMMIT: checkpoint_id must be a 64-char content id';
  end if;
  if coalesce((p_checkpoint->>'live_execution')::boolean, false) then
    raise exception 'PHASE70_COMMIT: live execution checkpoint rejected';
  end if;
  if jsonb_typeof(v_runtime) <> 'object' or jsonb_typeof(v_journal) <> 'object' then
    raise exception 'PHASE70_COMMIT: runtime_checkpoint and journal_manifest are required';
  end if;
  if coalesce((v_runtime->>'live_execution')::boolean, false) then
    raise exception 'PHASE70_COMMIT: nested runtime crossed live boundary';
  end if;
  if coalesce((v_journal->>'append_only')::boolean, false) is not true
     or coalesce((v_journal->>'shadow_only')::boolean, false) is not true
     or coalesce((v_journal->>'live_execution')::boolean, false) is not false then
    raise exception 'PHASE70_COMMIT: journal must be append-only shadow-only';
  end if;

  v_ledger := v_runtime->'shadow_ledger_manifest';
  v_cycles := v_journal->'cycles';
  v_entries := v_journal->'entries';
  if jsonb_typeof(v_ledger) <> 'object'
     or jsonb_typeof(v_cycles) <> 'object'
     or jsonb_typeof(v_entries) <> 'array' then
    raise exception 'PHASE70_COMMIT: malformed ledger/journal payload';
  end if;

  v_journal_hash := nullif(trim(v_journal->>'journal_hash'), '');
  v_head_state_id := nullif(trim(v_ledger->>'head_state_id'), '');
  v_pending_cycle_id := nullif(trim(v_ledger->>'pending_cycle_id'), '');
  v_runtime_pending := nullif(trim(v_runtime->>'pending_cycle_id'), '');
  if v_journal_hash is null or length(v_journal_hash) <> 64 then
    raise exception 'PHASE70_COMMIT: journal_hash must be a 64-char content id';
  end if;
  if v_head_state_id is null or length(v_head_state_id) <> 64 then
    raise exception 'PHASE70_COMMIT: head_state_id must be a 64-char content id';
  end if;
  if v_pending_cycle_id is distinct from v_runtime_pending then
    raise exception 'PHASE70_COMMIT: runtime/ledger pending_cycle_id mismatch';
  end if;

  if coalesce((v_runtime->'paper'->>'live_execution')::boolean, false)
     or coalesce((v_runtime->'paper'->>'paper_only')::boolean, false) is not true then
    raise exception 'PHASE70_COMMIT: paper checkpoint crossed shadow/paper boundary';
  end if;

  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, version, lease_until
    into v_current_owner, v_current_fence, v_current_version, v_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if not found
     or v_current_owner <> p_owner_token
     or v_current_fence <> p_fencing_token
     or v_lease_until <= v_now then
    insert into public.brian_shadow_runtime_events(
      runtime_id, owner_token, fencing_token, runtime_version,
      event, observed_at, lease_until, checkpoint_id,
      metadata
    ) values (
      p_runtime_id, p_owner_token, p_fencing_token, v_current_version,
      'LEASE_LOST', v_now, v_lease_until, v_checkpoint_id,
      jsonb_build_object(
        'current_fence', v_current_fence,
        'expected_version', p_expected_version
      )
    );
    return jsonb_build_object(
      'committed', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'fencing_token', v_current_fence,
      'version', v_current_version,
      'checkpoint_id', v_checkpoint_id
    );
  end if;

  select version, checkpoint_payload
    into v_existing_version, v_existing_payload
  from public.brian_shadow_runtime_checkpoints
  where runtime_id = p_runtime_id
    and checkpoint_id = v_checkpoint_id;

  if found then
    if v_existing_payload is distinct from p_checkpoint then
      raise exception 'PHASE70_CHECKPOINT_ID_CONFLICT: identical checkpoint_id has different payload';
    end if;
    insert into public.brian_shadow_runtime_events(
      runtime_id, owner_token, fencing_token, runtime_version,
      event, observed_at, lease_until, checkpoint_id,
      metadata
    ) values (
      p_runtime_id, p_owner_token, p_fencing_token, v_current_version,
      'CHECKPOINT_DUPLICATE', v_now, v_lease_until, v_checkpoint_id,
      jsonb_build_object('committed_version', v_existing_version)
    );
    return jsonb_build_object(
      'committed', true,
      'duplicate', true,
      'status', case
        when v_existing_version = v_current_version then 'DUPLICATE_CURRENT'
        else 'DUPLICATE_HISTORICAL'
      end,
      'runtime_id', p_runtime_id,
      'fencing_token', p_fencing_token,
      'version', v_existing_version,
      'current_version', v_current_version,
      'checkpoint_id', v_checkpoint_id
    );
  end if;

  if v_current_version <> p_expected_version then
    insert into public.brian_shadow_runtime_events(
      runtime_id, owner_token, fencing_token, runtime_version,
      event, observed_at, lease_until, checkpoint_id,
      metadata
    ) values (
      p_runtime_id, p_owner_token, p_fencing_token, v_current_version,
      'CAS_CONFLICT', v_now, v_lease_until, v_checkpoint_id,
      jsonb_build_object('expected_version', p_expected_version)
    );
    return jsonb_build_object(
      'committed', false,
      'status', 'CAS_CONFLICT',
      'runtime_id', p_runtime_id,
      'fencing_token', p_fencing_token,
      'version', v_current_version,
      'expected_version', p_expected_version,
      'checkpoint_id', v_checkpoint_id
    );
  end if;

  v_incoming_entry_count := jsonb_array_length(v_entries);
  select count(*) into v_persisted_entry_count
  from public.brian_shadow_runtime_journal_entries
  where runtime_id = p_runtime_id;
  if v_incoming_entry_count < v_persisted_entry_count then
    raise exception 'PHASE70_JOURNAL_TRUNCATION: incoming % < persisted %',
      v_incoming_entry_count, v_persisted_entry_count;
  end if;

  -- Validate ordered global hash-chain structure before any evidence insert.
  for v_entry in select value from jsonb_array_elements(v_entries)
  loop
    if jsonb_typeof(v_entry) <> 'object' then
      raise exception 'PHASE70_JOURNAL: every entry must be an object';
    end if;
    if (v_entry->>'sequence')::bigint <> v_expected_sequence then
      raise exception 'PHASE70_JOURNAL_SEQUENCE: expected %, got %',
        v_expected_sequence, v_entry->>'sequence';
    end if;

    v_entry_id := nullif(trim(v_entry->>'entry_id'), '');
    v_entry_previous := nullif(trim(v_entry->>'previous_entry_id'), '');
    v_entry_cycle_id := nullif(trim(v_entry->>'cycle_id'), '');
    v_entry_cycle_hash := nullif(trim(v_entry->>'cycle_hash'), '');
    v_artifact_hash := nullif(trim(v_entry->>'artifact_hash'), '');
    v_artifact_ref := nullif(trim(v_entry->>'artifact_ref'), '');
    v_stage := nullif(trim(v_entry->>'stage'), '');

    if v_entry_id is null or length(v_entry_id) <> 64
       or v_entry_cycle_id is null
       or v_entry_cycle_hash is null or length(v_entry_cycle_hash) <> 64
       or v_artifact_hash is null or length(v_artifact_hash) <> 64
       or v_artifact_ref is null then
      raise exception 'PHASE70_JOURNAL: invalid entry identity at sequence %', v_expected_sequence;
    end if;
    if v_expected_sequence = 0 then
      if v_entry_previous is not null then
        raise exception 'PHASE70_JOURNAL_CHAIN: first entry must not have previous_entry_id';
      end if;
    elsif v_entry_previous is distinct from v_previous_entry_id then
      raise exception 'PHASE70_JOURNAL_CHAIN: broken previous_entry_id at sequence %',
        v_expected_sequence;
    end if;
    if v_stage not in (
      'CYCLE_CREATED',
      'PAPER_APPLIED',
      'LOCAL_PROJECTED',
      'RECONCILIATION_REQUIRED',
      'RECONCILED',
      'COMMITTED',
      'ABORTED'
    ) then
      raise exception 'PHASE70_JOURNAL: unsupported stage %', v_stage;
    end if;
    if not (v_cycles ? v_entry_cycle_id) then
      raise exception 'PHASE70_JOURNAL: entry references missing cycle %', v_entry_cycle_id;
    end if;

    select entry_id, entry_payload
      into v_existing_entry_id, v_existing_entry
    from public.brian_shadow_runtime_journal_entries
    where runtime_id = p_runtime_id
      and sequence = v_expected_sequence;

    if found and (
      v_existing_entry_id <> v_entry_id
      or v_existing_entry is distinct from v_entry
    ) then
      raise exception 'PHASE70_JOURNAL_PREFIX_CONFLICT at sequence %', v_expected_sequence;
    end if;

    v_previous_entry_id := v_entry_id;
    v_expected_sequence := v_expected_sequence + 1;
  end loop;

  -- Every full cycle body must be shadow-only and anchored by exactly one
  -- CYCLE_CREATED entry carrying its immutable cycle hash.
  for v_cycle_key, v_cycle_payload in
    select key, value from jsonb_each(v_cycles)
  loop
    if jsonb_typeof(v_cycle_payload) <> 'object'
       or nullif(trim(v_cycle_payload->>'cycle_id'), '') is distinct from v_cycle_key
       or coalesce((v_cycle_payload->>'shadow_only')::boolean, false) is not true
       or coalesce((v_cycle_payload->>'live_execution')::boolean, false) is not false
       or coalesce((v_cycle_payload->>'account_state_mutated')::boolean, false) is not false then
      raise exception 'PHASE70_CYCLE: invalid shadow cycle body %', v_cycle_key;
    end if;

    select min(value->>'cycle_hash'), count(*)
      into v_cycle_hash, v_created_count
    from jsonb_array_elements(v_entries)
    where value->>'cycle_id' = v_cycle_key
      and value->>'stage' = 'CYCLE_CREATED';

    if v_created_count <> 1 or v_cycle_hash is null or length(v_cycle_hash) <> 64 then
      raise exception 'PHASE70_CYCLE: % requires exactly one CYCLE_CREATED hash anchor', v_cycle_key;
    end if;

    if exists(
      select 1
      from public.brian_shadow_runtime_cycles c
      where c.runtime_id = p_runtime_id
        and c.cycle_id = v_cycle_key
        and (c.cycle_hash <> v_cycle_hash or c.cycle_payload is distinct from v_cycle_payload)
    ) then
      raise exception 'PHASE70_CYCLE_CONFLICT: cycle % changed durable body/hash', v_cycle_key;
    end if;
  end loop;

  v_new_version := v_current_version + 1;

  for v_cycle_key, v_cycle_payload in
    select key, value from jsonb_each(v_cycles)
  loop
    select min(value->>'cycle_hash')
      into v_cycle_hash
    from jsonb_array_elements(v_entries)
    where value->>'cycle_id' = v_cycle_key
      and value->>'stage' = 'CYCLE_CREATED';

    insert into public.brian_shadow_runtime_cycles(
      runtime_id, cycle_id, cycle_hash, cycle_payload, first_seen_version
    ) values (
      p_runtime_id, v_cycle_key, v_cycle_hash, v_cycle_payload, v_new_version
    )
    on conflict (runtime_id, cycle_id) do nothing;
  end loop;

  for v_entry in select value from jsonb_array_elements(v_entries)
  loop
    insert into public.brian_shadow_runtime_journal_entries(
      runtime_id, sequence, entry_id, previous_entry_id,
      cycle_id, stage, cycle_hash, artifact_hash, artifact_ref,
      entry_payload, first_seen_version
    ) values (
      p_runtime_id,
      (v_entry->>'sequence')::bigint,
      v_entry->>'entry_id',
      nullif(trim(v_entry->>'previous_entry_id'), ''),
      v_entry->>'cycle_id',
      v_entry->>'stage',
      v_entry->>'cycle_hash',
      v_entry->>'artifact_hash',
      v_entry->>'artifact_ref',
      v_entry,
      v_new_version
    )
    on conflict (runtime_id, sequence) do nothing;
  end loop;

  insert into public.brian_shadow_runtime_checkpoints(
    runtime_id, version, checkpoint_id, checkpoint_payload,
    journal_hash, head_state_id, pending_cycle_id,
    fencing_token, committed_at
  ) values (
    p_runtime_id, v_new_version, v_checkpoint_id, p_checkpoint,
    v_journal_hash, v_head_state_id, v_pending_cycle_id,
    p_fencing_token, v_now
  );

  update public.brian_shadow_runtime_heads
    set version = v_new_version,
        checkpoint_id = v_checkpoint_id,
        checkpoint_payload = p_checkpoint,
        journal_hash = v_journal_hash,
        head_state_id = v_head_state_id,
        pending_cycle_id = v_pending_cycle_id,
        updated_at = v_now
    where runtime_id = p_runtime_id
      and owner_token = p_owner_token
      and fencing_token = p_fencing_token;

  insert into public.brian_shadow_runtime_events(
    runtime_id, owner_token, fencing_token, runtime_version,
    event, observed_at, lease_until, checkpoint_id,
    metadata
  ) values (
    p_runtime_id, p_owner_token, p_fencing_token, v_new_version,
    'CHECKPOINT_COMMITTED', v_now, v_lease_until, v_checkpoint_id,
    jsonb_build_object(
      'expected_version', p_expected_version,
      'journal_hash', v_journal_hash,
      'head_state_id', v_head_state_id,
      'pending_cycle_id', v_pending_cycle_id,
      'journal_entries', v_incoming_entry_count
    )
  );

  return jsonb_build_object(
    'committed', true,
    'duplicate', false,
    'status', 'COMMITTED',
    'runtime_id', p_runtime_id,
    'fencing_token', p_fencing_token,
    'version', v_new_version,
    'checkpoint_id', v_checkpoint_id,
    'journal_hash', v_journal_hash,
    'head_state_id', v_head_state_id,
    'pending_cycle_id', v_pending_cycle_id
  );
end;
$$;

create or replace function public.brian_read_shadow_runtime_checkpoint(
  p_runtime_id text
) returns jsonb
language sql
security definer
set search_path = pg_catalog, public
stable
as $$
  select case
    when h.runtime_id is null then null
    else jsonb_build_object(
      'runtime_id', h.runtime_id,
      'version', h.version,
      'checkpoint_id', h.checkpoint_id,
      'checkpoint_payload', h.checkpoint_payload,
      'journal_hash', h.journal_hash,
      'head_state_id', h.head_state_id,
      'pending_cycle_id', h.pending_cycle_id,
      'fencing_token', h.fencing_token,
      'lease_until', h.lease_until,
      'shadow_only', h.shadow_only,
      'live_execution', h.live_execution
    )
  end
  from (select p_runtime_id as requested_id) q
  left join public.brian_shadow_runtime_heads h
    on h.runtime_id = q.requested_id;
$$;

revoke all on function public.brian_acquire_shadow_runtime_lease(text,text,integer)
  from public, anon, authenticated;
revoke all on function public.brian_renew_shadow_runtime_lease(text,text,bigint,integer)
  from public, anon, authenticated;
revoke all on function public.brian_release_shadow_runtime_lease(text,text,bigint)
  from public, anon, authenticated;
revoke all on function public.brian_commit_shadow_runtime_checkpoint(text,text,bigint,bigint,jsonb)
  from public, anon, authenticated;
revoke all on function public.brian_read_shadow_runtime_checkpoint(text)
  from public, anon, authenticated;

grant execute on function public.brian_acquire_shadow_runtime_lease(text,text,integer)
  to service_role;
grant execute on function public.brian_renew_shadow_runtime_lease(text,text,bigint,integer)
  to service_role;
grant execute on function public.brian_release_shadow_runtime_lease(text,text,bigint)
  to service_role;
grant execute on function public.brian_commit_shadow_runtime_checkpoint(text,text,bigint,bigint,jsonb)
  to service_role;
grant execute on function public.brian_read_shadow_runtime_checkpoint(text)
  to service_role;

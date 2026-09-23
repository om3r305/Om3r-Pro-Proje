-- Brian Phase 73: durable operational-risk ledger persistence.
--
-- GitHub-only until explicit rollout. This extends the Phase70 runtime lease /
-- fencing boundary to Phase72 operational-risk state. Risk state and execution
-- state share the same runtime ownership primitive, so a stale worker cannot
-- advance either side after lease takeover.
--
-- The mutable head is operational state. Snapshots, ledger entries and audit
-- events are append-only evidence.

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

create table if not exists public.brian_operational_risk_heads (
  runtime_id text primary key,
  version bigint not null default 0 check (version >= 0),
  ledger_hash text,
  manifest jsonb,
  policy_hash text,
  head_entry_id text,
  current_state text,
  halt_latched boolean,
  updated_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  check (
    (version = 0
      and ledger_hash is null
      and manifest is null
      and policy_hash is null
      and head_entry_id is null
      and current_state is null
      and halt_latched is null)
    or
    (version > 0
      and ledger_hash is not null
      and manifest is not null
      and policy_hash is not null
      and current_state in ('ACTIVE','REDUCING','HALTED')
      and halt_latched is not null)
  )
);

alter table public.brian_operational_risk_heads enable row level security;
revoke all on public.brian_operational_risk_heads from anon, authenticated, service_role;

create table if not exists public.brian_operational_risk_snapshots (
  runtime_id text not null,
  version bigint not null check (version > 0),
  ledger_hash text not null,
  manifest jsonb not null,
  policy_hash text not null,
  head_entry_id text,
  current_state text not null check (current_state in ('ACTIVE','REDUCING','HALTED')),
  halt_latched boolean not null,
  fencing_token bigint not null check (fencing_token > 0),
  committed_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, version),
  unique (runtime_id, ledger_hash),
  check (length(ledger_hash) = 64),
  check (length(policy_hash) = 64),
  check (head_entry_id is null or length(head_entry_id) = 64),
  check (jsonb_typeof(manifest) = 'object'),
  check (halt_latched = (current_state = 'HALTED'))
);

create index if not exists brian_operational_risk_snapshots_runtime_time_idx
  on public.brian_operational_risk_snapshots(runtime_id, committed_at desc);

alter table public.brian_operational_risk_snapshots enable row level security;
revoke all on public.brian_operational_risk_snapshots from anon, authenticated, service_role;

drop trigger if exists brian_operational_risk_snapshots_append_only
  on public.brian_operational_risk_snapshots;
create trigger brian_operational_risk_snapshots_append_only
  before update or delete on public.brian_operational_risk_snapshots
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_operational_risk_entries (
  runtime_id text not null,
  sequence bigint not null check (sequence >= 0),
  entry_id text not null,
  previous_entry_id text,
  policy_hash text not null,
  receipt_id text not null,
  receipt_timestamp double precision not null,
  previous_state text not null check (previous_state in ('ACTIVE','REDUCING','HALTED')),
  trading_state text not null check (trading_state in ('ACTIVE','REDUCING','HALTED')),
  halt_latched boolean not null,
  entry_payload jsonb not null,
  first_seen_version bigint not null check (first_seen_version > 0),
  created_at timestamptz not null default now(),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  primary key (runtime_id, sequence),
  unique (runtime_id, entry_id),
  unique (runtime_id, receipt_id),
  check (length(entry_id) = 64),
  check (previous_entry_id is null or length(previous_entry_id) = 64),
  check (length(policy_hash) = 64),
  check (length(receipt_id) = 64),
  check (
    receipt_timestamp not in (
      'NaN'::double precision,
      'Infinity'::double precision,
      '-Infinity'::double precision
    )
  ),
  check (halt_latched = (trading_state = 'HALTED')),
  check (jsonb_typeof(entry_payload) = 'object')
);

alter table public.brian_operational_risk_entries enable row level security;
revoke all on public.brian_operational_risk_entries from anon, authenticated, service_role;

drop trigger if exists brian_operational_risk_entries_append_only
  on public.brian_operational_risk_entries;
create trigger brian_operational_risk_entries_append_only
  before update or delete on public.brian_operational_risk_entries
  for each row execute function public.brian_reject_mutation();

create table if not exists public.brian_operational_risk_events (
  event_sequence bigint generated always as identity primary key,
  runtime_id text not null,
  owner_token text not null,
  fencing_token bigint,
  risk_version bigint,
  event text not null check (
    event in (
      'RISK_COMMITTED',
      'RISK_DUPLICATE',
      'RISK_CAS_CONFLICT',
      'RISK_LEASE_LOST'
    )
  ),
  observed_at timestamptz not null default now(),
  ledger_hash text,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution)
);

create index if not exists brian_operational_risk_events_runtime_time_idx
  on public.brian_operational_risk_events(runtime_id, observed_at desc, event_sequence desc);

alter table public.brian_operational_risk_events enable row level security;
revoke all on public.brian_operational_risk_events from anon, authenticated, service_role;

drop trigger if exists brian_operational_risk_events_append_only
  on public.brian_operational_risk_events;
create trigger brian_operational_risk_events_append_only
  before update or delete on public.brian_operational_risk_events
  for each row execute function public.brian_reject_mutation();

create or replace function public.brian_commit_operational_risk_ledger(
  p_runtime_id text,
  p_owner_token text,
  p_fencing_token bigint,
  p_expected_version bigint,
  p_manifest jsonb
) returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_now timestamptz;
  v_runtime_owner text;
  v_runtime_fence bigint;
  v_runtime_lease_until timestamptz;
  v_current_version bigint := 0;
  v_current_manifest jsonb;
  v_ledger_hash text;
  v_policy_hash text;
  v_head_entry_id text;
  v_current_state text;
  v_halt_latched boolean;
  v_entries jsonb;
  v_incoming_count bigint;
  v_persisted_count bigint;
  v_expected_sequence bigint := 0;
  v_previous_entry_id text := null;
  v_entry jsonb;
  v_entry_id text;
  v_entry_previous text;
  v_entry_policy_hash text;
  v_receipt jsonb;
  v_receipt_id text;
  v_receipt_timestamp double precision;
  v_previous_state text;
  v_trading_state text;
  v_entry_halt_latched boolean;
  v_existing_entry_id text;
  v_existing_entry_payload jsonb;
  v_existing_version bigint;
  v_existing_manifest jsonb;
  v_new_version bigint;
begin
  if nullif(trim(p_runtime_id), '') is null
     or nullif(trim(p_owner_token), '') is null
     or p_fencing_token is null
     or p_fencing_token <= 0
     or p_expected_version is null
     or p_expected_version < 0 then
    raise exception 'PHASE73_RISK_COMMIT: valid runtime/owner/fence/expected_version required';
  end if;
  if p_manifest is null or jsonb_typeof(p_manifest) <> 'object' then
    raise exception 'PHASE73_RISK_COMMIT: manifest must be an object';
  end if;
  if coalesce((p_manifest->>'append_only')::boolean, false) is not true
     or coalesce((p_manifest->>'shadow_only')::boolean, false) is not true
     or coalesce((p_manifest->>'live_execution')::boolean, false) is not false then
    raise exception 'PHASE73_RISK_COMMIT: manifest must be append-only shadow-only';
  end if;

  v_ledger_hash := nullif(trim(p_manifest->>'ledger_hash'), '');
  v_policy_hash := nullif(trim(p_manifest->>'policy_hash'), '');
  v_head_entry_id := nullif(trim(p_manifest->>'head_entry_id'), '');
  v_current_state := nullif(trim(p_manifest->>'current_state'), '');
  v_halt_latched := (p_manifest->>'halt_latched')::boolean;
  v_entries := p_manifest->'entries';

  if v_ledger_hash is null or length(v_ledger_hash) <> 64
     or v_policy_hash is null or length(v_policy_hash) <> 64
     or v_current_state not in ('ACTIVE','REDUCING','HALTED')
     or v_halt_latched is distinct from (v_current_state = 'HALTED')
     or jsonb_typeof(v_entries) <> 'array' then
    raise exception 'PHASE73_RISK_COMMIT: invalid manifest anchors';
  end if;

  v_incoming_count := jsonb_array_length(v_entries);
  if (p_manifest->>'entry_count')::bigint <> v_incoming_count then
    raise exception 'PHASE73_RISK_COMMIT: entry_count mismatch';
  end if;
  if v_incoming_count = 0 and v_head_entry_id is not null then
    raise exception 'PHASE73_RISK_COMMIT: empty ledger cannot have head_entry_id';
  end if;
  if v_incoming_count > 0 and (v_head_entry_id is null or length(v_head_entry_id) <> 64) then
    raise exception 'PHASE73_RISK_COMMIT: nonempty ledger requires head_entry_id';
  end if;

  -- Same advisory key as Phase70: runtime ownership, execution checkpoint CAS,
  -- and risk-ledger CAS serialize on one database mutex.
  perform pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtext('brian_shadow_runtime:' || p_runtime_id)
  );
  v_now := clock_timestamp();

  select owner_token, fencing_token, lease_until
    into v_runtime_owner, v_runtime_fence, v_runtime_lease_until
  from public.brian_shadow_runtime_heads
  where runtime_id = p_runtime_id
  for update;

  if not found
     or v_runtime_owner <> p_owner_token
     or v_runtime_fence <> p_fencing_token
     or v_runtime_lease_until <= v_now then
    insert into public.brian_operational_risk_events(
      runtime_id, owner_token, fencing_token, risk_version,
      event, observed_at, ledger_hash, metadata
    ) values (
      p_runtime_id, p_owner_token, p_fencing_token, null,
      'RISK_LEASE_LOST', v_now, v_ledger_hash,
      jsonb_build_object('current_fence', v_runtime_fence)
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'status', 'LEASE_LOST',
      'runtime_id', p_runtime_id,
      'fencing_token', v_runtime_fence,
      'version', null,
      'ledger_hash', v_ledger_hash
    );
  end if;

  select version, manifest
    into v_current_version, v_current_manifest
  from public.brian_operational_risk_heads
  where runtime_id = p_runtime_id
  for update;

  if not found then
    v_current_version := 0;
    v_current_manifest := null;
    insert into public.brian_operational_risk_heads(runtime_id)
    values (p_runtime_id);
  end if;

  select version, manifest
    into v_existing_version, v_existing_manifest
  from public.brian_operational_risk_snapshots
  where runtime_id = p_runtime_id
    and ledger_hash = v_ledger_hash;

  if found then
    if v_existing_manifest is distinct from p_manifest then
      raise exception 'PHASE73_LEDGER_HASH_CONFLICT: identical ledger_hash has different manifest';
    end if;
    insert into public.brian_operational_risk_events(
      runtime_id, owner_token, fencing_token, risk_version,
      event, observed_at, ledger_hash,
      metadata
    ) values (
      p_runtime_id, p_owner_token, p_fencing_token, v_current_version,
      'RISK_DUPLICATE', v_now, v_ledger_hash,
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
      'ledger_hash', v_ledger_hash
    );
  end if;

  if v_current_version <> p_expected_version then
    insert into public.brian_operational_risk_events(
      runtime_id, owner_token, fencing_token, risk_version,
      event, observed_at, ledger_hash,
      metadata
    ) values (
      p_runtime_id, p_owner_token, p_fencing_token, v_current_version,
      'RISK_CAS_CONFLICT', v_now, v_ledger_hash,
      jsonb_build_object('expected_version', p_expected_version)
    );
    return jsonb_build_object(
      'committed', false,
      'duplicate', false,
      'status', 'CAS_CONFLICT',
      'runtime_id', p_runtime_id,
      'fencing_token', p_fencing_token,
      'version', v_current_version,
      'expected_version', p_expected_version,
      'ledger_hash', v_ledger_hash
    );
  end if;

  select count(*) into v_persisted_count
  from public.brian_operational_risk_entries
  where runtime_id = p_runtime_id;

  if v_incoming_count < v_persisted_count then
    raise exception 'PHASE73_LEDGER_TRUNCATION: incoming % < persisted %',
      v_incoming_count, v_persisted_count;
  end if;

  -- Validate sequence, hash-chain links and persisted-prefix identity before any insert.
  for v_entry in select value from jsonb_array_elements(v_entries)
  loop
    if jsonb_typeof(v_entry) <> 'object' then
      raise exception 'PHASE73_LEDGER: every entry must be an object';
    end if;
    if (v_entry->>'sequence')::bigint <> v_expected_sequence then
      raise exception 'PHASE73_LEDGER_SEQUENCE: expected %, got %',
        v_expected_sequence, v_entry->>'sequence';
    end if;

    v_entry_id := nullif(trim(v_entry->>'entry_id'), '');
    v_entry_previous := nullif(trim(v_entry->>'previous_entry_id'), '');
    v_entry_policy_hash := nullif(trim(v_entry->>'policy_hash'), '');
    v_receipt := v_entry->'receipt';
    if v_entry_id is null or length(v_entry_id) <> 64
       or v_entry_policy_hash is distinct from v_policy_hash
       or jsonb_typeof(v_receipt) <> 'object' then
      raise exception 'PHASE73_LEDGER: invalid entry at sequence %', v_expected_sequence;
    end if;

    if v_expected_sequence = 0 then
      if v_entry_previous is not null then
        raise exception 'PHASE73_LEDGER_CHAIN: first entry cannot have previous';
      end if;
    elsif v_entry_previous is distinct from v_previous_entry_id then
      raise exception 'PHASE73_LEDGER_CHAIN: broken previous id at sequence %',
        v_expected_sequence;
    end if;

    v_receipt_id := nullif(trim(v_receipt->>'receipt_id'), '');
    v_receipt_timestamp := (v_receipt->>'timestamp')::double precision;
    v_previous_state := nullif(trim(v_receipt->>'previous_state'), '');
    v_trading_state := nullif(trim(v_receipt->>'trading_state'), '');
    v_entry_halt_latched := (v_receipt->>'halt_latched')::boolean;

    if v_receipt_id is null or length(v_receipt_id) <> 64
       or v_previous_state not in ('ACTIVE','REDUCING','HALTED')
       or v_trading_state not in ('ACTIVE','REDUCING','HALTED')
       or v_entry_halt_latched is distinct from (v_trading_state = 'HALTED') then
      raise exception 'PHASE73_LEDGER: invalid receipt at sequence %', v_expected_sequence;
    end if;

    select entry_id, entry_payload
      into v_existing_entry_id, v_existing_entry_payload
    from public.brian_operational_risk_entries
    where runtime_id = p_runtime_id
      and sequence = v_expected_sequence;

    if found and (
      v_existing_entry_id <> v_entry_id
      or v_existing_entry_payload is distinct from v_entry
    ) then
      raise exception 'PHASE73_LEDGER_PREFIX_CONFLICT at sequence %', v_expected_sequence;
    end if;

    v_previous_entry_id := v_entry_id;
    v_expected_sequence := v_expected_sequence + 1;
  end loop;

  if v_incoming_count > 0 and v_previous_entry_id is distinct from v_head_entry_id then
    raise exception 'PHASE73_LEDGER: head_entry_id does not match final entry';
  end if;

  v_new_version := v_current_version + 1;

  for v_entry in select value from jsonb_array_elements(v_entries)
  loop
    v_receipt := v_entry->'receipt';
    insert into public.brian_operational_risk_entries(
      runtime_id, sequence, entry_id, previous_entry_id, policy_hash,
      receipt_id, receipt_timestamp, previous_state, trading_state,
      halt_latched, entry_payload, first_seen_version
    ) values (
      p_runtime_id,
      (v_entry->>'sequence')::bigint,
      v_entry->>'entry_id',
      nullif(trim(v_entry->>'previous_entry_id'), ''),
      v_entry->>'policy_hash',
      v_receipt->>'receipt_id',
      (v_receipt->>'timestamp')::double precision,
      v_receipt->>'previous_state',
      v_receipt->>'trading_state',
      (v_receipt->>'halt_latched')::boolean,
      v_entry,
      v_new_version
    )
    on conflict (runtime_id, sequence) do nothing;
  end loop;

  insert into public.brian_operational_risk_snapshots(
    runtime_id, version, ledger_hash, manifest, policy_hash,
    head_entry_id, current_state, halt_latched, fencing_token,
    committed_at
  ) values (
    p_runtime_id, v_new_version, v_ledger_hash, p_manifest, v_policy_hash,
    v_head_entry_id, v_current_state, v_halt_latched, p_fencing_token,
    v_now
  );

  update public.brian_operational_risk_heads
    set version = v_new_version,
        ledger_hash = v_ledger_hash,
        manifest = p_manifest,
        policy_hash = v_policy_hash,
        head_entry_id = v_head_entry_id,
        current_state = v_current_state,
        halt_latched = v_halt_latched,
        updated_at = v_now
    where runtime_id = p_runtime_id;

  insert into public.brian_operational_risk_events(
    runtime_id, owner_token, fencing_token, risk_version,
    event, observed_at, ledger_hash,
    metadata
  ) values (
    p_runtime_id, p_owner_token, p_fencing_token, v_new_version,
    'RISK_COMMITTED', v_now, v_ledger_hash,
    jsonb_build_object(
      'expected_version', p_expected_version,
      'entry_count', v_incoming_count,
      'current_state', v_current_state,
      'halt_latched', v_halt_latched
    )
  );

  return jsonb_build_object(
    'committed', true,
    'duplicate', false,
    'status', 'COMMITTED',
    'runtime_id', p_runtime_id,
    'fencing_token', p_fencing_token,
    'version', v_new_version,
    'ledger_hash', v_ledger_hash,
    'policy_hash', v_policy_hash,
    'head_entry_id', v_head_entry_id,
    'current_state', v_current_state,
    'halt_latched', v_halt_latched
  );
end;
$$;

create or replace function public.brian_read_operational_risk_ledger(
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
      'ledger_hash', h.ledger_hash,
      'manifest', h.manifest,
      'policy_hash', h.policy_hash,
      'head_entry_id', h.head_entry_id,
      'current_state', h.current_state,
      'halt_latched', h.halt_latched,
      'shadow_only', h.shadow_only,
      'live_execution', h.live_execution
    )
  end
  from (select p_runtime_id as requested_id) q
  left join public.brian_operational_risk_heads h
    on h.runtime_id = q.requested_id;
$$;

revoke all on function public.brian_commit_operational_risk_ledger(text,text,bigint,bigint,jsonb)
  from public, anon, authenticated;
revoke all on function public.brian_read_operational_risk_ledger(text)
  from public, anon, authenticated;

grant execute on function public.brian_commit_operational_risk_ledger(text,text,bigint,bigint,jsonb)
  to service_role;
grant execute on function public.brian_read_operational_risk_ledger(text)
  to service_role;

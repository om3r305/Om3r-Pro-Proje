-- Brian 2026 Global Intelligence Fabric persistence.
-- SHADOW_RESEARCH_ONLY. No exchange execution surface is created here.

create extension if not exists pgcrypto;

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

create table if not exists public.brian_raw_captures (
  capture_id text primary key,
  provider text not null,
  record_type text not null,
  observed_at timestamptz not null,
  captured_at timestamptz not null,
  provenance_uri text,
  payload_hash text not null,
  payload jsonb not null,
  created_at timestamptz not null default now(),
  constraint brian_capture_time_order check (captured_at >= observed_at)
);

create table if not exists public.brian_intel_events (
  event_id text primary key,
  asset text not null,
  event_kind text not null,
  source_kind text not null,
  source_id text not null,
  published_at timestamptz,
  first_observed_at timestamptz not null,
  captured_at timestamptz not null,
  claim text not null,
  direction smallint not null default 0 check (direction between -1 and 1),
  magnitude double precision,
  trust_class text not null,
  entity_confidence double precision not null check (entity_confidence between 0 and 1),
  content_fingerprint text not null,
  corroboration_key text,
  provenance_uri text,
  pit_verified boolean not null default false,
  raw_capture_id text references public.brian_raw_captures(capture_id),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  constraint brian_event_capture_order check (captured_at >= first_observed_at)
);

create table if not exists public.brian_entities (
  entity_id text primary key,
  canonical_name text not null,
  entity_type text not null,
  first_observed_at timestamptz not null,
  created_at timestamptz not null default now()
);

create table if not exists public.brian_entity_labels (
  label_id text primary key,
  entity_id text not null references public.brian_entities(entity_id),
  provider text not null,
  label text not null,
  trust_class text not null,
  confidence double precision not null check (confidence between 0 and 1),
  observed_at timestamptz not null,
  valid_from timestamptz not null,
  valid_until timestamptz,
  provenance_uri text,
  content_hash text not null,
  created_at timestamptz not null default now(),
  constraint brian_label_not_hindsight check (valid_from <= observed_at),
  constraint brian_label_valid_window check (valid_until is null or valid_until >= valid_from)
);

create table if not exists public.brian_entity_edges (
  edge_id text primary key,
  src_entity_id text not null references public.brian_entities(entity_id),
  dst_entity_id text not null references public.brian_entities(entity_id),
  relation text not null,
  provider text not null,
  trust_class text not null,
  confidence double precision not null check (confidence between 0 and 1),
  observed_at timestamptz not null,
  valid_from timestamptz not null,
  provenance_uri text,
  created_at timestamptz not null default now(),
  constraint brian_edge_not_hindsight check (valid_from <= observed_at),
  constraint brian_edge_not_self check (src_entity_id <> dst_entity_id)
);

create table if not exists public.brian_whale_flows (
  flow_id text primary key,
  asset text not null,
  chain text,
  tx_hash text,
  from_entity_id text references public.brian_entities(entity_id),
  to_entity_id text references public.brian_entities(entity_id),
  flow_kind text not null,
  economic_direction smallint not null default 0 check (economic_direction between -1 and 1),
  amount_asset numeric,
  amount_usd numeric,
  label_confidence double precision not null default 0 check (label_confidence between 0 and 1),
  observed_at timestamptz not null,
  event_id text references public.brian_intel_events(event_id),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.brian_source_outcomes (
  outcome_id text primary key,
  source_id text not null,
  event_kind text not null,
  event_id text not null references public.brian_intel_events(event_id),
  resolved_at timestamptz not null,
  direction_correct boolean,
  latency_seconds double precision,
  manipulation_detected boolean not null default false,
  gross_return double precision,
  net_return double precision,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.brian_universe_snapshots (
  snapshot_id text primary key,
  provider text not null,
  observed_at timestamptz not null,
  eligible_count integer not null check (eligible_count >= 0),
  candidates jsonb not null,
  raw_capture_ids text[] not null default '{}',
  created_at timestamptz not null default now()
);

create table if not exists public.brian_opportunity_observations (
  observation_id text primary key,
  asset text not null,
  observed_at timestamptz not null,
  event_truth_score double precision not null check (event_truth_score between 0 and 1),
  manipulation_risk double precision not null check (manipulation_risk between 0 and 1),
  social_authenticity double precision check (social_authenticity between 0 and 1),
  smart_money_score double precision check (smart_money_score between -1 and 1),
  market_confirmation double precision check (market_confirmation between -1 and 1),
  priority_score double precision not null check (priority_score between 0 and 1),
  veto_reason text,
  source_event_ids text[] not null default '{}',
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists public.brian_opportunity_outcomes (
  outcome_id text primary key,
  observation_id text not null references public.brian_opportunity_observations(observation_id),
  resolved_at timestamptz not null,
  horizon_seconds integer not null check (horizon_seconds > 0),
  gross_return double precision,
  net_return double precision,
  mae double precision,
  mfe double precision,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create or replace function public.brian_validate_source_outcome_time()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
declare event_seen timestamptz;
begin
  select first_observed_at into event_seen from public.brian_intel_events where event_id = new.event_id;
  if event_seen is null then raise exception 'BRIAN_CAUSALITY: parent event not found'; end if;
  if new.resolved_at < event_seen then raise exception 'BRIAN_CAUSALITY: source outcome precedes parent event'; end if;
  return new;
end;
$$;

create or replace function public.brian_validate_opportunity_outcome_time()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
declare opportunity_seen timestamptz;
begin
  select observed_at into opportunity_seen from public.brian_opportunity_observations where observation_id = new.observation_id;
  if opportunity_seen is null then raise exception 'BRIAN_CAUSALITY: parent opportunity not found'; end if;
  if new.resolved_at < opportunity_seen then raise exception 'BRIAN_CAUSALITY: opportunity outcome precedes observation'; end if;
  return new;
end;
$$;

create index if not exists brian_raw_captures_provider_time_idx on public.brian_raw_captures(provider, observed_at desc);
create index if not exists brian_intel_events_asset_time_idx on public.brian_intel_events(asset, first_observed_at desc);
create index if not exists brian_intel_events_source_time_idx on public.brian_intel_events(source_id, first_observed_at desc);
create index if not exists brian_intel_events_kind_time_idx on public.brian_intel_events(event_kind, first_observed_at desc);
create index if not exists brian_entity_labels_entity_time_idx on public.brian_entity_labels(entity_id, observed_at desc);
create index if not exists brian_entity_edges_src_time_idx on public.brian_entity_edges(src_entity_id, observed_at desc);
create index if not exists brian_entity_edges_dst_time_idx on public.brian_entity_edges(dst_entity_id, observed_at desc);
create index if not exists brian_whale_flows_asset_time_idx on public.brian_whale_flows(asset, observed_at desc);
create index if not exists brian_source_outcomes_source_kind_time_idx on public.brian_source_outcomes(source_id, event_kind, resolved_at desc);
create index if not exists brian_universe_snapshots_time_idx on public.brian_universe_snapshots(observed_at desc);
create index if not exists brian_opportunity_obs_asset_time_idx on public.brian_opportunity_observations(asset, observed_at desc);
create index if not exists brian_opportunity_outcomes_resolved_idx on public.brian_opportunity_outcomes(resolved_at desc);

alter table public.brian_raw_captures enable row level security;
alter table public.brian_intel_events enable row level security;
alter table public.brian_entities enable row level security;
alter table public.brian_entity_labels enable row level security;
alter table public.brian_entity_edges enable row level security;
alter table public.brian_whale_flows enable row level security;
alter table public.brian_source_outcomes enable row level security;
alter table public.brian_universe_snapshots enable row level security;
alter table public.brian_opportunity_observations enable row level security;
alter table public.brian_opportunity_outcomes enable row level security;

revoke all on public.brian_raw_captures from anon, authenticated;
revoke all on public.brian_intel_events from anon, authenticated;
revoke all on public.brian_entities from anon, authenticated;
revoke all on public.brian_entity_labels from anon, authenticated;
revoke all on public.brian_entity_edges from anon, authenticated;
revoke all on public.brian_whale_flows from anon, authenticated;
revoke all on public.brian_source_outcomes from anon, authenticated;
revoke all on public.brian_universe_snapshots from anon, authenticated;
revoke all on public.brian_opportunity_observations from anon, authenticated;
revoke all on public.brian_opportunity_outcomes from anon, authenticated;

revoke update, delete, truncate, references, trigger on public.brian_raw_captures from service_role;
revoke update, delete, truncate, references, trigger on public.brian_intel_events from service_role;
revoke update, delete, truncate, references, trigger on public.brian_entities from service_role;
revoke update, delete, truncate, references, trigger on public.brian_entity_labels from service_role;
revoke update, delete, truncate, references, trigger on public.brian_entity_edges from service_role;
revoke update, delete, truncate, references, trigger on public.brian_whale_flows from service_role;
revoke update, delete, truncate, references, trigger on public.brian_source_outcomes from service_role;
revoke update, delete, truncate, references, trigger on public.brian_universe_snapshots from service_role;
revoke update, delete, truncate, references, trigger on public.brian_opportunity_observations from service_role;
revoke update, delete, truncate, references, trigger on public.brian_opportunity_outcomes from service_role;

grant select, insert on public.brian_raw_captures to service_role;
grant select, insert on public.brian_intel_events to service_role;
grant select, insert on public.brian_entities to service_role;
grant select, insert on public.brian_entity_labels to service_role;
grant select, insert on public.brian_entity_edges to service_role;
grant select, insert on public.brian_whale_flows to service_role;
grant select, insert on public.brian_source_outcomes to service_role;
grant select, insert on public.brian_universe_snapshots to service_role;
grant select, insert on public.brian_opportunity_observations to service_role;
grant select, insert on public.brian_opportunity_outcomes to service_role;

alter default privileges for role postgres in schema public revoke all on tables from service_role;
alter default privileges for role postgres in schema public grant select, insert on tables to service_role;

drop trigger if exists brian_source_outcomes_causal on public.brian_source_outcomes;
create trigger brian_source_outcomes_causal before insert on public.brian_source_outcomes for each row execute function public.brian_validate_source_outcome_time();
drop trigger if exists brian_opportunity_outcomes_causal on public.brian_opportunity_outcomes;
create trigger brian_opportunity_outcomes_causal before insert on public.brian_opportunity_outcomes for each row execute function public.brian_validate_opportunity_outcome_time();

do $$
declare t text;
begin
  foreach t in array array[
    'brian_raw_captures','brian_intel_events','brian_entities','brian_entity_labels','brian_entity_edges',
    'brian_whale_flows','brian_source_outcomes','brian_universe_snapshots','brian_opportunity_observations','brian_opportunity_outcomes'
  ] loop
    execute format('drop trigger if exists %I_append_only on public.%I', t, t);
    execute format('create trigger %I_append_only before update or delete on public.%I for each row execute function public.brian_reject_mutation()', t, t);
  end loop;
end;
$$;

-- Brian Evolution OS Layer 2: World Brain persistence.
-- GitHub-only preparation until explicit merge/deploy.
-- MAIN/ALPHA intelligence only; DIP is intentionally outside scope.
-- All rows are prospective, append-only, SHADOW ONLY and cannot execute trades.

create table if not exists public.brian_world_event_frames (
  frame_id text primary key,
  event_id text not null references public.brian_intel_events(event_id),
  observed_at timestamptz not null,
  published_at timestamptz,
  event_kind text not null,
  source_id text not null,
  claim text not null,
  primary_asset text,
  entity_ids text[] not null default '{}',
  narrative_ids text[] not null default '{}',
  direction_hint smallint not null default 0 check (direction_hint between -1 and 1),
  source_trust_class text not null,
  provenance_uri text,
  stage text not null default 'RESEARCHING' check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  evidence_refs text[] not null default '{}',
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  direct_alpha_influence boolean not null default false check (not direct_alpha_influence),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_world_entity_observations (
  observation_id text primary key,
  entity_id text not null,
  canonical_name text not null,
  entity_type text not null check (entity_type in (
    'ASSET','COMPANY','PERSON','COUNTRY','CENTRAL_BANK','REGULATOR','COMMODITY',
    'CURRENCY','SECTOR','TECHNOLOGY','PRODUCT','PROTOCOL','THEME'
  )),
  observed_at timestamptz not null,
  event_id text not null references public.brian_intel_events(event_id),
  match_kind text not null check (match_kind in ('ASSET_FIELD','ALIAS','PHRASE')),
  confidence double precision not null check (confidence between 0 and 1),
  provenance_uri text,
  evidence_refs text[] not null default '{}',
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  direct_alpha_influence boolean not null default false check (not direct_alpha_influence),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_world_relation_assertions (
  assertion_id text primary key,
  src_entity_id text not null,
  dst_entity_id text not null,
  relation text not null,
  observed_at timestamptz not null,
  event_id text not null references public.brian_intel_events(event_id),
  confidence double precision not null check (confidence between 0 and 1),
  mechanism text not null,
  stage text not null default 'RESEARCHING' check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  evidence_refs text[] not null default '{}',
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  direct_alpha_influence boolean not null default false check (not direct_alpha_influence),
  created_at timestamptz not null default now(),
  constraint brian_world_relation_no_self check (src_entity_id <> dst_entity_id)
);

create table if not exists public.brian_world_narrative_snapshots (
  snapshot_id text primary key,
  narrative_id text not null,
  label text not null,
  observed_at timestamptz not null,
  event_ids text[] not null default '{}',
  entity_ids text[] not null default '{}',
  strength double precision not null check (strength between 0 and 1),
  breadth integer not null check (breadth >= 0),
  direction_balance double precision not null check (direction_balance between -1 and 1),
  stage text not null default 'RESEARCHING' check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  evidence_refs text[] not null default '{}',
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  direct_alpha_influence boolean not null default false check (not direct_alpha_influence),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_world_future_events (
  future_event_id text primary key,
  event_kind text not null,
  scheduled_at timestamptz not null,
  first_observed_at timestamptz not null,
  title text not null,
  entity_ids text[] not null default '{}',
  asset_ids text[] not null default '{}',
  source_event_id text not null references public.brian_intel_events(event_id),
  confidence double precision not null check (confidence between 0 and 1),
  stage text not null default 'VERIFYING' check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  evidence_refs text[] not null default '{}',
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  direct_alpha_influence boolean not null default false check (not direct_alpha_influence),
  created_at timestamptz not null default now(),
  constraint brian_world_future_event_is_future_at_discovery check (scheduled_at > first_observed_at)
);

create table if not exists public.brian_world_causal_mechanisms (
  mechanism_id text primary key,
  narrative_id text not null,
  observed_at timestamptz not null,
  cause text not null,
  transmission jsonb not null,
  affected_assets jsonb not null,
  confidence double precision not null check (confidence between 0 and 1),
  stage text not null default 'RESEARCHING' check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  evidence_refs text[] not null default '{}',
  counter_evidence_required boolean not null default true check (counter_evidence_required),
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  direct_alpha_influence boolean not null default false check (not direct_alpha_influence),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_world_scenario_snapshots (
  scenario_id text primary key,
  mechanism_id text not null,
  observed_at timestamptz not null,
  branch text not null check (branch in ('MECHANISM_HOLDS','MECHANISM_BREAKS')),
  assumptions text[] not null default '{}',
  invalidators text[] not null default '{}',
  asset_impacts jsonb not null,
  confidence double precision not null check (confidence between 0 and 1),
  stage text not null default 'RESEARCHING' check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  evidence_refs text[] not null default '{}',
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  direct_alpha_influence boolean not null default false check (not direct_alpha_influence),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_world_asset_impact_candidates (
  impact_id text primary key,
  asset_id text not null,
  observed_at timestamptz not null,
  mechanism_id text not null,
  scenario_id text not null,
  conditional_direction smallint not null check (conditional_direction between -1 and 1),
  confidence double precision not null check (confidence between 0 and 1),
  rationale text not null,
  stage text not null default 'RESEARCHING' check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  evidence_refs text[] not null default '{}',
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  direct_alpha_influence boolean not null default false check (not direct_alpha_influence),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_world_brain_runs (
  run_id text primary key,
  started_at timestamptz not null,
  finished_at timestamptz not null,
  status text not null check (status in ('SUCCESS','DEGRADED','FAILED','SKIPPED_LEASE_CONTENDED')),
  input_events integer not null default 0 check (input_events >= 0),
  event_frames integer not null default 0 check (event_frames >= 0),
  entity_observations integer not null default 0 check (entity_observations >= 0),
  relation_assertions integer not null default 0 check (relation_assertions >= 0),
  narrative_snapshots integer not null default 0 check (narrative_snapshots >= 0),
  future_events integer not null default 0 check (future_events >= 0),
  causal_mechanisms integer not null default 0 check (causal_mechanisms >= 0),
  scenario_snapshots integer not null default 0 check (scenario_snapshots >= 0),
  asset_impacts integer not null default 0 check (asset_impacts >= 0),
  error_class text,
  error_message text,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  direct_alpha_influence boolean not null default false check (not direct_alpha_influence),
  created_at timestamptz not null default now(),
  constraint brian_world_brain_run_time_order check (finished_at >= started_at)
);

create index if not exists brian_world_event_frames_time_idx on public.brian_world_event_frames(observed_at desc);
create index if not exists brian_world_event_frames_asset_time_idx on public.brian_world_event_frames(primary_asset, observed_at desc);
create index if not exists brian_world_entity_observations_entity_time_idx on public.brian_world_entity_observations(entity_id, observed_at desc);
create index if not exists brian_world_relation_src_time_idx on public.brian_world_relation_assertions(src_entity_id, observed_at desc);
create index if not exists brian_world_relation_dst_time_idx on public.brian_world_relation_assertions(dst_entity_id, observed_at desc);
create index if not exists brian_world_narrative_time_idx on public.brian_world_narrative_snapshots(narrative_id, observed_at desc);
create index if not exists brian_world_future_events_time_idx on public.brian_world_future_events(scheduled_at asc);
create index if not exists brian_world_causal_time_idx on public.brian_world_causal_mechanisms(observed_at desc, narrative_id);
create index if not exists brian_world_scenario_time_idx on public.brian_world_scenario_snapshots(observed_at desc, mechanism_id);
create index if not exists brian_world_asset_impact_time_idx on public.brian_world_asset_impact_candidates(asset_id, observed_at desc);
create index if not exists brian_world_brain_run_time_idx on public.brian_world_brain_runs(started_at desc);

alter table public.brian_world_event_frames enable row level security;
alter table public.brian_world_entity_observations enable row level security;
alter table public.brian_world_relation_assertions enable row level security;
alter table public.brian_world_narrative_snapshots enable row level security;
alter table public.brian_world_future_events enable row level security;
alter table public.brian_world_causal_mechanisms enable row level security;
alter table public.brian_world_scenario_snapshots enable row level security;
alter table public.brian_world_asset_impact_candidates enable row level security;
alter table public.brian_world_brain_runs enable row level security;

revoke all on public.brian_world_event_frames from anon, authenticated;
revoke all on public.brian_world_entity_observations from anon, authenticated;
revoke all on public.brian_world_relation_assertions from anon, authenticated;
revoke all on public.brian_world_narrative_snapshots from anon, authenticated;
revoke all on public.brian_world_future_events from anon, authenticated;
revoke all on public.brian_world_causal_mechanisms from anon, authenticated;
revoke all on public.brian_world_scenario_snapshots from anon, authenticated;
revoke all on public.brian_world_asset_impact_candidates from anon, authenticated;
revoke all on public.brian_world_brain_runs from anon, authenticated;

revoke update, delete, truncate, references, trigger on public.brian_world_event_frames from service_role;
revoke update, delete, truncate, references, trigger on public.brian_world_entity_observations from service_role;
revoke update, delete, truncate, references, trigger on public.brian_world_relation_assertions from service_role;
revoke update, delete, truncate, references, trigger on public.brian_world_narrative_snapshots from service_role;
revoke update, delete, truncate, references, trigger on public.brian_world_future_events from service_role;
revoke update, delete, truncate, references, trigger on public.brian_world_causal_mechanisms from service_role;
revoke update, delete, truncate, references, trigger on public.brian_world_scenario_snapshots from service_role;
revoke update, delete, truncate, references, trigger on public.brian_world_asset_impact_candidates from service_role;
revoke update, delete, truncate, references, trigger on public.brian_world_brain_runs from service_role;

grant select, insert on public.brian_world_event_frames to service_role;
grant select, insert on public.brian_world_entity_observations to service_role;
grant select, insert on public.brian_world_relation_assertions to service_role;
grant select, insert on public.brian_world_narrative_snapshots to service_role;
grant select, insert on public.brian_world_future_events to service_role;
grant select, insert on public.brian_world_causal_mechanisms to service_role;
grant select, insert on public.brian_world_scenario_snapshots to service_role;
grant select, insert on public.brian_world_asset_impact_candidates to service_role;
grant select, insert on public.brian_world_brain_runs to service_role;

do $$
declare
  t text;
  trigger_name text;
begin
  foreach t in array array[
    'brian_world_event_frames','brian_world_entity_observations','brian_world_relation_assertions',
    'brian_world_narrative_snapshots','brian_world_future_events','brian_world_causal_mechanisms',
    'brian_world_scenario_snapshots','brian_world_asset_impact_candidates','brian_world_brain_runs'
  ] loop
    trigger_name := t || '_append_only';
    execute format('drop trigger if exists %I on public.%I', trigger_name, t);
    execute format(
      'create trigger %I before update or delete on public.%I for each row execute function public.brian_reject_mutation()',
      trigger_name,
      t
    );
  end loop;
end;
$$;

create or replace view public.brian_world_latest_narratives
with (security_invoker = true) as
select distinct on (narrative_id)
  snapshot_id, narrative_id, label, observed_at, event_ids, entity_ids, strength,
  breadth, direction_balance, stage, evidence_refs, evidence_class, shadow_only,
  live_execution, direct_alpha_influence, created_at
from public.brian_world_narrative_snapshots
order by narrative_id, observed_at desc, created_at desc;

create or replace view public.brian_world_upcoming_events
with (security_invoker = true) as
select future_event_id, event_kind, scheduled_at, first_observed_at, title,
  entity_ids, asset_ids, source_event_id, confidence, stage, evidence_refs,
  evidence_class, shadow_only, live_execution, direct_alpha_influence, created_at
from public.brian_world_future_events
where scheduled_at >= now()
order by scheduled_at asc;

revoke all on public.brian_world_latest_narratives from anon, authenticated;
revoke all on public.brian_world_upcoming_events from anon, authenticated;
grant select on public.brian_world_latest_narratives to service_role;
grant select on public.brian_world_upcoming_events to service_role;

-- Brian Evolution OS foundation.
-- GitHub preparation only until explicitly merged/deployed.
-- MAIN/ALPHA only; DIP is intentionally not referenced or controlled here.
-- PROSPECTIVE_EVOLUTION_SHADOW only. No exchange execution surface.

create table if not exists public.brian_evolution_events (
  event_id text primary key,
  entity_type text not null,
  entity_id text not null,
  event_type text not null,
  occurred_at timestamptz not null,
  stage text not null check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  title text not null,
  summary text not null,
  evidence_refs text[] not null default '{}',
  payload jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_evolution_capability_snapshots (
  snapshot_id text primary key,
  capability_id text not null,
  observed_at timestamptz not null,
  domain text not null check (domain in (
    'WORLD_SOURCE','MARKET_DATA','NEWS_MACRO','ENTITY_GRAPH','CROSS_ASSET','SENSOR',
    'MODEL','ALPHA','RESEARCH','CODEGEN','EXPERIMENT','PORTFOLIO','EXIT','OBSERVABILITY'
  )),
  name text not null,
  version text,
  stage text not null check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  health text not null check (health in ('HEALTHY','DEGRADED','STALE','MISSING','DISABLED')),
  description text not null,
  source_ids text[] not null default '{}',
  dependencies text[] not null default '{}',
  limitations text[] not null default '{}',
  evidence_refs text[] not null default '{}',
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_world_source_candidates (
  candidate_id text primary key,
  source_id text not null,
  discovered_at timestamptz not null,
  canonical_uri text not null,
  provider text not null,
  source_kind text not null,
  authority_class text not null check (authority_class in (
    'OFFICIAL_PRIMARY','INDEPENDENT_PROFESSIONAL','COMMUNITY','UNKNOWN'
  )),
  access_mode text not null check (access_mode in (
    'PUBLIC_NO_KEY','API_KEY_REQUIRED','LICENSED_REQUIRED','UNAVAILABLE'
  )),
  stage text not null check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  freshness_seconds integer check (freshness_seconds is null or freshness_seconds >= 0),
  corroboration_required boolean not null default true,
  manipulation_risk double precision not null check (manipulation_risk between 0 and 1),
  rationale text not null,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_evolution_hypothesis_snapshots (
  snapshot_id text primary key,
  hypothesis_id text not null,
  created_at_source timestamptz not null,
  observed_at timestamptz not null,
  title text not null,
  problem_statement text not null,
  proposed_mechanism text not null,
  target_capabilities text[] not null default '{}',
  evidence_refs text[] not null default '{}',
  counter_evidence_refs text[] not null default '{}',
  measurable_success_criteria text[] not null default '{}',
  stage text not null check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  uncertainty double precision not null check (uncertainty between 0 and 1),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_evolution_code_candidates (
  candidate_id text primary key,
  hypothesis_id text not null,
  proposed_at timestamptz not null,
  parent_commit text not null,
  changed_paths text[] not null default '{}',
  test_plan text[] not null default '{}',
  test_results jsonb not null default '{}'::jsonb,
  replay_results jsonb not null default '{}'::jsonb,
  stress_results jsonb not null default '{}'::jsonb,
  prospective_results jsonb not null default '{}'::jsonb,
  contamination_declaration text not null,
  stage text not null check (stage in (
    'DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE',
    'ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED'
  )),
  autonomous_apply_allowed boolean not null default false,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  constraint brian_evolution_candidate_requires_paths check (cardinality(changed_paths) > 0),
  constraint brian_evolution_candidate_no_auto_apply check (not autonomous_apply_allowed)
);

create index if not exists brian_evolution_events_entity_time_idx
  on public.brian_evolution_events(entity_type, entity_id, occurred_at desc);
create index if not exists brian_evolution_events_time_idx
  on public.brian_evolution_events(occurred_at desc);
create index if not exists brian_evolution_capability_time_idx
  on public.brian_evolution_capability_snapshots(capability_id, observed_at desc);
create index if not exists brian_world_source_time_idx
  on public.brian_world_source_candidates(source_id, discovered_at desc);
create index if not exists brian_evolution_hypothesis_time_idx
  on public.brian_evolution_hypothesis_snapshots(hypothesis_id, observed_at desc);
create index if not exists brian_evolution_code_candidate_time_idx
  on public.brian_evolution_code_candidates(proposed_at desc);

alter table public.brian_evolution_events enable row level security;
alter table public.brian_evolution_capability_snapshots enable row level security;
alter table public.brian_world_source_candidates enable row level security;
alter table public.brian_evolution_hypothesis_snapshots enable row level security;
alter table public.brian_evolution_code_candidates enable row level security;

revoke all on public.brian_evolution_events from anon, authenticated;
revoke all on public.brian_evolution_capability_snapshots from anon, authenticated;
revoke all on public.brian_world_source_candidates from anon, authenticated;
revoke all on public.brian_evolution_hypothesis_snapshots from anon, authenticated;
revoke all on public.brian_evolution_code_candidates from anon, authenticated;

revoke update, delete, truncate, references, trigger on public.brian_evolution_events from service_role;
revoke update, delete, truncate, references, trigger on public.brian_evolution_capability_snapshots from service_role;
revoke update, delete, truncate, references, trigger on public.brian_world_source_candidates from service_role;
revoke update, delete, truncate, references, trigger on public.brian_evolution_hypothesis_snapshots from service_role;
revoke update, delete, truncate, references, trigger on public.brian_evolution_code_candidates from service_role;

grant select, insert on public.brian_evolution_events to service_role;
grant select, insert on public.brian_evolution_capability_snapshots to service_role;
grant select, insert on public.brian_world_source_candidates to service_role;
grant select, insert on public.brian_evolution_hypothesis_snapshots to service_role;
grant select, insert on public.brian_evolution_code_candidates to service_role;

do $$
declare
  t text;
  trigger_name text;
begin
  foreach t in array array[
    'brian_evolution_events',
    'brian_evolution_capability_snapshots',
    'brian_world_source_candidates',
    'brian_evolution_hypothesis_snapshots',
    'brian_evolution_code_candidates'
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

create or replace view public.brian_evolution_latest_capabilities
with (security_invoker = true) as
select distinct on (capability_id)
  snapshot_id, capability_id, observed_at, domain, name, version, stage, health,
  description, source_ids, dependencies, limitations, evidence_refs, metadata,
  evidence_class, shadow_only, live_execution, created_at
from public.brian_evolution_capability_snapshots
order by capability_id, observed_at desc, created_at desc;

revoke all on public.brian_evolution_latest_capabilities from anon, authenticated;
grant select on public.brian_evolution_latest_capabilities to service_role;

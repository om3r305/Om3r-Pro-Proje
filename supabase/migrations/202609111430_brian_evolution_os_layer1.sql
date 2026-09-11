-- Brian Evolution OS Layer 1 persistence.
-- GitHub preparation only until explicit merge/deploy.
-- MAIN/ALPHA only. No DIP control/data/runtime surfaces.
-- Append-only prospective shadow evidence.

create table if not exists public.brian_evolution_gap_snapshots (
  snapshot_id text primary key,
  gap_id text not null,
  observed_at timestamptz not null,
  capability_id text not null,
  domain text not null,
  severity text not null check (severity in ('LOW','MEDIUM','HIGH','CRITICAL')),
  reason text not null,
  suggested_action text not null,
  evidence_refs text[] not null default '{}',
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_world_source_assessments (
  assessment_id text primary key,
  source_id text not null,
  assessed_at timestamptz not null,
  authority_score double precision not null check (authority_score between 0 and 1),
  freshness_score double precision not null check (freshness_score between 0 and 1),
  manipulation_penalty double precision not null check (manipulation_penalty between 0 and 1),
  corroboration_penalty double precision not null check (corroboration_penalty between 0 and 1),
  access_penalty double precision not null check (access_penalty between 0 and 1),
  trust_score double precision not null check (trust_score between 0 and 1),
  eligible_for_research boolean not null,
  eligible_for_decision_evidence boolean not null default false,
  reasons text[] not null default '{}',
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  constraint brian_source_assessment_no_direct_decision_without_active_evidence
    check (not live_execution)
);

create table if not exists public.brian_evolution_orchestrator_runs (
  run_id text primary key,
  started_at timestamptz not null,
  finished_at timestamptz not null,
  status text not null check (status in ('SUCCESS','DEGRADED','FAILED','SKIPPED_LEASE_CONTENDED')),
  capability_snapshots integer not null default 0 check (capability_snapshots >= 0),
  gap_snapshots integer not null default 0 check (gap_snapshots >= 0),
  source_candidates integer not null default 0 check (source_candidates >= 0),
  source_assessments integer not null default 0 check (source_assessments >= 0),
  journal_events integer not null default 0 check (journal_events >= 0),
  error_class text,
  error_message text,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class = 'PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  constraint brian_evolution_orchestrator_time_order check (finished_at >= started_at)
);

create index if not exists brian_evolution_gap_time_idx
  on public.brian_evolution_gap_snapshots(observed_at desc, severity);
create index if not exists brian_evolution_gap_capability_idx
  on public.brian_evolution_gap_snapshots(capability_id, observed_at desc);
create index if not exists brian_world_source_assessment_time_idx
  on public.brian_world_source_assessments(source_id, assessed_at desc);
create index if not exists brian_evolution_orchestrator_run_time_idx
  on public.brian_evolution_orchestrator_runs(started_at desc);

alter table public.brian_evolution_gap_snapshots enable row level security;
alter table public.brian_world_source_assessments enable row level security;
alter table public.brian_evolution_orchestrator_runs enable row level security;

revoke all on public.brian_evolution_gap_snapshots from anon, authenticated;
revoke all on public.brian_world_source_assessments from anon, authenticated;
revoke all on public.brian_evolution_orchestrator_runs from anon, authenticated;

revoke update, delete, truncate, references, trigger on public.brian_evolution_gap_snapshots from service_role;
revoke update, delete, truncate, references, trigger on public.brian_world_source_assessments from service_role;
revoke update, delete, truncate, references, trigger on public.brian_evolution_orchestrator_runs from service_role;

grant select, insert on public.brian_evolution_gap_snapshots to service_role;
grant select, insert on public.brian_world_source_assessments to service_role;
grant select, insert on public.brian_evolution_orchestrator_runs to service_role;

do $$
declare
  t text;
  trigger_name text;
begin
  foreach t in array array[
    'brian_evolution_gap_snapshots',
    'brian_world_source_assessments',
    'brian_evolution_orchestrator_runs'
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

create or replace view public.brian_evolution_latest_gaps
with (security_invoker = true) as
select distinct on (gap_id)
  snapshot_id, gap_id, observed_at, capability_id, domain, severity, reason,
  suggested_action, evidence_refs, metadata, evidence_class, shadow_only,
  live_execution, created_at
from public.brian_evolution_gap_snapshots
order by gap_id, observed_at desc, created_at desc;

revoke all on public.brian_evolution_latest_gaps from anon, authenticated;
grant select on public.brian_evolution_latest_gaps to service_role;

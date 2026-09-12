-- Brian Evolution OS Layer 3: bounded self-coding sandbox persistence.
-- GitHub-only until explicit rollout. MAIN/ALPHA research only; DIP is outside scope.
-- Append-only receipts. No canonical code apply and no live execution surface.

create table if not exists public.brian_evolution_codegen_requests (
  request_id text primary key,
  candidate_id text not null,
  hypothesis_id text not null,
  requested_at timestamptz not null,
  parent_commit text not null,
  branch_name text not null check (branch_name like 'evolution-candidate/%'),
  changed_paths text[] not null default '{}',
  objective text not null,
  constraints text[] not null default '{}',
  success_criteria text[] not null default '{}',
  evidence_refs text[] not null default '{}',
  contamination_declaration text not null,
  external_generator_required boolean not null default true check (external_generator_required),
  required_human_review boolean not null default true check (required_human_review),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  autonomous_apply_allowed boolean not null default false check (not autonomous_apply_allowed),
  created_at timestamptz not null default now(),
  constraint brian_evolution_codegen_request_paths check (cardinality(changed_paths) between 1 and 10)
);

create table if not exists public.brian_evolution_code_artifact_receipts (
  receipt_id text primary key,
  candidate_id text not null,
  hypothesis_id text not null,
  evidence_kind text not null check (evidence_kind in ('GENERATED','TYPECHECK','UNIT','REPLAY','STRESS','PROSPECTIVE')),
  observed_at timestamptz not null,
  passed boolean,
  artifact_sha256 text,
  parent_commit text,
  branch_name text,
  changed_paths text[] not null default '{}',
  patch_bytes integer check (patch_bytes is null or patch_bytes between 1 and 250000),
  generated_by text,
  generator_run_id text,
  provenance_complete boolean not null default false,
  protected_scope_clear boolean not null default false,
  leakage_detected boolean not null default false,
  evidence_refs text[] not null default '{}',
  payload jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  autonomous_apply_allowed boolean not null default false check (not autonomous_apply_allowed),
  created_at timestamptz not null default now()
);

create index if not exists brian_evolution_codegen_candidate_time_idx
  on public.brian_evolution_codegen_requests(candidate_id, requested_at desc);
create index if not exists brian_evolution_artifact_candidate_kind_time_idx
  on public.brian_evolution_code_artifact_receipts(candidate_id, evidence_kind, observed_at desc);

alter table public.brian_evolution_codegen_requests enable row level security;
alter table public.brian_evolution_code_artifact_receipts enable row level security;

revoke all on public.brian_evolution_codegen_requests from anon, authenticated;
revoke all on public.brian_evolution_code_artifact_receipts from anon, authenticated;
revoke update, delete, truncate, references, trigger on public.brian_evolution_codegen_requests from service_role;
revoke update, delete, truncate, references, trigger on public.brian_evolution_code_artifact_receipts from service_role;
grant select, insert on public.brian_evolution_codegen_requests to service_role;
grant select, insert on public.brian_evolution_code_artifact_receipts to service_role;

drop trigger if exists brian_evolution_codegen_requests_append_only on public.brian_evolution_codegen_requests;
create trigger brian_evolution_codegen_requests_append_only
before update or delete on public.brian_evolution_codegen_requests
for each row execute function public.brian_reject_mutation();

drop trigger if exists brian_evolution_code_artifact_receipts_append_only on public.brian_evolution_code_artifact_receipts;
create trigger brian_evolution_code_artifact_receipts_append_only
before update or delete on public.brian_evolution_code_artifact_receipts
for each row execute function public.brian_reject_mutation();

create or replace view public.brian_evolution_latest_code_artifact_receipts
with (security_invoker = true) as
select distinct on (candidate_id, evidence_kind)
  receipt_id, candidate_id, hypothesis_id, evidence_kind, observed_at, passed,
  artifact_sha256, parent_commit, branch_name, changed_paths, patch_bytes,
  generated_by, generator_run_id, provenance_complete, protected_scope_clear,
  leakage_detected, evidence_refs, payload, evidence_class, shadow_only,
  live_execution, autonomous_apply_allowed, created_at
from public.brian_evolution_code_artifact_receipts
order by candidate_id, evidence_kind, observed_at desc, created_at desc;

revoke all on public.brian_evolution_latest_code_artifact_receipts from anon, authenticated;
grant select on public.brian_evolution_latest_code_artifact_receipts to service_role;

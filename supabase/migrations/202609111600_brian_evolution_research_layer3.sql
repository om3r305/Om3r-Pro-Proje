-- Brian Evolution OS Layer 3: Hypothesis / Experiment / Drift / Promotion Lab.
-- GitHub-only until explicit rollout. DIP remains outside scope.
-- Append-only, prospective SHADOW evidence. No autonomous production apply.

create table if not exists public.brian_evolution_experiments (
  experiment_id text primary key,
  hypothesis_id text not null,
  created_at_source timestamptz not null,
  control_version text not null,
  challenger_version text not null,
  mode text not null check (mode in ('REPLAY','STRESS','PROSPECTIVE_SHADOW')),
  minimum_samples integer not null check (minimum_samples > 0),
  minimum_regimes integer not null check (minimum_regimes > 0),
  success_metrics text[] not null default '{}',
  hard_fail_conditions text[] not null default '{}',
  contamination_rules text[] not null default '{}',
  stage text not null check (stage in ('DISCOVERED','VERIFYING','RESEARCHING','EXPERIMENTAL','SHADOW_CANDIDATE','ACTIVE','DECAYING','REJECTED','RETIRED','ARCHIVED')),
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW' check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  autonomous_apply_allowed boolean not null default false check (not autonomous_apply_allowed),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_evolution_experiment_results (
  result_id text primary key,
  experiment_id text not null,
  measured_at timestamptz not null,
  role text not null check (role in ('CONTROL','CHALLENGER')),
  samples integer not null check (samples >= 0),
  regimes integer not null check (regimes >= 0),
  net_edge_bps double precision,
  gross_edge_bps double precision,
  max_drawdown_pct double precision,
  favorable_after_cost_rate double precision check (favorable_after_cost_rate is null or favorable_after_cost_rate between 0 and 1),
  turnover double precision,
  cost_bps double precision,
  leakage_detected boolean not null default false,
  data_quality_ok boolean not null default false,
  stability_score double precision check (stability_score is null or stability_score between 0 and 1),
  complexity_delta integer not null default 0,
  metric_payload jsonb not null default '{}'::jsonb,
  evidence_refs text[] not null default '{}',
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW' check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_evolution_drift_snapshots (
  drift_id text primary key,
  observed_at timestamptz not null,
  metric_id text not null,
  baseline double precision not null,
  recent double precision not null,
  absolute_delta double precision not null,
  relative_delta double precision,
  severity text not null check (severity in ('NONE','WATCH','MATERIAL','SEVERE')),
  direction text not null check (direction in ('UP','DOWN','FLAT')),
  evidence_refs text[] not null default '{}',
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW' check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_evolution_promotion_decisions (
  decision_id text primary key,
  experiment_id text not null,
  decided_at timestamptz not null,
  decision text not null check (decision in ('PROMOTE_CANDIDATE','KEEP_EXPERIMENTAL','REJECT')),
  score double precision not null check (score between 0 and 1),
  reasons text[] not null default '{}',
  required_next_stage text not null check (required_next_stage in ('EXPERIMENTAL','SHADOW_CANDIDATE','REJECTED')),
  control_result_id text,
  challenger_result_id text,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW' check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  autonomous_apply_allowed boolean not null default false check (not autonomous_apply_allowed),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_evolution_code_review_receipts (
  receipt_id text primary key,
  candidate_id text not null,
  reviewed_at timestamptz not null,
  protected_scope_clear boolean not null,
  tests_green boolean not null,
  replay_green boolean not null,
  stress_green boolean not null,
  prospective_green boolean not null,
  leakage_clear boolean not null,
  reviewer_kind text not null check (reviewer_kind in ('AUTOMATED_GUARD','PROMOTION_COUNCIL','HUMAN')),
  verdict text not null check (verdict in ('BLOCK','CONTINUE_TESTING','READY_FOR_HUMAN_REVIEW')),
  reasons text[] not null default '{}',
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW' check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  autonomous_apply_allowed boolean not null default false check (not autonomous_apply_allowed),
  created_at timestamptz not null default now()
);

create index if not exists brian_evolution_experiments_hypothesis_idx on public.brian_evolution_experiments(hypothesis_id, created_at desc);
create index if not exists brian_evolution_experiment_results_idx on public.brian_evolution_experiment_results(experiment_id, measured_at desc);
create index if not exists brian_evolution_drift_metric_idx on public.brian_evolution_drift_snapshots(metric_id, observed_at desc);
create index if not exists brian_evolution_promotion_experiment_idx on public.brian_evolution_promotion_decisions(experiment_id, decided_at desc);
create index if not exists brian_evolution_code_review_candidate_idx on public.brian_evolution_code_review_receipts(candidate_id, reviewed_at desc);

alter table public.brian_evolution_experiments enable row level security;
alter table public.brian_evolution_experiment_results enable row level security;
alter table public.brian_evolution_drift_snapshots enable row level security;
alter table public.brian_evolution_promotion_decisions enable row level security;
alter table public.brian_evolution_code_review_receipts enable row level security;

revoke all on public.brian_evolution_experiments from anon, authenticated;
revoke all on public.brian_evolution_experiment_results from anon, authenticated;
revoke all on public.brian_evolution_drift_snapshots from anon, authenticated;
revoke all on public.brian_evolution_promotion_decisions from anon, authenticated;
revoke all on public.brian_evolution_code_review_receipts from anon, authenticated;

revoke update, delete, truncate, references, trigger on public.brian_evolution_experiments from service_role;
revoke update, delete, truncate, references, trigger on public.brian_evolution_experiment_results from service_role;
revoke update, delete, truncate, references, trigger on public.brian_evolution_drift_snapshots from service_role;
revoke update, delete, truncate, references, trigger on public.brian_evolution_promotion_decisions from service_role;
revoke update, delete, truncate, references, trigger on public.brian_evolution_code_review_receipts from service_role;

grant select, insert on public.brian_evolution_experiments to service_role;
grant select, insert on public.brian_evolution_experiment_results to service_role;
grant select, insert on public.brian_evolution_drift_snapshots to service_role;
grant select, insert on public.brian_evolution_promotion_decisions to service_role;
grant select, insert on public.brian_evolution_code_review_receipts to service_role;

do $$
declare t text; trigger_name text;
begin
  foreach t in array array['brian_evolution_experiments','brian_evolution_experiment_results','brian_evolution_drift_snapshots','brian_evolution_promotion_decisions','brian_evolution_code_review_receipts'] loop
    trigger_name := t || '_append_only';
    execute format('drop trigger if exists %I on public.%I', trigger_name, t);
    execute format('create trigger %I before update or delete on public.%I for each row execute function public.brian_reject_mutation()', trigger_name, t);
  end loop;
end;
$$;

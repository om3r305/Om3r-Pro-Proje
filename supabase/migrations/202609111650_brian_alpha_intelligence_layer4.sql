-- Brian Evolution OS Layer 4: ALPHA expected-edge + prospective reliability challenger.
-- GitHub-only until explicit rollout. SHADOW ONLY. Canonical ALPHA is not mutated here.
-- DIP is a separate brain/treasury and is outside this layer.

create table if not exists public.brian_alpha_expected_edge_challenger (
  edge_id text primary key,
  decision_id text not null unique,
  observed_at timestamptz not null,
  evaluated_at timestamptz not null,
  asset_id text not null,
  canonical_action text not null check (canonical_action in ('OPEN_LONG','OPEN_SHORT')),
  direction smallint not null check (direction in (-1,1)),
  canonical_evidence_score double precision not null,
  support_groups text[] not null default '{}',
  source_observation_ids text[] not null default '{}',
  reliability_window_end timestamptz,
  reliability_generated_at timestamptz,
  expected_gross_move_bps double precision,
  estimated_round_trip_cost_bps double precision,
  uncertainty_penalty_bps double precision,
  event_decay_penalty_bps double precision,
  expected_net_edge_bps double precision,
  minimum_net_margin_bps double precision not null check (minimum_net_margin_bps >= 0),
  recommendation text not null check (recommendation in (
    'ALLOW_EDGE','DOWNGRADE_TO_WAIT','COST_UNAVAILABLE',
    'INSUFFICIENT_LAGGED_EVIDENCE','CONTAMINATED_EVIDENCE'
  )),
  eligible boolean not null default false,
  mature_group_count integer not null default 0 check (mature_group_count >= 0),
  group_contributions jsonb not null default '[]'::jsonb,
  reliability_weights jsonb not null default '{}'::jsonb,
  pit_clear boolean not null default false,
  reasons text[] not null default '{}',
  model_version text not null,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_EVOLUTION_SHADOW'
    check (evidence_class='PROSPECTIVE_EVOLUTION_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  canonical_mutation boolean not null default false check (not canonical_mutation),
  created_at timestamptz not null default now(),
  constraint brian_alpha_edge_time_order check (evaluated_at >= observed_at),
  constraint brian_alpha_edge_cost_nonnegative check (estimated_round_trip_cost_bps is null or estimated_round_trip_cost_bps >= 0),
  constraint brian_alpha_edge_eligible_requires_pit check (not eligible or pit_clear),
  constraint brian_alpha_edge_eligible_requires_net check (not eligible or expected_net_edge_bps is not null)
);

create index if not exists brian_alpha_edge_time_idx
  on public.brian_alpha_expected_edge_challenger(observed_at desc);
create index if not exists brian_alpha_edge_asset_time_idx
  on public.brian_alpha_expected_edge_challenger(asset_id, observed_at desc);
create index if not exists brian_alpha_edge_recommendation_time_idx
  on public.brian_alpha_expected_edge_challenger(recommendation, observed_at desc);

alter table public.brian_alpha_expected_edge_challenger enable row level security;
revoke all on public.brian_alpha_expected_edge_challenger from anon, authenticated;
revoke update, delete, truncate, references, trigger on public.brian_alpha_expected_edge_challenger from service_role;
grant select, insert on public.brian_alpha_expected_edge_challenger to service_role;

drop trigger if exists brian_alpha_expected_edge_challenger_append_only on public.brian_alpha_expected_edge_challenger;
create trigger brian_alpha_expected_edge_challenger_append_only
before update or delete on public.brian_alpha_expected_edge_challenger
for each row execute function public.brian_reject_mutation();

create or replace view public.brian_alpha_expected_edge_latest_by_asset
with (security_invoker = true) as
select distinct on (asset_id)
  edge_id, decision_id, observed_at, evaluated_at, asset_id, canonical_action,
  direction, canonical_evidence_score, support_groups, reliability_window_end,
  reliability_generated_at, expected_gross_move_bps, estimated_round_trip_cost_bps,
  uncertainty_penalty_bps, event_decay_penalty_bps, expected_net_edge_bps,
  minimum_net_margin_bps, recommendation, eligible, mature_group_count,
  group_contributions, reliability_weights, pit_clear, reasons, model_version,
  metadata, evidence_class, shadow_only, live_execution, canonical_mutation, created_at
from public.brian_alpha_expected_edge_challenger
order by asset_id, observed_at desc, created_at desc;

revoke all on public.brian_alpha_expected_edge_latest_by_asset from anon, authenticated;
grant select on public.brian_alpha_expected_edge_latest_by_asset to service_role;

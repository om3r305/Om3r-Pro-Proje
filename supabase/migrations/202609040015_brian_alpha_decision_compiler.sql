-- Brian ALPHA Decision Compiler v2 persistence.
-- SHADOW ONLY. This migration creates no exchange/order execution surface and does not modify
-- Phase 3.7 tables, checkpoint, thresholds, learning, or history.

create table if not exists public.brian_dynamic_cost_quotes (
  quote_id text primary key,
  compiler_version text not null,
  asset_id text not null,
  observed_at timestamptz not null,
  side text not null check (side in ('BUY','SELL')),
  requested_notional_usd numeric not null check (requested_notional_usd > 0),
  filled_notional_usd numeric not null check (filled_notional_usd >= 0),
  fill_ratio double precision not null check (fill_ratio between 0 and 1),
  fillable boolean not null,
  fee_bps double precision not null check (fee_bps >= 0),
  spread_bps double precision not null check (spread_bps >= 0),
  depth_slippage_bps double precision not null check (depth_slippage_bps >= 0),
  one_way_cost_bps double precision not null check (one_way_cost_bps >= 0),
  estimated_round_trip_cost_bps double precision not null check (estimated_round_trip_cost_bps >= 0),
  quality text not null check (quality in ('L2_OBSERVED','DEGRADED_TOP_OF_BOOK','UNAVAILABLE')),
  source_ids text[] not null default '{}',
  reason text not null,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW'
    check (evidence_class='PROSPECTIVE_DEVELOPMENT_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create table if not exists public.brian_alpha_decisions (
  decision_id text primary key,
  compiler_version text not null,
  observed_at timestamptz not null,
  asset_id text not null,
  action text not null check (action in ('OPEN_LONG','OPEN_SHORT','WAIT','VETO')),
  direction smallint not null check (direction between -1 and 1),
  evidence_score double precision not null check (evidence_score between 0 and 1),
  independent_group_count integer not null check (independent_group_count >= 0),
  support_groups text[] not null default '{}',
  conflict_groups text[] not null default '{}',
  source_observation_ids text[] not null default '{}',
  source_intrabar_event_ids text[] not null default '{}',
  source_cost_quote_id text references public.brian_dynamic_cost_quotes(quote_id),
  requested_virtual_notional_usd numeric not null check (requested_virtual_notional_usd >= 0),
  gross_edge_bps double precision,
  estimated_round_trip_cost_bps double precision,
  net_edge_bps double precision,
  veto_reason text,
  reason text not null,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW'
    check (evidence_class='PROSPECTIVE_DEVELOPMENT_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  constraint brian_alpha_action_direction check (
    (action='OPEN_LONG' and direction=1) or
    (action='OPEN_SHORT' and direction=-1) or
    (action in ('WAIT','VETO') and direction between -1 and 1)
  )
);

create table if not exists public.brian_alpha_decision_outcomes (
  outcome_id text primary key,
  decision_id text not null references public.brian_alpha_decisions(decision_id),
  asset_id text not null,
  horizon_seconds integer not null check (horizon_seconds in (300,900,3600)),
  observed_at timestamptz not null,
  resolved_at timestamptz not null,
  reference_price numeric not null check (reference_price > 0),
  resolved_price numeric not null check (resolved_price > 0),
  gross_return double precision not null,
  direction_adjusted_return double precision not null,
  mfe double precision,
  mae double precision,
  classification text not null,
  explanation text not null,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW'
    check (evidence_class='PROSPECTIVE_DEVELOPMENT_SHADOW'),
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  constraint brian_alpha_outcome_time_order check (resolved_at >= observed_at),
  unique(decision_id,horizon_seconds)
);

create index if not exists brian_dynamic_cost_asset_time_idx
  on public.brian_dynamic_cost_quotes(asset_id,observed_at desc);
create index if not exists brian_alpha_decisions_asset_time_idx
  on public.brian_alpha_decisions(asset_id,observed_at desc);
create index if not exists brian_alpha_decisions_action_time_idx
  on public.brian_alpha_decisions(action,observed_at desc);
create index if not exists brian_alpha_outcomes_decision_horizon_idx
  on public.brian_alpha_decision_outcomes(decision_id,horizon_seconds);
create index if not exists brian_alpha_outcomes_resolved_idx
  on public.brian_alpha_decision_outcomes(resolved_at desc);

alter table public.brian_dynamic_cost_quotes enable row level security;
alter table public.brian_alpha_decisions enable row level security;
alter table public.brian_alpha_decision_outcomes enable row level security;

revoke all on public.brian_dynamic_cost_quotes from anon,authenticated;
revoke all on public.brian_alpha_decisions from anon,authenticated;
revoke all on public.brian_alpha_decision_outcomes from anon,authenticated;

revoke update,delete,truncate,references,trigger on public.brian_dynamic_cost_quotes from service_role;
revoke update,delete,truncate,references,trigger on public.brian_alpha_decisions from service_role;
revoke update,delete,truncate,references,trigger on public.brian_alpha_decision_outcomes from service_role;

grant select,insert on public.brian_dynamic_cost_quotes to service_role;
grant select,insert on public.brian_alpha_decisions to service_role;
grant select,insert on public.brian_alpha_decision_outcomes to service_role;

drop trigger if exists brian_dynamic_cost_quotes_append_only on public.brian_dynamic_cost_quotes;
create trigger brian_dynamic_cost_quotes_append_only before update or delete on public.brian_dynamic_cost_quotes
  for each row execute function public.brian_reject_mutation();
drop trigger if exists brian_alpha_decisions_append_only on public.brian_alpha_decisions;
create trigger brian_alpha_decisions_append_only before update or delete on public.brian_alpha_decisions
  for each row execute function public.brian_reject_mutation();
drop trigger if exists brian_alpha_decision_outcomes_append_only on public.brian_alpha_decision_outcomes;
create trigger brian_alpha_decision_outcomes_append_only before update or delete on public.brian_alpha_decision_outcomes
  for each row execute function public.brian_reject_mutation();

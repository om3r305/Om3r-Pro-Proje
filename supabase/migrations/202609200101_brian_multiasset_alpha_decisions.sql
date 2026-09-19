create table if not exists public.brian_multiasset_alpha_decisions (
  decision_id text primary key,
  observed_at timestamptz not null,
  asset_id text not null,
  asset_class text not null,
  provider_time timestamptz not null,
  observed_reference_price numeric not null,
  action text not null,
  direction smallint not null check (direction in (-1,0,1)),
  evidence_score double precision not null,
  independent_group_count integer not null,
  support_groups text[] not null default '{}',
  conflict_groups text[] not null default '{}',
  linked_event_ids text[] not null default '{}',
  requested_virtual_notional_usd numeric not null default 0,
  estimated_round_trip_cost_bps double precision,
  reason text not null,
  veto_reason text,
  session_state text not null,
  data_latency_seconds double precision not null,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true,
  live_execution boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists brian_multiasset_alpha_asset_time_idx
  on public.brian_multiasset_alpha_decisions(asset_id, observed_at desc);

create or replace view public.brian_multiasset_alpha_latest as
select distinct on (asset_id) *
from public.brian_multiasset_alpha_decisions
order by asset_id, observed_at desc, created_at desc;

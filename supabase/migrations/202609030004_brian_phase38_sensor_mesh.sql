-- Brian Phase 3.8 Global Sensor Mesh persistence.
-- PROSPECTIVE_DEVELOPMENT_SHADOW only. No exchange execution surface.

create table if not exists public.brian_sensor_observations (
  observation_id text primary key,
  eye_id text not null,
  template_id text not null,
  asset_id text not null,
  market_domain text not null,
  sensor_family text not null,
  horizon text not null,
  independent_group text not null,
  observed_at timestamptz not null,
  direction smallint not null check (direction between -1 and 1),
  strength double precision not null check (strength between 0 and 1),
  confidence double precision not null check (confidence between 0 and 1),
  reliability double precision not null check (reliability between 0 and 1),
  available boolean not null,
  source_ids text[] not null default '{}',
  reason text not null,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  constraint brian_sensor_no_fabricated_direction check (available or direction = 0),
  constraint brian_sensor_available_has_source check ((not available) or cardinality(source_ids) > 0)
);

create table if not exists public.brian_micro_book_receipts (
  receipt_id text primary key,
  eye_id text not null,
  asset_id text not null,
  horizon text not null,
  observed_from timestamptz not null,
  observed_until timestamptz not null,
  starting_equity numeric not null check (starting_equity in (2,3,5,10,20)),
  ending_equity numeric not null check (ending_equity >= 0),
  net_pnl numeric not null,
  max_drawdown_pct double precision not null check (max_drawdown_pct between 0 and 100),
  turnover_notional numeric not null check (turnover_notional >= 0),
  trading_cost numeric not null check (trading_cost >= 0),
  active_decisions integer not null check (active_decisions >= 0),
  wins integer not null check (wins >= 0),
  losses integer not null check (losses >= 0),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  constraint brian_micro_book_time_order check (observed_until >= observed_from),
  constraint brian_micro_book_counts check (wins + losses <= active_decisions)
);

create table if not exists public.brian_opportunity_tournament_rounds (
  round_id text primary key,
  observed_at timestamptz not null,
  logical_eye_count integer not null check (logical_eye_count >= 0),
  candidate_count integer not null check (candidate_count >= 0),
  eligible_count integer not null check (eligible_count >= 0),
  virtual_allocated_usd numeric not null check (virtual_allocated_usd >= 0 and virtual_allocated_usd <= 500),
  virtual_unallocated_usd numeric not null check (virtual_unallocated_usd >= 0 and virtual_unallocated_usd <= 500),
  candidates jsonb not null,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  constraint brian_tournament_virtual_cap check (virtual_allocated_usd + virtual_unallocated_usd <= 500.00000001)
);

create table if not exists public.brian_missed_opportunity_receipts (
  receipt_id text primary key,
  asset_id text not null,
  horizon text not null,
  observed_at timestamptz not null,
  resolved_at timestamptz not null,
  opportunity_score double precision not null check (opportunity_score between 0 and 1),
  brian_action text not null check (brian_action in ('BUY','SELL','WAIT','OUT_OF_UNIVERSE')),
  hindsight_gross_return double precision,
  hindsight_net_return double precision,
  mfe double precision,
  mae double precision,
  classification text not null,
  explanation text not null,
  source_observation_ids text[] not null default '{}',
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  constraint brian_missed_opportunity_time_order check (resolved_at >= observed_at)
);

create index if not exists brian_sensor_obs_asset_time_idx on public.brian_sensor_observations(asset_id, observed_at desc);
create index if not exists brian_sensor_obs_eye_time_idx on public.brian_sensor_observations(eye_id, observed_at desc);
create index if not exists brian_sensor_obs_family_time_idx on public.brian_sensor_observations(sensor_family, observed_at desc);
create index if not exists brian_micro_book_eye_time_idx on public.brian_micro_book_receipts(eye_id, observed_until desc);
create index if not exists brian_tournament_time_idx on public.brian_opportunity_tournament_rounds(observed_at desc);
create index if not exists brian_missed_opportunity_time_idx on public.brian_missed_opportunity_receipts(resolved_at desc);

alter table public.brian_sensor_observations enable row level security;
alter table public.brian_micro_book_receipts enable row level security;
alter table public.brian_opportunity_tournament_rounds enable row level security;
alter table public.brian_missed_opportunity_receipts enable row level security;

revoke all on public.brian_sensor_observations from anon, authenticated;
revoke all on public.brian_micro_book_receipts from anon, authenticated;
revoke all on public.brian_opportunity_tournament_rounds from anon, authenticated;
revoke all on public.brian_missed_opportunity_receipts from anon, authenticated;

revoke update, delete, truncate, references, trigger on public.brian_sensor_observations from service_role;
revoke update, delete, truncate, references, trigger on public.brian_micro_book_receipts from service_role;
revoke update, delete, truncate, references, trigger on public.brian_opportunity_tournament_rounds from service_role;
revoke update, delete, truncate, references, trigger on public.brian_missed_opportunity_receipts from service_role;

grant select, insert on public.brian_sensor_observations to service_role;
grant select, insert on public.brian_micro_book_receipts to service_role;
grant select, insert on public.brian_opportunity_tournament_rounds to service_role;
grant select, insert on public.brian_missed_opportunity_receipts to service_role;

do $$
declare
  t text;
  trigger_name text;
begin
  foreach t in array array[
    'brian_sensor_observations',
    'brian_micro_book_receipts',
    'brian_opportunity_tournament_rounds',
    'brian_missed_opportunity_receipts'
  ] loop
    trigger_name := t || '_append_only';
    execute format('drop trigger if exists %I on public.%I', trigger_name, t);
    execute format('create trigger %I before update or delete on public.%I for each row execute function public.brian_reject_mutation()', trigger_name, t);
  end loop;
end;
$$;

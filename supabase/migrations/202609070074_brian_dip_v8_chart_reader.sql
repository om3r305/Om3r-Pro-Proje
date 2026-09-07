-- Brian DIP V8 Chart Reader evidence contract.
-- Dip Lab only. SHADOW ONLY. No Phase 3.7 / main cashbox mutation.

alter table public.brian_dip_foresight
  add column if not exists thesis_id text,
  add column if not exists setup text,
  add column if not exists regime text,
  add column if not exists venue text,
  add column if not exists target_price numeric,
  add column if not exists invalidation_price numeric,
  add column if not exists structural_invalidation_price numeric,
  add column if not exists raw_conviction double precision,
  add column if not exists calibrated_probability double precision,
  add column if not exists calibration_samples integer,
  add column if not exists resolver_version text,
  add column if not exists metric_version text,
  add column if not exists resolution_reason text,
  add column if not exists structure_fingerprint text;

update public.brian_dip_foresight
set metric_version = coalesce(metric_version, 'legacy-v7-direction-8m'),
    resolver_version = coalesce(resolver_version, 'legacy-v7-browser')
where metric_version is null or resolver_version is null;

create index if not exists brian_dip_foresight_v8_unresolved_idx
  on public.brian_dip_foresight (session_id, symbol, created_at)
  where resolved_at is null and metric_version = 'target-before-invalidation-v8';

create index if not exists brian_dip_foresight_v8_calibration_idx
  on public.brian_dip_foresight (setup, direction, venue, resolved_at desc)
  where resolved_at is not null and metric_version = 'target-before-invalidation-v8';

create table if not exists public.brian_dip_theses (
  row_id text primary key,
  thesis_id text not null,
  session_id text not null,
  symbol text not null,
  generated_at timestamptz not null default now(),
  setup text not null,
  direction text not null,
  thesis_state text not null,
  regime text not null,
  venue text not null,
  entry_low numeric,
  entry_high numeric,
  invalidation_price numeric,
  structural_invalidation_price numeric,
  target_price numeric,
  rr double precision,
  target_distance_bps double precision,
  cost_bps double precision,
  raw_conviction double precision,
  calibrated_probability double precision,
  calibration_samples integer not null default 0,
  why jsonb not null default '[]'::jsonb,
  veto jsonb not null default '[]'::jsonb,
  structure jsonb not null default '{}'::jsonb,
  flow jsonb not null default '{}'::jsonb,
  thesis_hash_input text not null,
  evidence_class text not null default 'AGGRESSIVE_DIP_V8_THESIS_SHADOW',
  shadow_only boolean not null default true check (shadow_only = true),
  live_execution boolean not null default false check (live_execution = false),
  created_at timestamptz not null default now()
);

create index if not exists brian_dip_theses_session_symbol_idx
  on public.brian_dip_theses (session_id, symbol, generated_at desc);

create index if not exists brian_dip_theses_thesis_id_idx
  on public.brian_dip_theses (thesis_id, generated_at desc);

comment on table public.brian_dip_theses is
  'Brian DIP V8 single-thesis evidence. Dip Lab only, shadow-only, no main Brian/Phase 3.7 mutation.';

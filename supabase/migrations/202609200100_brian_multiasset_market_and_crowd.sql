create table if not exists public.brian_multiasset_market_marks (
  mark_id text primary key,
  asset_id text not null,
  asset_class text not null,
  provider_symbol text not null,
  provider text not null,
  provider_quality text not null,
  observed_at timestamptz not null,
  provider_time timestamptz not null,
  price numeric not null,
  open_price numeric,
  high_price numeric,
  low_price numeric,
  previous_close numeric,
  volume numeric,
  return_5m double precision,
  return_1h double precision,
  session_state text not null,
  data_latency_seconds double precision not null,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true,
  live_execution boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists brian_multiasset_market_marks_asset_time_idx
  on public.brian_multiasset_market_marks(asset_id, provider_time desc);

create or replace view public.brian_multiasset_market_latest as
select distinct on (asset_id)
  mark_id,asset_id,asset_class,provider_symbol,provider,provider_quality,
  observed_at,provider_time,price,open_price,high_price,low_price,previous_close,volume,
  return_5m,return_1h,session_state,data_latency_seconds,metadata,
  evidence_class,shadow_only,live_execution,created_at
from public.brian_multiasset_market_marks
order by asset_id, provider_time desc, created_at desc;

create table if not exists public.brian_crowd_behavior_frames (
  frame_id text primary key,
  asset_id text not null,
  observed_at timestamptz not null,
  state text not null,
  direction smallint not null check (direction in (-1,0,1)),
  strength double precision not null,
  confidence double precision not null,
  supporting_groups text[] not null default '{}',
  conflict_groups text[] not null default '{}',
  source_observation_ids text[] not null default '{}',
  reason text not null,
  metadata jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true,
  live_execution boolean not null default false,
  created_at timestamptz not null default now()
);

create index if not exists brian_crowd_behavior_frames_asset_time_idx
  on public.brian_crowd_behavior_frames(asset_id, observed_at desc);

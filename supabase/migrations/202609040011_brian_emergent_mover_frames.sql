-- Brian 2026 Emergent Mover prospective observation store.
-- SHADOW_RESEARCH_ONLY. No exchange execution or promotion surface is created here.
--
-- A frame stores the current cross-sectional market state plus the comparison report used to
-- choose research-attention candidates. It deliberately does not store trade actions.

create table if not exists public.brian_emergent_mover_frames (
  frame_id text primary key,
  universe_snapshot_id text not null references public.brian_universe_snapshots(snapshot_id),
  provider text not null,
  observed_at timestamptz not null,
  baseline_observed_at timestamptz,
  comparison_age_ms bigint,
  comparable boolean not null,
  eligible_count integer not null check (eligible_count >= 0),
  dropped_symbol_count integer not null default 0 check (dropped_symbol_count >= 0),
  degraded_sources text[] not null default '{}',
  raw_capture_ids text[] not null default '{}',
  state jsonb not null,
  report jsonb not null,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW'
    check (evidence_class = 'PROSPECTIVE_DEVELOPMENT_SHADOW'),
  shadow_only boolean not null default true check (shadow_only = true),
  live_execution boolean not null default false check (live_execution = false),
  created_at timestamptz not null default now(),
  constraint brian_emergent_comparison_age_valid check (
    comparison_age_ms is null or comparison_age_ms > 0
  ),
  constraint brian_emergent_baseline_time_valid check (
    baseline_observed_at is null or baseline_observed_at < observed_at
  )
);

create index if not exists brian_emergent_mover_frames_time_idx
  on public.brian_emergent_mover_frames(observed_at desc);
create index if not exists brian_emergent_mover_frames_provider_time_idx
  on public.brian_emergent_mover_frames(provider, observed_at desc);

alter table public.brian_emergent_mover_frames enable row level security;

revoke all on public.brian_emergent_mover_frames from anon, authenticated;
revoke update, delete, truncate, references, trigger
  on public.brian_emergent_mover_frames from service_role;
grant select, insert on public.brian_emergent_mover_frames to service_role;

drop trigger if exists brian_emergent_mover_frames_append_only
  on public.brian_emergent_mover_frames;
create trigger brian_emergent_mover_frames_append_only
  before update or delete on public.brian_emergent_mover_frames
  for each row execute function public.brian_reject_mutation();

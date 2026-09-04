-- Brian 2026 real L2 capture ordering + raw-segment lineage.
-- SHADOW_RESEARCH_ONLY. No execution surface is created.
--
-- PR #37 created the normalized source-event table but deliberately left live collector wiring
-- for follow-up. This migration adds the information a real collector needs for deterministic
-- replay: one collector session id, one strict arrival sequence, transport connection generation,
-- per-symbol sync/resync generation, and an auditable pointer into compressed raw venue segments.

create table if not exists public.brian_l2_raw_segments (
  segment_id text primary key,
  raw_capture_id text not null unique references public.brian_raw_captures(capture_id),
  collector_session_id text not null,
  first_arrival_seq bigint not null check (first_arrival_seq > 0),
  last_arrival_seq bigint not null check (last_arrival_seq >= first_arrival_seq),
  message_count integer not null check (message_count > 0),
  observed_at timestamptz not null,
  captured_at timestamptz not null,
  compression text not null default 'gzip' check (compression = 'gzip'),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW'
    check (evidence_class = 'PROSPECTIVE_DEVELOPMENT_SHADOW'),
  shadow_only boolean not null default true check (shadow_only = true),
  live_execution boolean not null default false check (live_execution = false),
  created_at timestamptz not null default now(),
  constraint brian_l2_raw_segment_time_order check (captured_at >= observed_at),
  constraint brian_l2_raw_segment_contiguous_count check (
    message_count::bigint = last_arrival_seq - first_arrival_seq + 1
  ),
  constraint brian_l2_raw_segment_session_pair unique (segment_id, collector_session_id),
  constraint brian_l2_raw_segment_session_start unique (collector_session_id, first_arrival_seq)
);

create index if not exists brian_l2_raw_segments_session_range_idx
  on public.brian_l2_raw_segments(collector_session_id, first_arrival_seq, last_arrival_seq);

alter table public.brian_l2_source_events
  add column if not exists collector_session_id text,
  add column if not exists arrival_seq bigint,
  add column if not exists connection_generation integer,
  add column if not exists sync_generation integer,
  add column if not exists transport text,
  add column if not exists raw_segment_id text,
  add column if not exists raw_message_index integer;

-- There was intentionally no live writer before this migration. If somebody did write legacy L2
-- rows anyway, fail closed rather than inventing session/order/raw lineage after the fact.
do $$
begin
  if exists (
    select 1 from public.brian_l2_source_events
    where collector_session_id is null
       or arrival_seq is null
       or connection_generation is null
       or sync_generation is null
       or transport is null
       or raw_segment_id is null
       or raw_message_index is null
  ) then
    raise exception 'BRIAN_L2_LINEAGE: legacy source events without deterministic capture lineage exist';
  end if;
end
$$;

alter table public.brian_l2_source_events
  alter column collector_session_id set not null,
  alter column arrival_seq set not null,
  alter column connection_generation set not null,
  alter column sync_generation set not null,
  alter column transport set not null,
  alter column raw_segment_id set not null,
  alter column raw_message_index set not null;

do $$
begin
  if not exists (select 1 from pg_constraint where conname = 'brian_l2_arrival_seq_positive') then
    alter table public.brian_l2_source_events
      add constraint brian_l2_arrival_seq_positive check (arrival_seq > 0);
  end if;
  if not exists (select 1 from pg_constraint where conname = 'brian_l2_connection_generation_positive') then
    alter table public.brian_l2_source_events
      add constraint brian_l2_connection_generation_positive check (connection_generation > 0);
  end if;
  if not exists (select 1 from pg_constraint where conname = 'brian_l2_sync_generation_positive') then
    alter table public.brian_l2_source_events
      add constraint brian_l2_sync_generation_positive check (sync_generation > 0);
  end if;
  if not exists (select 1 from pg_constraint where conname = 'brian_l2_raw_message_index_nonnegative') then
    alter table public.brian_l2_source_events
      add constraint brian_l2_raw_message_index_nonnegative check (raw_message_index >= 0);
  end if;
  if not exists (select 1 from pg_constraint where conname = 'brian_l2_transport_kind_valid') then
    alter table public.brian_l2_source_events
      add constraint brian_l2_transport_kind_valid check (
        (kind = 'depth_diff' and transport = 'binance_spot_diff_depth_ws') or
        (kind = 'depth_snapshot' and transport = 'binance_spot_rest_depth_snapshot') or
        (kind = 'top_of_book' and transport = 'binance_spot_book_ticker_ws')
      );
  end if;
  if not exists (select 1 from pg_constraint where conname = 'brian_l2_source_event_raw_segment_fk') then
    alter table public.brian_l2_source_events
      add constraint brian_l2_source_event_raw_segment_fk
      foreign key (raw_segment_id, collector_session_id)
      references public.brian_l2_raw_segments(segment_id, collector_session_id);
  end if;
end
$$;

create unique index if not exists brian_l2_source_events_session_arrival_uidx
  on public.brian_l2_source_events(collector_session_id, arrival_seq);
create unique index if not exists brian_l2_source_events_raw_message_uidx
  on public.brian_l2_source_events(raw_segment_id, raw_message_index);
create index if not exists brian_l2_source_events_replay_idx
  on public.brian_l2_source_events(collector_session_id, symbol, sync_generation, arrival_seq);

create table if not exists public.brian_l2_capture_session_events (
  session_event_id text primary key,
  collector_session_id text not null,
  event_kind text not null check (event_kind in (
    'STARTED','WS_CONNECTED','SYNC_STARTED','SNAPSHOT_TOO_OLD','SYNCED',
    'GAP_INVALIDATED','RESYNC_STARTED','WS_DISCONNECTED','STOPPED','FAILED'
  )),
  venue text not null,
  symbol text,
  connection_generation integer not null check (connection_generation > 0),
  sync_generation integer check (sync_generation is null or sync_generation > 0),
  arrival_seq_boundary bigint check (arrival_seq_boundary is null or arrival_seq_boundary >= 0),
  observed_at timestamptz not null,
  details jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW'
    check (evidence_class = 'PROSPECTIVE_DEVELOPMENT_SHADOW'),
  shadow_only boolean not null default true check (shadow_only = true),
  live_execution boolean not null default false check (live_execution = false),
  created_at timestamptz not null default now()
);

create index if not exists brian_l2_capture_session_events_replay_idx
  on public.brian_l2_capture_session_events(collector_session_id, observed_at, created_at);

alter table public.brian_l2_raw_segments enable row level security;
alter table public.brian_l2_capture_session_events enable row level security;

revoke all on public.brian_l2_raw_segments from anon, authenticated;
revoke all on public.brian_l2_capture_session_events from anon, authenticated;
revoke update, delete, truncate, references, trigger on public.brian_l2_raw_segments from service_role;
revoke update, delete, truncate, references, trigger on public.brian_l2_capture_session_events from service_role;
grant select, insert on public.brian_l2_raw_segments to service_role;
grant select, insert on public.brian_l2_capture_session_events to service_role;

-- Reassert source-event privileges after ALTER TABLE for clarity/fail-closed reviewability.
revoke update, delete, truncate, references, trigger on public.brian_l2_source_events from service_role;
grant select, insert on public.brian_l2_source_events to service_role;

drop trigger if exists brian_l2_raw_segments_append_only on public.brian_l2_raw_segments;
create trigger brian_l2_raw_segments_append_only
  before update or delete on public.brian_l2_raw_segments
  for each row execute function public.brian_reject_mutation();

drop trigger if exists brian_l2_capture_session_events_append_only on public.brian_l2_capture_session_events;
create trigger brian_l2_capture_session_events_append_only
  before update or delete on public.brian_l2_capture_session_events
  for each row execute function public.brian_reject_mutation();

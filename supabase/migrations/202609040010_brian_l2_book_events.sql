-- Brian Reflex/L2 Event-State Foundation (issue #32, task after Item 1). Observation
-- infrastructure only: an append-only, prospectively timestamped store for normalized
-- Level-2/best-book events (see supabase/functions/_shared/l2_book.ts for the normalization
-- layer and the reducer that rebuilds current state from these rows). This migration adds NO
-- OBI/OFI signal, NO micro-price predictor, NO decision threshold, and NO order placement --
-- it only captures normalized book state and the raw identifiers needed to later detect
-- duplicates, out-of-order updates, sequence gaps, and staleness. Nothing in brian-2026 writes to
-- this table yet; wiring a live collector to a real venue feed is explicit follow-up work, not
-- part of this PR. brian-live-shadow / Phase 3.7 is untouched by this file.
--
-- PROSPECTIVE_DEVELOPMENT_SHADOW only, following the same evidence_class/shadow_only/
-- live_execution convention as every other Phase 3.8+ table (e.g.
-- 202609030004_brian_phase38_sensor_mesh.sql). Mutable operational-state tables are the
-- documented exception to this append-only doctrine (see
-- 202609030009_brian_collector_lease.sql); this table is not one of those -- every row here is a
-- point-in-time observation and is never updated after insert.
--
-- This file assumes public.brian_reject_mutation() already exists from
-- 202609030001_brian_intelligence_memory.sql, matching every other non-standalone Phase
-- migration in this chain (only 202609030009 redefines it, because that migration was
-- specifically built to also be applied standalone against a vanilla Postgres CI database).

create table if not exists public.brian_l2_book_events (
  event_id text primary key,
  venue text not null,
  symbol text not null,
  -- Exchange-reported event time, when the venue provides one; null otherwise (per spec, "when
  -- available" -- not every venue message carries its own timestamp).
  exchange_event_at timestamptz,
  -- When the collector received the raw message off the wire.
  collector_received_at timestamptz not null,
  -- When normalization ran and produced this row.
  ingest_at timestamptz not null,
  -- Opaque venue update/sequence id as text (ids can exceed bigint/safe-integer range for some
  -- venues); null when the venue does not provide one for this message.
  source_sequence text,
  best_bid numeric not null check (best_bid > 0),
  best_ask numeric not null check (best_ask > 0),
  mid_price numeric not null check (mid_price > 0),
  spread numeric not null check (spread >= 0),
  -- Top-N levels, best-first, as [{price, size}, ...]. N itself is recorded in depth_n below --
  -- explicit/configurable per capture, never a trading threshold.
  bids jsonb not null default '[]'::jsonb,
  asks jsonb not null default '[]'::jsonb,
  depth_n integer not null check (depth_n >= 0),
  -- Raw fields worth preserving for audit/lineage (which venue, which raw ids) without keeping
  -- the entire raw payload.
  source_lineage jsonb not null default '{}'::jsonb,
  -- ingest_at - (exchange_event_at ?? collector_received_at), in milliseconds, clamped to >= 0.
  freshness_ms integer not null check (freshness_ms >= 0),
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now(),
  -- Same invariant public.l2_book.ts's validateBookEvent enforces before a row is ever built:
  -- a crossed/inverted book (best_bid >= best_ask) must never be recorded as current state.
  constraint brian_l2_book_no_inversion check (best_bid < best_ask)
);

create index if not exists brian_l2_book_venue_symbol_time_idx
  on public.brian_l2_book_events(venue, symbol, ingest_at desc);
-- Supports efficiently finding the prior event for a given venue+symbol when reconstructing
-- sequence-gap/duplicate/out-of-order history from the stored table (the in-memory reducer in
-- l2_book.ts does this directly over an already-fetched ordered array; this index is for a
-- future consumer that needs to do the equivalent lookup against the table itself).
create index if not exists brian_l2_book_sequence_idx
  on public.brian_l2_book_events(venue, symbol, source_sequence);

alter table public.brian_l2_book_events enable row level security;
revoke all on public.brian_l2_book_events from anon, authenticated;
revoke update, delete, truncate, references, trigger on public.brian_l2_book_events from service_role;
grant select, insert on public.brian_l2_book_events to service_role;

drop trigger if exists brian_l2_book_events_append_only on public.brian_l2_book_events;
create trigger brian_l2_book_events_append_only
  before update or delete on public.brian_l2_book_events
  for each row execute function public.brian_reject_mutation();

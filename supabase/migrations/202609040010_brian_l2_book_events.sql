-- Brian Reflex/L2 Event-State Foundation (issue #32). Observation infrastructure only.
--
-- Revised per GPT-5.6 Sol's review on PR #37: this table now persists normalized SOURCE EVENTS
-- only (what a venue actually sent -- a top-of-book tick, a depth snapshot, or a depth diff),
-- never derived BOOK STATE (best bid/ask, mid, spread, or a reconstructed depth book). Those are
-- a function of a source-event *stream*, computed by the pure reducers in
-- supabase/functions/_shared/l2_book.ts (reduceTopOfBookEvents / reconstructDepthBook), and are
-- not persisted here. The first version of this migration stored a single flattened
-- best_bid/best_ask/mid_price/spread shape that conflated a top-of-book tick with a full L2 event
-- and does not correspond to what any real venue message actually contains (a depth diff, in
-- particular, has no best bid/ask or complete top-N of its own).
--
-- `kind` discriminates which of the three normalized event shapes `payload` holds, mirroring
-- l2_book.ts's `BookSourceEvent` union exactly:
--   'top_of_book'    -> { updateId, bestBid: {price,size}, bestAsk: {price,size} }
--   'depth_snapshot' -> { lastUpdateId, bids: [{price,size}...], asks: [{price,size}...] }
--   'depth_diff'     -> { firstUpdateId, finalUpdateId, bidMutations: [...], askMutations: [...] }
-- Prices/sizes/update-ids are stored exactly as the venue sent them (decimal/integer strings) --
-- this table never coerces them to a numeric type, for the same reason l2_book.ts never parses
-- them to `number`: a float column could silently round a value the venue sent exactly.
--
-- This migration adds NO OBI/OFI signal, NO micro-price predictor, NO decision threshold, and NO
-- order placement. Nothing in brian-2026 writes to this table yet; wiring a live collector to a
-- real venue feed is explicit follow-up work. brian-live-shadow / Phase 3.7 is untouched.
--
-- PROSPECTIVE_DEVELOPMENT_SHADOW only, following the same evidence_class/shadow_only/
-- live_execution convention as every other Phase 3.8+ table. Purely append-only -- every row is a
-- point-in-time source event and is never updated after insert (no mutable-operational-state
-- exception here, unlike 202609030009_brian_collector_lease.sql's lease table).
--
-- This file assumes public.brian_reject_mutation() already exists from
-- 202609030001_brian_intelligence_memory.sql, matching every other non-standalone Phase
-- migration in this chain.

drop table if exists public.brian_l2_book_events;

create table if not exists public.brian_l2_source_events (
  event_id text primary key,
  kind text not null check (kind in ('top_of_book', 'depth_snapshot', 'depth_diff')),
  venue text not null,
  symbol text not null,
  -- Exchange-reported event time, when the venue's message carries one (e.g. a Binance depth
  -- diff has one; a Binance bookTicker does not) -- null otherwise.
  exchange_event_at timestamptz,
  -- When the collector received the raw message off the wire.
  collector_received_at timestamptz not null,
  -- When normalization ran and produced this row.
  ingest_at timestamptz not null,
  -- ingest_at - (exchange_event_at ?? collector_received_at), clamped to >= 0: "how stale is
  -- this", for freshness gating.
  age_ms integer not null check (age_ms >= 0),
  -- The same delta, NOT clamped: a negative value is real, observable clock skew (or a
  -- reordered/backdated message) that age_ms alone would hide.
  clock_skew_ms integer not null,
  -- Kind-specific normalized fields -- see the kind-by-kind shapes documented above. Every
  -- decimal/integer value inside is stored exactly as the venue sent it (a string), never
  -- coerced to a numeric column.
  payload jsonb not null,
  -- Raw fields worth preserving for audit/lineage (which venue, which raw ids) without keeping
  -- the entire raw payload.
  source_lineage jsonb not null default '{}'::jsonb,
  evidence_class text not null default 'PROSPECTIVE_DEVELOPMENT_SHADOW',
  shadow_only boolean not null default true check (shadow_only),
  live_execution boolean not null default false check (not live_execution),
  created_at timestamptz not null default now()
);

create index if not exists brian_l2_source_events_venue_symbol_time_idx
  on public.brian_l2_source_events(venue, symbol, ingest_at desc);
-- Supports efficiently replaying a venue+symbol's ordered event stream through the pure reducers
-- (reduceTopOfBookEvents / reconstructDepthBook), which do the actual state reconstruction.
create index if not exists brian_l2_source_events_kind_idx
  on public.brian_l2_source_events(venue, symbol, kind, ingest_at desc);

alter table public.brian_l2_source_events enable row level security;
revoke all on public.brian_l2_source_events from anon, authenticated;
revoke update, delete, truncate, references, trigger on public.brian_l2_source_events from service_role;
grant select, insert on public.brian_l2_source_events to service_role;

drop trigger if exists brian_l2_source_events_append_only on public.brian_l2_source_events;
create trigger brian_l2_source_events_append_only
  before update or delete on public.brian_l2_source_events
  for each row execute function public.brian_reject_mutation();

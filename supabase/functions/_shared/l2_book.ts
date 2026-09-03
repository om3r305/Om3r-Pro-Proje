// Reflex/L2 event-state foundation (brian-2026 issue #32, task after Item 1). Observation
// infrastructure only: a normalized Level-2/best-book event contract, one primary-venue
// normalizer plus a second, structurally distinct venue adapter to prove the boundary is real
// (not one venue's field names baked into the contract), and a minimal pure reducer that rebuilds
// current best-book state from an ordered event stream while flagging duplicates, out-of-order
// arrivals, sequence gaps, staleness, and bid/ask inversion.
//
// This module contains NO OBI/OFI signal, NO micro-price predictor, NO decision threshold, and NO
// order placement -- it only captures and normalizes state. Nothing in brian-2026 calls this module
// yet; wiring a live collector to a real venue feed, and any alpha built on top of this state, are
// explicitly out of scope for this foundation and are follow-up work. No Phase 3.7 file is touched
// or imported here.
//
// Every function here is pure: no Deno.serve, no network fetch, no Supabase client, no
// Date.now()/crypto calls that would make output depend on wall-clock time. Callers (a future live
// collector) supply collectorReceivedAt/ingestAt explicitly, which is what makes the reducer's
// "deterministic rebuild from a fixture stream" property possible -- the same event array always
// reduces to the same state, in tests and in production alike.

export const EVIDENCE_CLASS = "PROSPECTIVE_DEVELOPMENT_SHADOW" as const;
export const DEFAULT_DEPTH_N = 5;
export const DEFAULT_STALE_AFTER_MS = 5_000;

export const BINANCE_VENUE = "binance";
export const COINBASE_VENUE = "coinbase";

function finite(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

// ---------------------------------------------------------------------------------------------
// Normalized contract
// ---------------------------------------------------------------------------------------------

export interface DepthLevel {
  price: number;
  size: number;
}

export interface NormalizedBookEvent {
  venue: string;
  symbol: string;
  /** Exchange-reported event time (ISO-8601), when the venue provides one; null otherwise. */
  exchangeEventAt: string | null;
  /** When the collector received the raw message off the wire (ISO-8601). */
  collectorReceivedAt: string;
  /** When normalization ran and produced this event (ISO-8601). */
  ingestAt: string;
  /** Opaque venue update/sequence id, as a string (ids can exceed safe-integer range); null if the venue does not provide one for this message. */
  sourceSequence: string | null;
  bestBid: number;
  bestAsk: number;
  midPrice: number;
  spread: number;
  /** Top-N levels, best-first. May be fewer than depthN if the raw message did not carry that many. */
  bids: DepthLevel[];
  asks: DepthLevel[];
  /** The N this event's bids/asks were capped to -- explicit, configurable, not a trading threshold. */
  depthN: number;
  /** Raw fields worth preserving for audit/lineage (which venue, which raw ids) without keeping the entire raw payload. */
  sourceLineage: Record<string, unknown>;
  /** ingestAt - (exchangeEventAt ?? collectorReceivedAt), in milliseconds, clamped to >= 0. */
  freshnessMs: number;
  evidenceClass: typeof EVIDENCE_CLASS;
  shadowOnly: true;
}

export interface NormalizeContext {
  collectorReceivedAt: string;
  ingestAt: string;
  /** Defaults to DEFAULT_DEPTH_N. Explicit/configurable per call, never a trading threshold. */
  depthN?: number;
}

export interface BookValidationResult {
  valid: boolean;
  reason?: string;
}

/** Rejects a non-positive or crossed/inverted book (best_bid >= best_ask). This is the one
 * hard invariant every venue adapter (and the reducer, as defense-in-depth) enforces before an
 * event is ever treated as current state. */
export function validateBookEvent(event: Pick<NormalizedBookEvent, "bestBid" | "bestAsk">): BookValidationResult {
  if (!(event.bestBid > 0) || !(event.bestAsk > 0)) {
    return { valid: false, reason: "non-positive best_bid/best_ask" };
  }
  if (event.bestBid >= event.bestAsk) {
    return { valid: false, reason: "bid/ask inversion: best_bid >= best_ask" };
  }
  return { valid: true };
}

function freshnessMs(exchangeEventAt: string | null, collectorReceivedAt: string, ingestAt: string): number {
  const reference = exchangeEventAt ?? collectorReceivedAt;
  const ms = Date.parse(ingestAt) - Date.parse(reference);
  return Number.isFinite(ms) ? Math.max(0, ms) : 0;
}

export type NormalizeResult =
  | { ok: true; event: NormalizedBookEvent }
  | { ok: false; reason: string };

export interface VenueBookAdapter<TRaw> {
  venue: string;
  normalize(raw: TRaw, context: NormalizeContext): NormalizeResult;
}

// ---------------------------------------------------------------------------------------------
// Primary venue adapter: Binance best-bookTicker (+ optional depth levels on the same message).
// ---------------------------------------------------------------------------------------------

export interface BinanceRawBookTicker {
  /** Update id, when present (Binance bookTicker/depth streams both expose one). */
  u?: number | string;
  /** Symbol, e.g. "BTCUSDT". */
  s: string;
  /** Best bid price/qty as strings, matching Binance's wire format. */
  b: string;
  B: string;
  /** Best ask price/qty as strings. */
  a: string;
  A: string;
  /** Event time in epoch ms, present on stream messages but not on the plain REST bookTicker snapshot. */
  E?: number;
  /** Optional depth levels beyond best, as [price, qty] string pairs -- present only when the
   * collector also captured a depth snapshot/diff for this symbol at this instant. */
  bids?: [string, string][];
  asks?: [string, string][];
}

export const binanceBookTickerAdapter: VenueBookAdapter<BinanceRawBookTicker> = {
  venue: BINANCE_VENUE,
  normalize(raw, context): NormalizeResult {
    const bestBid = finite(raw.b);
    const bestAsk = finite(raw.a);
    const validation = validateBookEvent({ bestBid, bestAsk });
    if (!validation.valid) return { ok: false, reason: validation.reason! };

    const depthN = context.depthN ?? DEFAULT_DEPTH_N;
    const bids = raw.bids?.length
      ? raw.bids.slice(0, depthN).map(([price, size]) => ({ price: finite(price), size: finite(size) }))
      : [{ price: bestBid, size: finite(raw.B) }];
    const asks = raw.asks?.length
      ? raw.asks.slice(0, depthN).map(([price, size]) => ({ price: finite(price), size: finite(size) }))
      : [{ price: bestAsk, size: finite(raw.A) }];
    const exchangeEventAt = raw.E != null ? new Date(raw.E).toISOString() : null;

    return {
      ok: true,
      event: {
        venue: BINANCE_VENUE,
        symbol: raw.s,
        exchangeEventAt,
        collectorReceivedAt: context.collectorReceivedAt,
        ingestAt: context.ingestAt,
        sourceSequence: raw.u != null ? String(raw.u) : null,
        bestBid,
        bestAsk,
        midPrice: (bestBid + bestAsk) / 2,
        spread: bestAsk - bestBid,
        bids,
        asks,
        depthN,
        sourceLineage: { venue: BINANCE_VENUE, updateId: raw.u ?? null, symbol: raw.s },
        freshnessMs: freshnessMs(exchangeEventAt, context.collectorReceivedAt, context.ingestAt),
        evidenceClass: EVIDENCE_CLASS,
        shadowOnly: true,
      },
    };
  },
};

// ---------------------------------------------------------------------------------------------
// Second venue adapter: proves the adapter boundary is a real seam, not a Binance-shaped
// contract with a second name on it. Field names/shape below are a simplified, illustrative
// second-venue message -- deliberately structurally different from Binance's (nested price/size
// pairs, a top-level ISO timestamp, a `sequence` field instead of `u`) -- not a byte-for-byte copy
// of any single real exchange's current wire format; wiring an actual second live venue is a
// follow-up task and must be verified against that venue's real API docs at that time.
// ---------------------------------------------------------------------------------------------

export interface SecondVenueRawTicker {
  product: string;
  time: string;
  bestBid: { price: string; size: string };
  bestAsk: { price: string; size: string };
  depth?: { bids: { price: string; size: string }[]; asks: { price: string; size: string }[] };
  sequence?: number | string;
}

export const secondVenueTickerAdapter: VenueBookAdapter<SecondVenueRawTicker> = {
  venue: COINBASE_VENUE,
  normalize(raw, context): NormalizeResult {
    const bestBid = finite(raw.bestBid.price);
    const bestAsk = finite(raw.bestAsk.price);
    const validation = validateBookEvent({ bestBid, bestAsk });
    if (!validation.valid) return { ok: false, reason: validation.reason! };

    const depthN = context.depthN ?? DEFAULT_DEPTH_N;
    const bids = raw.depth?.bids.length
      ? raw.depth.bids.slice(0, depthN).map((l) => ({ price: finite(l.price), size: finite(l.size) }))
      : [{ price: bestBid, size: finite(raw.bestBid.size) }];
    const asks = raw.depth?.asks.length
      ? raw.depth.asks.slice(0, depthN).map((l) => ({ price: finite(l.price), size: finite(l.size) }))
      : [{ price: bestAsk, size: finite(raw.bestAsk.size) }];

    return {
      ok: true,
      event: {
        venue: COINBASE_VENUE,
        symbol: raw.product,
        exchangeEventAt: raw.time ?? null,
        collectorReceivedAt: context.collectorReceivedAt,
        ingestAt: context.ingestAt,
        sourceSequence: raw.sequence != null ? String(raw.sequence) : null,
        bestBid,
        bestAsk,
        midPrice: (bestBid + bestAsk) / 2,
        spread: bestAsk - bestBid,
        bids,
        asks,
        depthN,
        sourceLineage: { venue: COINBASE_VENUE, product: raw.product, sequence: raw.sequence ?? null },
        freshnessMs: freshnessMs(raw.time ?? null, context.collectorReceivedAt, context.ingestAt),
        evidenceClass: EVIDENCE_CLASS,
        shadowOnly: true,
      },
    };
  },
};

// ---------------------------------------------------------------------------------------------
// Sequence and freshness classification
// ---------------------------------------------------------------------------------------------

export type SequenceClassification = "no_sequence" | "expected" | "duplicate" | "out_of_order" | "gap";

function parseSequence(sourceSequence: string | null): bigint | null {
  if (sourceSequence === null) return null;
  try {
    return BigInt(sourceSequence);
  } catch {
    return null;
  }
}

/** Compares a candidate sequence id against the last one this reducer applied for the same
 * (venue, symbol). `previous`/`current` are already-parsed bigints (or null when the venue didn't
 * supply one). Pure and total: every (previous, current) pair maps to exactly one classification. */
export function classifySequence(previous: bigint | null, current: bigint | null): SequenceClassification {
  if (current === null) return "no_sequence";
  if (previous === null) return "expected";
  if (current === previous) return "duplicate";
  if (current < previous) return "out_of_order";
  if (current === previous + 1n) return "expected";
  return "gap";
}

export type FreshnessClassification = "FRESH" | "STALE";

export function classifyFreshness(freshnessMsValue: number, staleAfterMs: number = DEFAULT_STALE_AFTER_MS): FreshnessClassification {
  return freshnessMsValue > staleAfterMs ? "STALE" : "FRESH";
}

// ---------------------------------------------------------------------------------------------
// Reducer: rebuild current best-book state from an ordered event stream.
// ---------------------------------------------------------------------------------------------

export interface BookState {
  venue: string;
  symbol: string;
  /** The latest event actually applied to reach this state. */
  event: NormalizedBookEvent;
  lastSequence: bigint | null;
  freshness: FreshnessClassification;
  /** Count of events applied (not ignored/rejected) to reach this state, including the current one. */
  appliedCount: number;
}

export interface BookStateIssue {
  venue: string;
  symbol: string;
  kind: "invalid" | "duplicate" | "out_of_order" | "gap";
  reason: string;
  event: NormalizedBookEvent;
}

export interface ReduceResult {
  /** Keyed by `${venue}:${symbol}` -- see keyOf. */
  states: Map<string, BookState>;
  issues: BookStateIssue[];
}

export function keyOf(venue: string, symbol: string): string {
  return `${venue}:${symbol}`;
}

/**
 * Folds an ordered array of normalized events into current per-(venue,symbol) best-book state.
 * "Ordered" means arrival/processing order, not necessarily monotonically increasing venue
 * sequence numbers -- detecting when it is NOT monotonic is exactly this function's job.
 *
 * Per event:
 *   - bid/ask inversion (defense-in-depth; adapters should already have rejected this) -> issue
 *     kind "invalid", state unchanged.
 *   - duplicate sourceSequence (matches the last applied sequence for this venue+symbol) -> issue
 *     kind "duplicate", state unchanged: a replay must never silently re-apply.
 *   - out-of-order sourceSequence (behind the last applied sequence) -> issue kind "out_of_order",
 *     state unchanged: a straggler must never regress state that a newer event already advanced.
 *   - sequence gap (jumps ahead by more than one) -> issue kind "gap", but the event IS still
 *     applied -- state must keep moving forward on incomplete history rather than freeze forever
 *     waiting for updates that may never arrive; the gap is recorded for reconciliation instead.
 *   - no sequence id at all -> always applied (best-effort, arrival-order only), and does not
 *     reset the sequence baseline: a later sequenced event is still compared against the last
 *     sequence this reducer actually saw.
 *   - freshness (FRESH/STALE) is computed and stored on every applied state regardless of the
 *     above, since staleness is a property of an individual event, not of sequencing.
 */
export function reduceBookEvents(events: NormalizedBookEvent[], options?: { staleAfterMs?: number }): ReduceResult {
  const staleAfterMs = options?.staleAfterMs ?? DEFAULT_STALE_AFTER_MS;
  const states = new Map<string, BookState>();
  const issues: BookStateIssue[] = [];

  for (const event of events) {
    const validation = validateBookEvent(event);
    if (!validation.valid) {
      issues.push({ venue: event.venue, symbol: event.symbol, kind: "invalid", reason: validation.reason!, event });
      continue;
    }

    const key = keyOf(event.venue, event.symbol);
    const prior = states.get(key) ?? null;
    const currentSeq = parseSequence(event.sourceSequence);
    const classification = classifySequence(prior?.lastSequence ?? null, currentSeq);

    if (classification === "duplicate") {
      issues.push({
        venue: event.venue,
        symbol: event.symbol,
        kind: "duplicate",
        reason: `duplicate sourceSequence ${event.sourceSequence}`,
        event,
      });
      continue;
    }
    if (classification === "out_of_order") {
      issues.push({
        venue: event.venue,
        symbol: event.symbol,
        kind: "out_of_order",
        reason: `sourceSequence ${event.sourceSequence} is behind last applied sequence ${prior?.lastSequence ?? "null"}`,
        event,
      });
      continue;
    }
    if (classification === "gap") {
      issues.push({
        venue: event.venue,
        symbol: event.symbol,
        kind: "gap",
        reason: `sequence gap for ${key}: expected ${(prior!.lastSequence as bigint) + 1n}, got ${currentSeq}`,
        event,
      });
    }

    states.set(key, {
      venue: event.venue,
      symbol: event.symbol,
      event,
      lastSequence: currentSeq ?? prior?.lastSequence ?? null,
      freshness: classifyFreshness(event.freshnessMs, staleAfterMs),
      appliedCount: (prior?.appliedCount ?? 0) + 1,
    });
  }

  return { states, issues };
}

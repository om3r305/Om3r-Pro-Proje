// Reflex/L2 event-state foundation (brian-2026 issue #32). Observation infrastructure only.
//
// This is a semantic correction of the first version of this module, per GPT-5.6 Sol's review on
// PR #37 (cross-checked against official Binance Spot and Coinbase Exchange WebSocket docs). The
// first version conflated several distinct concepts into one generic "NormalizedBookEvent" and
// got Binance's bookTicker.u semantics wrong. This version fixes that by:
//
//   1. Separating normalized SOURCE EVENTS (what a venue actually sent, kind-discriminated: a
//      top-of-book tick, a depth snapshot, or a depth diff) from derived BOOK STATE (the
//      reconstructed current best-book / depth-book, which is a *function of* a source event
//      stream, not a property of any single message).
//   2. Matching real venue wire shapes exactly. Binance's bookTicker payload is only
//      {u,s,b,B,a,A} -- no event time, no depth. Its diff-depth payload is a different message:
//      {e,E,s,U,u,b,a}. Its REST/partial-depth snapshot is a third shape: {lastUpdateId,bids,asks}.
//      Synthesizing depth fields into a bookTicker-shaped event (as the first version did) does
//      not correspond to anything a real venue sends.
//   3. Not treating Binance bookTicker.u as a +1 sequence. bookTicker is emitted on best-price/
//      qty change, so consecutive messages are not required to be consecutive update ids -- large
//      jumps are normal, not data loss. Only equal-or-behind values are meaningful (duplicate/
//      stale), never "not exactly +1".
//   4. Implementing Binance's actual documented local-order-book procedure for the depth path,
//      corrected a second time per GPT-5.6 Sol's follow-up review: the per-diff update rule is
//      the inequality Binance actually documents, not a "normally equal" observation --
//      (a) ignore (discard, no state change) a diff whose finalUpdateId (u) is at or behind the
//      locally-applied update id; (b) a diff whose firstUpdateId (U) is *strictly greater* than
//      localUpdateId + 1 is a real gap -- something was missed -- and invalidates the book;
//      (c) anything else, including a diff that overlaps already-applied history but still
//      extends past it, applies normally. This same single rule (`tryApplyDiff`) is used both for
//      steady-state updates and to replay the buffered diffs after a startup snapshot, so there is
//      one source of truth for what "valid" means, not two subtly different formulas. A real gap
//      fails closed: the reconstructed book is marked INVALID and must not be exposed as current
//      state until a fresh snapshot resyncs it -- Binance's own "discard the local book and
//      restart" instruction, not "flag and keep applying". Startup buffering (fetch a snapshot,
//      discard buffered diffs it already covers, replay the rest) is its own explicit pure
//      primitive, `synchronizeDepthBookStartup`, so this module never claims to buffer
//      pre-snapshot diffs inside the plain ordered-stream reducer when it actually only discards
//      them there.
//   5. Preserving exact decimal price/size strings end to end. Prices/sizes are never parsed to
//      JS `number` in the source-event or book-state layers (a `number` cannot represent every
//      decimal string exactly and silently rounds); comparisons and zero-quantity deletion use
//      exact scaled-BigInt decimal arithmetic instead. Malformed decimal/integer/timestamp fields
//      fail the whole normalize() call closed (an explicit `{ ok: false, reason }`) rather than
//      being coerced to 0/null and silently accepted.
//   6. Keeping non-negative "age" (how stale is this, for freshness gating) and signed
//      "clock skew" (ingestAt minus the reference timestamp, which can be negative) as two
//      separate fields, instead of collapsing both into one clamped-to-zero number that would
//      hide a skewed or reordered clock as if the data were simply fresh.
//   7. Not naming the illustrative second-venue adapter after a real exchange. It normalizes a
//      deliberately synthetic, structurally-different raw shape solely to prove the adapter
//      boundary is a real seam; it is exported under an explicitly fictional venue id
//      (FIXTURE_VENUE) so it can never be mistaken for real Coinbase (or any other real venue)
//      lineage. A real second-venue adapter is follow-up work, built from that venue's own
//      official docs at that time.
//
// Every function here remains pure: no Deno.serve, no network fetch, no Supabase client, and no
// Date.now() -- every timestamp is supplied by the caller, which is what makes the reducers'
// fixture-stream reconstructions deterministic. Nothing in brian-2026 calls this module yet. No
// OBI/OFI signal, no micro-price predictor, no decision threshold, no order placement.

export const EVIDENCE_CLASS = "PROSPECTIVE_DEVELOPMENT_SHADOW" as const;
export const BINANCE_VENUE = "binance";
// Deliberately not a real exchange name -- see file header point 7. Never treat this venue id as
// live market data or wire it to any real order/execution surface.
export const FIXTURE_VENUE = "fixture-illustrative-venue";

// -------------------------------------------------------------------------------------------
// Exact-decimal helpers. Every price/size the venue sends is a decimal string; we keep it that
// way through the event/state boundary and only ever compare/subtract/detect-zero via
// scaled-BigInt integer arithmetic, never via `Number(...)`, which cannot represent every decimal
// exactly and would silently reintroduce the rounding this module exists to avoid.
// -------------------------------------------------------------------------------------------

const DECIMAL_STRING_RE = /^\d+(\.\d+)?$/;
const INTEGER_STRING_RE = /^\d+$/;

export function isValidDecimalString(value: unknown): value is string {
  return typeof value === "string" && DECIMAL_STRING_RE.test(value);
}

export function isValidIntegerString(value: unknown): value is string {
  return typeof value === "string" && INTEGER_STRING_RE.test(value);
}

/**
 * Validates a venue update/sequence id (Binance's u/U/lastUpdateId are int64) before it is ever
 * stringified. A string is accepted whenever it matches the integer format -- strings are
 * lossless regardless of magnitude. A `number` is accepted only when it is a safe integer
 * (<= Number.MAX_SAFE_INTEGER): a real int64 id larger than that, if it ever arrived as a JS
 * `number` (e.g. via naive JSON.parse upstream), may already have been silently rounded before
 * this code ever sees it -- stringifying that rounded value would fabricate a precise-looking but
 * wrong id instead of failing closed. Reject it instead of laundering the precision loss.
 */
export function isSafeIntegerIdInput(value: unknown): value is string | number {
  if (typeof value === "string") return isValidIntegerString(value);
  if (typeof value === "number") return Number.isInteger(value) && value >= 0 && value <= Number.MAX_SAFE_INTEGER;
  return false;
}

/** The largest epoch-millisecond value JavaScript's Date can represent (the documented ECMA-262
 * limit: ±100,000,000 days from the epoch). Number.MAX_SAFE_INTEGER is far larger than this, so a
 * positive safe integer alone is not sufficient -- a value beyond this bound produces an Invalid
 * Date, and calling `.toISOString()` on one throws RangeError instead of failing closed. */
const MAX_REPRESENTABLE_DATE_EPOCH_MILLIS = 8_640_000_000_000_000;

export function isValidEpochMillis(value: unknown): value is number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value <= 0) return false;
  if (value > MAX_REPRESENTABLE_DATE_EPOCH_MILLIS) return false;
  return !Number.isNaN(new Date(value).getTime());
}

export function isValidIsoTimestamp(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 && Number.isFinite(Date.parse(value));
}

function scaledBigInt(value: string): { scaled: bigint; scale: number } {
  const [intPart, fracPart = ""] = value.split(".");
  return { scaled: BigInt((intPart || "0") + fracPart), scale: fracPart.length };
}

/** Exact decimal comparison via integer arithmetic -- never loses precision the way
 * `Number(a) - Number(b)` could for high-precision venue decimals. */
export function compareDecimalStrings(a: string, b: string): -1 | 0 | 1 {
  const da = scaledBigInt(a);
  const db = scaledBigInt(b);
  const scale = Math.max(da.scale, db.scale);
  const av = da.scaled * 10n ** BigInt(scale - da.scale);
  const bv = db.scaled * 10n ** BigInt(scale - db.scale);
  if (av < bv) return -1;
  if (av > bv) return 1;
  return 0;
}

export function isZeroQuantity(value: string): boolean {
  return compareDecimalStrings(value, "0") === 0;
}

/** Canonicalizes a decimal string for use as a price-level map key, so "25.10" and "25.1" are
 * treated as the same level (as real venues do), without ever going through a lossy float. */
export function canonicalDecimalKey(value: string): string {
  let { scaled, scale } = scaledBigInt(value);
  while (scale > 0 && scaled % 10n === 0n) {
    scaled /= 10n;
    scale -= 1;
  }
  return `${scaled.toString()}e-${scale}`;
}

function sortedLevels(levels: Map<string, DecimalQuote>, direction: "desc" | "asc"): DecimalQuote[] {
  const arr = [...levels.values()];
  arr.sort((a, b) => (direction === "desc" ? -1 : 1) * compareDecimalStrings(a.price, b.price));
  return arr;
}

// -------------------------------------------------------------------------------------------
// Freshness / clock skew
// -------------------------------------------------------------------------------------------

export interface Freshness {
  /** ingestAt - referenceAt, clamped to >= 0. "How stale is this", for freshness gating. */
  ageMs: number;
  /** ingestAt - referenceAt, NOT clamped. Negative means the reference timestamp is ahead of
   * ingestAt (clock skew, or a reordered/backdated message) -- a real, observable data-quality
   * fact that clamping to 0 would silently hide. */
  clockSkewMs: number;
}

export function computeFreshness(referenceAt: string, ingestAt: string): Freshness {
  const raw = Date.parse(ingestAt) - Date.parse(referenceAt);
  return { ageMs: Math.max(0, raw), clockSkewMs: raw };
}

// -------------------------------------------------------------------------------------------
// Source events: what a venue actually sent, normalized but not yet reduced into book state.
// -------------------------------------------------------------------------------------------

export interface DecimalQuote {
  price: string;
  size: string;
}

export interface SourceEventMeta {
  venue: string;
  symbol: string;
  /** Exchange-reported event time (ISO-8601), when the venue's message carries one; null for
   * message types that don't (e.g. Binance bookTicker has no event time). */
  exchangeEventAt: string | null;
  /** When the collector received the raw message off the wire (ISO-8601). */
  collectorReceivedAt: string;
  /** When normalization ran and produced this event (ISO-8601). */
  ingestAt: string;
  ageMs: number;
  clockSkewMs: number;
  sourceLineage: Record<string, unknown>;
  evidenceClass: typeof EVIDENCE_CLASS;
  shadowOnly: true;
}

/** A top-of-book tick (e.g. Binance bookTicker). Carries only the best bid/ask -- no depth, no
 * claim about intervening order-book updates the venue didn't emit a tick for. */
export interface TopOfBookEvent extends SourceEventMeta {
  kind: "top_of_book";
  /** Venue update id, when the venue provides one (e.g. Binance's `u`). This is monotonic
   * evidence only -- NOT a +1 sequence -- see file header point 3. */
  updateId: string | null;
  bestBid: DecimalQuote;
  bestAsk: DecimalQuote;
}

/** A full or partial depth snapshot (e.g. Binance REST /api/v3/depth or a partial-depth stream). */
export interface DepthSnapshotEvent extends SourceEventMeta {
  kind: "depth_snapshot";
  lastUpdateId: string;
  bids: DecimalQuote[];
  asks: DecimalQuote[];
}

/** An incremental depth diff (e.g. Binance's <symbol>@depth stream). Carries only the levels that
 * changed -- it does NOT carry a complete book or a best bid/ask, and must never be treated as
 * one. */
export interface DepthDiffEvent extends SourceEventMeta {
  kind: "depth_diff";
  /** Binance `U`: first update id covered by this event. */
  firstUpdateId: string;
  /** Binance `u`: final (last) update id covered by this event. */
  finalUpdateId: string;
  bidMutations: DecimalQuote[];
  askMutations: DecimalQuote[];
}

export type BookSourceEvent = TopOfBookEvent | DepthSnapshotEvent | DepthDiffEvent;

export type NormalizeResult<TEvent> =
  | { ok: true; event: TEvent }
  | { ok: false; reason: string };

export interface NormalizeContext {
  collectorReceivedAt: string;
  ingestAt: string;
  /** Required only by adapters whose raw payload does not self-report a symbol (e.g. Binance's
   * REST depth snapshot response, which omits it -- the caller supplied it in the request URL). */
  symbol?: string;
}

export interface VenueBookAdapter<TRaw, TEvent extends SourceEventMeta = SourceEventMeta> {
  venue: string;
  normalize(raw: TRaw, context: NormalizeContext): NormalizeResult<TEvent>;
}

function validateContextTimestamps(context: NormalizeContext): string | null {
  if (!isValidIsoTimestamp(context.collectorReceivedAt)) return `invalid collectorReceivedAt: ${JSON.stringify(context.collectorReceivedAt)}`;
  if (!isValidIsoTimestamp(context.ingestAt)) return `invalid ingestAt: ${JSON.stringify(context.ingestAt)}`;
  return null;
}

function parseLevels(raw: unknown): { ok: true; levels: DecimalQuote[] } | { ok: false; reason: string } {
  if (!Array.isArray(raw)) return { ok: false, reason: `levels must be an array, got ${JSON.stringify(raw)}` };
  const levels: DecimalQuote[] = [];
  for (const entry of raw) {
    if (!Array.isArray(entry) || entry.length < 2) return { ok: false, reason: `malformed level entry: ${JSON.stringify(entry)}` };
    const [price, size] = entry;
    if (!isValidDecimalString(price)) return { ok: false, reason: `invalid level price: ${JSON.stringify(price)}` };
    if (!isValidDecimalString(size)) return { ok: false, reason: `invalid level size: ${JSON.stringify(size)}` };
    levels.push({ price, size });
  }
  return { ok: true, levels };
}

// -------------------------------------------------------------------------------------------
// Binance adapters -- one per real, distinct message shape. None of these invent fields the
// official payload does not have, and none synthesize depth into a top-of-book message or vice
// versa.
// -------------------------------------------------------------------------------------------

/** Official Binance Spot <symbol>@bookTicker payload. Exactly {u,s,b,B,a,A} -- no event time, no
 * depth arrays. (docs: "Pushes any update to the best bid or ask's price or quantity in real-time
 * for a specified symbol.") */
export interface BinanceRawBookTicker {
  u: number | string;
  s: string;
  b: string;
  B: string;
  a: string;
  A: string;
}

export const binanceBookTickerAdapter: VenueBookAdapter<BinanceRawBookTicker, TopOfBookEvent> = {
  venue: BINANCE_VENUE,
  normalize(raw, context): NormalizeResult<TopOfBookEvent> {
    const tsError = validateContextTimestamps(context);
    if (tsError) return { ok: false, reason: tsError };
    if (typeof raw.s !== "string" || raw.s.length === 0) return { ok: false, reason: "missing symbol (s)" };
    if (!isSafeIntegerIdInput(raw.u)) return { ok: false, reason: `invalid or unsafe updateId (u): ${JSON.stringify(raw.u)}` };
    for (const [field, value] of [["b", raw.b], ["B", raw.B], ["a", raw.a], ["A", raw.A]] as const) {
      if (!isValidDecimalString(value)) return { ok: false, reason: `invalid decimal for ${field}: ${JSON.stringify(value)}` };
    }
    if (compareDecimalStrings(raw.b, raw.a) >= 0) return { ok: false, reason: "bid/ask inversion: best_bid >= best_ask" };

    const freshness = computeFreshness(context.collectorReceivedAt, context.ingestAt);
    return {
      ok: true,
      event: {
        kind: "top_of_book",
        venue: BINANCE_VENUE,
        symbol: raw.s,
        exchangeEventAt: null,
        collectorReceivedAt: context.collectorReceivedAt,
        ingestAt: context.ingestAt,
        ageMs: freshness.ageMs,
        clockSkewMs: freshness.clockSkewMs,
        updateId: String(raw.u),
        bestBid: { price: raw.b, size: raw.B },
        bestAsk: { price: raw.a, size: raw.A },
        sourceLineage: { venue: BINANCE_VENUE, updateId: String(raw.u), symbol: raw.s },
        evidenceClass: EVIDENCE_CLASS,
        shadowOnly: true,
      },
    };
  },
};

/** Official Binance REST /api/v3/depth (or partial-depth stream) payload: {lastUpdateId,bids,asks}.
 * The response itself carries no symbol -- the caller requested one -- so `context.symbol` is
 * required here. */
export interface BinanceRawDepthSnapshot {
  lastUpdateId: number | string;
  bids: [string, string][];
  asks: [string, string][];
}

export const binanceDepthSnapshotAdapter: VenueBookAdapter<BinanceRawDepthSnapshot, DepthSnapshotEvent> = {
  venue: BINANCE_VENUE,
  normalize(raw, context): NormalizeResult<DepthSnapshotEvent> {
    const tsError = validateContextTimestamps(context);
    if (tsError) return { ok: false, reason: tsError };
    if (!context.symbol) return { ok: false, reason: "context.symbol is required for a Binance depth snapshot (the REST payload does not include one)" };
    if (!isSafeIntegerIdInput(raw.lastUpdateId)) return { ok: false, reason: `invalid or unsafe lastUpdateId: ${JSON.stringify(raw.lastUpdateId)}` };
    const bids = parseLevels(raw.bids);
    if (!bids.ok) return bids;
    const asks = parseLevels(raw.asks);
    if (!asks.ok) return asks;

    const freshness = computeFreshness(context.collectorReceivedAt, context.ingestAt);
    return {
      ok: true,
      event: {
        kind: "depth_snapshot",
        venue: BINANCE_VENUE,
        symbol: context.symbol,
        exchangeEventAt: null,
        collectorReceivedAt: context.collectorReceivedAt,
        ingestAt: context.ingestAt,
        ageMs: freshness.ageMs,
        clockSkewMs: freshness.clockSkewMs,
        lastUpdateId: String(raw.lastUpdateId),
        bids: bids.levels,
        asks: asks.levels,
        sourceLineage: { venue: BINANCE_VENUE, lastUpdateId: String(raw.lastUpdateId), symbol: context.symbol },
        evidenceClass: EVIDENCE_CLASS,
        shadowOnly: true,
      },
    };
  },
};

/** Official Binance <symbol>@depth diff-depth stream payload: {e,E,s,U,u,b,a}. `b`/`a` are level
 * MUTATIONS (a quantity of "0" deletes that price level) -- never a complete book. */
export interface BinanceRawDepthDiff {
  e: "depthUpdate";
  E: number;
  s: string;
  U: number | string;
  u: number | string;
  b: [string, string][];
  a: [string, string][];
}

export const binanceDepthDiffAdapter: VenueBookAdapter<BinanceRawDepthDiff, DepthDiffEvent> = {
  venue: BINANCE_VENUE,
  normalize(raw, context): NormalizeResult<DepthDiffEvent> {
    const tsError = validateContextTimestamps(context);
    if (tsError) return { ok: false, reason: tsError };
    if (raw.e !== "depthUpdate") return { ok: false, reason: `unexpected event type: ${JSON.stringify(raw.e)}` };
    if (typeof raw.s !== "string" || raw.s.length === 0) return { ok: false, reason: "missing symbol (s)" };
    if (!isSafeIntegerIdInput(raw.U) || !isSafeIntegerIdInput(raw.u)) {
      return { ok: false, reason: `invalid or unsafe U/u: ${JSON.stringify(raw.U)}/${JSON.stringify(raw.u)}` };
    }
    if (BigInt(raw.U) > BigInt(raw.u)) {
      return { ok: false, reason: `U (firstUpdateId, ${raw.U}) must not exceed u (finalUpdateId, ${raw.u})` };
    }
    if (!isValidEpochMillis(raw.E)) return { ok: false, reason: `invalid event time (E): ${JSON.stringify(raw.E)}` };
    const bidMutations = parseLevels(raw.b);
    if (!bidMutations.ok) return bidMutations;
    const askMutations = parseLevels(raw.a);
    if (!askMutations.ok) return askMutations;

    const exchangeEventAt = new Date(raw.E).toISOString();
    const freshness = computeFreshness(exchangeEventAt, context.ingestAt);
    return {
      ok: true,
      event: {
        kind: "depth_diff",
        venue: BINANCE_VENUE,
        symbol: raw.s,
        exchangeEventAt,
        collectorReceivedAt: context.collectorReceivedAt,
        ingestAt: context.ingestAt,
        ageMs: freshness.ageMs,
        clockSkewMs: freshness.clockSkewMs,
        firstUpdateId: String(raw.U),
        finalUpdateId: String(raw.u),
        bidMutations: bidMutations.levels,
        askMutations: askMutations.levels,
        sourceLineage: { venue: BINANCE_VENUE, U: String(raw.U), u: String(raw.u), symbol: raw.s },
        evidenceClass: EVIDENCE_CLASS,
        shadowOnly: true,
      },
    };
  },
};

// -------------------------------------------------------------------------------------------
// Fixture (explicitly synthetic) second-venue adapter -- proves the adapter boundary accepts a
// structurally different raw shape without being, or claiming to be, any real exchange. See file
// header point 7. Do not rename venue to a real exchange without implementing that exchange's
// actual official payload.
// -------------------------------------------------------------------------------------------

export interface FixtureRawTopOfBook {
  product: string;
  time: string;
  bestBid: { price: string; size: string };
  bestAsk: { price: string; size: string };
  sequence?: number | string;
}

export const fixtureTopOfBookAdapter: VenueBookAdapter<FixtureRawTopOfBook, TopOfBookEvent> = {
  venue: FIXTURE_VENUE,
  normalize(raw, context): NormalizeResult<TopOfBookEvent> {
    const tsError = validateContextTimestamps(context);
    if (tsError) return { ok: false, reason: tsError };
    if (typeof raw.product !== "string" || raw.product.length === 0) return { ok: false, reason: "missing product" };
    if (!isValidIsoTimestamp(raw.time)) return { ok: false, reason: `invalid exchange event time: ${JSON.stringify(raw.time)}` };
    for (const [field, value] of [["bestBid.price", raw.bestBid?.price], ["bestBid.size", raw.bestBid?.size], ["bestAsk.price", raw.bestAsk?.price], ["bestAsk.size", raw.bestAsk?.size]] as const) {
      if (!isValidDecimalString(value)) return { ok: false, reason: `invalid decimal for ${field}: ${JSON.stringify(value)}` };
    }
    if (compareDecimalStrings(raw.bestBid.price, raw.bestAsk.price) >= 0) return { ok: false, reason: "bid/ask inversion: best_bid >= best_ask" };
    if (raw.sequence != null && !isSafeIntegerIdInput(raw.sequence)) return { ok: false, reason: `invalid or unsafe sequence: ${JSON.stringify(raw.sequence)}` };

    const freshness = computeFreshness(raw.time, context.ingestAt);
    return {
      ok: true,
      event: {
        kind: "top_of_book",
        venue: FIXTURE_VENUE,
        symbol: raw.product,
        exchangeEventAt: raw.time,
        collectorReceivedAt: context.collectorReceivedAt,
        ingestAt: context.ingestAt,
        ageMs: freshness.ageMs,
        clockSkewMs: freshness.clockSkewMs,
        updateId: raw.sequence != null ? String(raw.sequence) : null,
        bestBid: { price: raw.bestBid.price, size: raw.bestBid.size },
        bestAsk: { price: raw.bestAsk.price, size: raw.bestAsk.size },
        sourceLineage: { venue: FIXTURE_VENUE, product: raw.product, sequence: raw.sequence ?? null },
        evidenceClass: EVIDENCE_CLASS,
        shadowOnly: true,
      },
    };
  },
};

export function keyOf(venue: string, symbol: string): string {
  return `${venue}:${symbol}`;
}

// -------------------------------------------------------------------------------------------
// Top-of-book state: "latest wins" with duplicate/out-of-order detection based on a
// non-decreasing update id. Deliberately NOT a +1/gap invariant -- see file header point 3.
// -------------------------------------------------------------------------------------------

export interface TopOfBookState {
  venue: string;
  symbol: string;
  event: TopOfBookEvent;
  lastUpdateId: bigint | null;
  appliedCount: number;
}

export interface TopOfBookIssue {
  venue: string;
  symbol: string;
  kind: "invalid" | "duplicate" | "out_of_order";
  reason: string;
  event: TopOfBookEvent;
}

export interface ReduceTopOfBookResult {
  states: Map<string, TopOfBookState>;
  issues: TopOfBookIssue[];
}

/**
 * Folds an ordered top-of-book event stream into current per-(venue,symbol) state.
 *
 *   - bid/ask inversion (defense-in-depth; adapters already reject this) -> issue "invalid",
 *     state unchanged.
 *   - an update id equal to the last applied one -> issue "duplicate", state unchanged.
 *   - an update id behind the last applied one -> issue "out_of_order", state unchanged.
 *   - any update id strictly ahead of the last applied one, by any margin -> always accepted.
 *     There is no gap concept here: a real venue's top-of-book stream (e.g. Binance bookTicker)
 *     is emitted on price/qty change, not on a fixed cadence, so large jumps between consecutive
 *     update ids are normal traffic, not evidence of missed messages.
 *   - an event with no update id at all -> always accepted (best-effort, arrival-order only), and
 *     does not reset the update-id baseline for a later event that does carry one.
 */
export function reduceTopOfBookEvents(events: TopOfBookEvent[]): ReduceTopOfBookResult {
  const states = new Map<string, TopOfBookState>();
  const issues: TopOfBookIssue[] = [];

  for (const event of events) {
    if (compareDecimalStrings(event.bestBid.price, event.bestAsk.price) >= 0) {
      issues.push({ venue: event.venue, symbol: event.symbol, kind: "invalid", reason: "bid/ask inversion: best_bid >= best_ask", event });
      continue;
    }

    const key = keyOf(event.venue, event.symbol);
    const prior = states.get(key) ?? null;
    const currentId = event.updateId != null ? BigInt(event.updateId) : null;

    if (currentId != null && prior?.lastUpdateId != null) {
      if (currentId === prior.lastUpdateId) {
        issues.push({ venue: event.venue, symbol: event.symbol, kind: "duplicate", reason: `duplicate updateId ${event.updateId}`, event });
        continue;
      }
      if (currentId < prior.lastUpdateId) {
        issues.push({ venue: event.venue, symbol: event.symbol, kind: "out_of_order", reason: `updateId ${event.updateId} is behind last applied ${prior.lastUpdateId}`, event });
        continue;
      }
    }

    states.set(key, {
      venue: event.venue,
      symbol: event.symbol,
      event,
      lastUpdateId: currentId ?? prior?.lastUpdateId ?? null,
      appliedCount: (prior?.appliedCount ?? 0) + 1,
    });
  }

  return { states, issues };
}

// -------------------------------------------------------------------------------------------
// Depth book reconstruction: Binance's actual documented local-order-book procedure. A real gap
// fails closed -- see file header point 4.
// -------------------------------------------------------------------------------------------

export type DepthBookStatus = "UNSYNCED" | "SYNCED" | "INVALID";

export interface DepthBookState {
  venue: string;
  symbol: string;
  status: DepthBookStatus;
  lastAppliedUpdateId: bigint | null;
  /**
   * Best-first. Only trustworthy as "the current book" when status === "SYNCED". Once status
   * becomes "INVALID" these are frozen at the last known-good (pre-gap) state, kept only for
   * forensic/debugging purposes -- callers MUST check `status` (or call `isDepthBookTrustworthy`)
   * before using them as live state. Never truncated -- every level currently held is returned.
   */
  bids: DecimalQuote[];
  asks: DecimalQuote[];
}

export function isDepthBookTrustworthy(state: DepthBookState): boolean {
  return state.status === "SYNCED";
}

export interface DepthBookIssue {
  venue: string;
  symbol: string;
  kind: "invalid" | "discarded_before_snapshot" | "stale_diff_discarded" | "gap_invalidated_book" | "diff_ignored_while_invalid" | "snapshot_too_old" | "lineage_mismatch" | "crossed_book_invalidated";
  reason: string;
  event: DepthSnapshotEvent | DepthDiffEvent;
}

export interface ReconstructDepthBookResult {
  states: Map<string, DepthBookState>;
  issues: DepthBookIssue[];
}

interface DepthBookInternal {
  status: DepthBookStatus;
  lastAppliedUpdateId: bigint | null;
  bids: Map<string, DecimalQuote>;
  asks: Map<string, DecimalQuote>;
}

function applyMutations(levels: Map<string, DecimalQuote>, mutations: DecimalQuote[]): void {
  for (const mutation of mutations) {
    const key = canonicalDecimalKey(mutation.price);
    if (isZeroQuantity(mutation.size)) {
      levels.delete(key);
    } else {
      levels.set(key, mutation);
    }
  }
}

function toDepthBookState(venue: string, symbol: string, internal: DepthBookInternal): DepthBookState {
  return {
    venue,
    symbol,
    status: internal.status,
    lastAppliedUpdateId: internal.lastAppliedUpdateId,
    bids: sortedLevels(internal.bids, "desc"),
    asks: sortedLevels(internal.asks, "asc"),
  };
}

function seedFromSnapshot(event: DepthSnapshotEvent): DepthBookInternal {
  const bids = new Map<string, DecimalQuote>();
  const asks = new Map<string, DecimalQuote>();
  for (const level of event.bids) bids.set(canonicalDecimalKey(level.price), level);
  for (const level of event.asks) asks.set(canonicalDecimalKey(level.price), level);
  return { status: "SYNCED", lastAppliedUpdateId: BigInt(event.lastUpdateId), bids, asks };
}

/** True when the reconstructed book's own best bid/ask are crossed or locked (bestBid >=
 * bestAsk). Only meaningful -- and only checked -- when both sides currently hold at least one
 * level; a one-sided book (e.g. right after a partial snapshot) is not itself a crossed book. */
function isBookCrossed(internal: DepthBookInternal): boolean {
  const bestBid = sortedLevels(internal.bids, "desc")[0];
  const bestAsk = sortedLevels(internal.asks, "asc")[0];
  if (!bestBid || !bestAsk) return false;
  return compareDecimalStrings(bestBid.price, bestAsk.price) >= 0;
}

/**
 * Seeds the book from a snapshot, then immediately validates the reconstructed best bid/ask
 * (per GPT-5.6 Sol's review: a corrupt/adversarial snapshot with bestBid >= bestAsk must never be
 * reported as a trustworthy SYNCED book). Crossed -> status forced to INVALID and an issue
 * returned, using the same INVALID/resync path a sequence gap uses -- there is only one way for a
 * reconstructed book to be untrustworthy, not a special case per failure mode.
 */
function seedFromSnapshotChecked(event: DepthSnapshotEvent): { internal: DepthBookInternal; issue: DepthBookIssue | null } {
  const internal = seedFromSnapshot(event);
  if (isBookCrossed(internal)) {
    internal.status = "INVALID";
    return {
      internal,
      issue: { venue: event.venue, symbol: event.symbol, kind: "crossed_book_invalidated", reason: "reconstructed book is crossed/inverted (best bid >= best ask) immediately after seeding from the snapshot", event },
    };
  }
  return { internal, issue: null };
}

/**
 * Applies one complete depth diff via `tryApplyDiff`, then -- only when that application
 * succeeded (no gap/stale issue) -- validates the resulting reconstructed best bid/ask. A diff
 * that leaves the book crossed is treated exactly like a sequence gap: status -> INVALID, an
 * issue is returned, and (since the mutation already happened) the crossed levels are what remain
 * -- this is intentional: the caller must resync from a fresh snapshot either way once INVALID,
 * and preserving the exact corrupt state is more useful for forensics than trying to roll it back.
 */
function tryApplyDiffChecked(internal: DepthBookInternal, event: DepthDiffEvent): DepthBookIssue | null {
  const issue = tryApplyDiff(internal, event);
  if (issue) return issue;
  if (isBookCrossed(internal)) {
    internal.status = "INVALID";
    return { venue: event.venue, symbol: event.symbol, kind: "crossed_book_invalidated", reason: "reconstructed book is crossed/inverted (best bid >= best ask) after applying a depth diff", event };
  }
  return null;
}

/**
 * The single update rule Binance's Spot docs actually document for applying a depth diff to an
 * already-seeded local book (used both for steady-state updates and to replay buffered diffs
 * after a startup snapshot -- see file header point 4):
 *
 *   - a diff whose finalUpdateId (u) is at or behind the locally-applied update id is entirely
 *     covered by what's already applied -- ignore it (discard, no state change).
 *   - a diff whose firstUpdateId (U) is *strictly greater* than localUpdateId + 1 means something
 *     was missed -- a real gap. Invalidate the book (status -> INVALID) and do NOT apply it.
 *   - anything else applies normally, INCLUDING a diff that overlaps already-applied history
 *     (firstUpdateId <= localUpdateId) as long as it still extends past it (finalUpdateId >
 *     localUpdateId) -- an overlapping-but-covering diff must never be falsely invalidated.
 *
 * Mutates `internal` in place; returns the issue to record, or null if the diff applied cleanly.
 */
function tryApplyDiff(internal: DepthBookInternal, event: DepthDiffEvent): DepthBookIssue | null {
  const firstId = BigInt(event.firstUpdateId);
  const finalId = BigInt(event.finalUpdateId);
  const lastApplied = internal.lastAppliedUpdateId as bigint;

  if (finalId <= lastApplied) {
    return { venue: event.venue, symbol: event.symbol, kind: "stale_diff_discarded", reason: `diff finalUpdateId ${event.finalUpdateId} is at or behind the local update id ${lastApplied}`, event };
  }
  if (firstId > lastApplied + 1n) {
    internal.status = "INVALID";
    return { venue: event.venue, symbol: event.symbol, kind: "gap_invalidated_book", reason: `sequence gap: diff firstUpdateId ${event.firstUpdateId} is ahead of local update id + 1 (${lastApplied + 1n}); events were missed`, event };
  }

  applyMutations(internal.bids, event.bidMutations);
  applyMutations(internal.asks, event.askMutations);
  internal.lastAppliedUpdateId = finalId;
  internal.status = "SYNCED";
  return null;
}

/**
 * Reconstructs current depth-book state from an ordered stream of depth snapshots and diffs.
 *
 *   1. A depth_snapshot always seeds (or reseeds) the book fresh via `seedFromSnapshot` -- a
 *      snapshot is strictly newer ground truth than anything reconstructed from diffs so far, so
 *      it always wins immediately (status -> SYNCED).
 *   2. A diff arriving before any snapshot has seeded the book for that (venue, symbol) is
 *      discarded -- issue "discarded_before_snapshot". This reducer does NOT buffer pre-snapshot
 *      diffs; if you need Binance's actual startup buffering procedure (fetch a snapshot, discard
 *      what it already covers, replay the rest), use `synchronizeDepthBookStartup` instead, which
 *      is the explicit primitive for that. Do not read "discarded" as "buffered" -- they are not
 *      the same thing.
 *   3. Once seeded, every diff is applied via `tryApplyDiff` -- the single Binance update rule
 *      (see its own doc comment). A real gap invalidates the book (status -> INVALID).
 *   4. While INVALID, every diff is ignored (issue "diff_ignored_while_invalid") until a fresh
 *      snapshot arrives and reseeds the book from step 1.
 *   5. After seeding a snapshot, and after applying a complete diff, the reconstructed best
 *      bid/ask are validated (when both sides are non-empty): a crossed/inverted result
 *      (bestBid >= bestAsk) invalidates the book (status -> INVALID, issue
 *      "crossed_book_invalidated") exactly like a sequence gap -- an adversarial/corrupt snapshot
 *      or diff must never be reported as a trustworthy SYNCED book.
 *
 * A quantity of exactly "0" on a mutation is an exact deletion instruction for that price level,
 * never a parse fallback; price levels are keyed by exact decimal value (`canonicalDecimalKey`),
 * so "25.10" and "25.1" are treated as one level, matching real venue semantics.
 */
export function reconstructDepthBook(events: (DepthSnapshotEvent | DepthDiffEvent)[]): ReconstructDepthBookResult {
  const internals = new Map<string, DepthBookInternal>();
  const issues: DepthBookIssue[] = [];

  for (const event of events) {
    const key = keyOf(event.venue, event.symbol);

    if (event.kind === "depth_snapshot") {
      const { internal, issue } = seedFromSnapshotChecked(event);
      internals.set(key, internal);
      if (issue) issues.push(issue);
      continue;
    }

    // event.kind === "depth_diff"
    let internal = internals.get(key);
    if (!internal || internal.status === "UNSYNCED") {
      if (!internal) {
        internal = { status: "UNSYNCED", lastAppliedUpdateId: null, bids: new Map(), asks: new Map() };
        internals.set(key, internal);
      }
      issues.push({ venue: event.venue, symbol: event.symbol, kind: "discarded_before_snapshot", reason: "diff received before any snapshot seeded the book for this venue/symbol; it is discarded, not buffered -- use synchronizeDepthBookStartup for real startup buffering semantics", event });
      continue;
    }

    if (internal.status === "INVALID") {
      issues.push({ venue: event.venue, symbol: event.symbol, kind: "diff_ignored_while_invalid", reason: "book is invalidated pending resync; diff ignored", event });
      continue;
    }

    const issue = tryApplyDiffChecked(internal, event);
    if (issue) issues.push(issue);
  }

  const states = new Map<string, DepthBookState>();
  for (const [key, internal] of internals) {
    const [venue, ...symbolParts] = key.split(":");
    states.set(key, toDepthBookState(venue, symbolParts.join(":"), internal));
  }
  return { states, issues };
}

export interface StartupSyncResult {
  /**
   * `outcome === "synced"` is a strong invariant: it is returned if and only if `state` is
   * present, `state.status === "SYNCED"`, and `isDepthBookTrustworthy(state)` is `true`. A caller
   * may gate on `outcome === "synced"` alone without separately re-checking `state.status`.
   *
   *   - "snapshot_too_old": the snapshot cannot be used as-is and a newer one must be fetched;
   *     no book is seeded, `state` is absent.
   *   - "invalid": either the input itself was invalid (a buffered/subsequent diff's venue or
   *     symbol did not match the snapshot's -- see `findLineageMismatch` -- in which case no
   *     mutation was ever applied and `state` is absent), or seeding/replay produced an
   *     untrustworthy book -- a real sequence gap, or a reconstructed book left crossed/inverted
   *     (bestBid >= bestAsk) by the snapshot itself or by a diff (in which case `state` IS
   *     present, frozen at the last known-good state, with `state.status === "INVALID"` --
   *     exactly like `reconstructDepthBook`'s own INVALID state -- but it must never be read as
   *     current/trustworthy; check `isDepthBookTrustworthy` or the outcome tag itself before using
   *     it for anything but forensics).
   */
  outcome: "synced" | "snapshot_too_old" | "invalid";
  state?: DepthBookState;
  issues: DepthBookIssue[];
}

/** Returns the first diff whose venue or symbol does not match the given ones, or null if every
 * diff belongs to the same (venue, symbol). Used to fail closed before `synchronizeDepthBookStartup`
 * applies any mutation -- a snapshot for one venue/symbol must never be advanced or mutated by a
 * diff that actually belongs to a different one. */
function findLineageMismatch(venue: string, symbol: string, diffs: DepthDiffEvent[]): DepthDiffEvent | null {
  for (const diff of diffs) {
    if (diff.venue !== venue || diff.symbol !== symbol) return diff;
  }
  return null;
}

/**
 * Implements Binance's actual documented startup procedure for a local order book, as its own
 * explicit pure primitive (per GPT-5.6 Sol's review: `reconstructDepthBook`'s plain ordered-stream
 * reducer discards pre-snapshot diffs rather than buffering them -- this is the primitive that
 * does implement the real buffering procedure):
 *
 *   1. Every buffered and subsequent diff must belong to the exact same (venue, symbol) as the
 *      snapshot. If any diff does not, fail closed BEFORE seeding or applying anything: returns
 *      `{ outcome: "invalid" }` with no `state` at all. `reconstructDepthBook` isolates events by
 *      (venue, symbol) automatically (a separate internal book per key); this primitive works
 *      against a single snapshot + flat diff arrays, so it must check this explicitly instead --
 *      a foreign diff must never be allowed to mutate or advance another symbol's book, and mixed
 *      lineage must never be able to produce a `"synced"` result.
 *   2. If any diffs were buffered before the snapshot arrived, and the snapshot's lastUpdateId is
 *      older than the FIRST buffered diff's firstUpdateId, the snapshot is too old to align
 *      against -- returns `{ outcome: "snapshot_too_old" }` without seeding anything; the caller
 *      must fetch a newer snapshot and try again.
 *   3. Otherwise, seed the local book from the snapshot and immediately validate its reconstructed
 *      best bid/ask (`seedFromSnapshotChecked`) -- a corrupt/adversarial snapshot with
 *      bestBid >= bestAsk invalidates the book on the spot, the same as a sequence gap would.
 *   4. Replay every buffered diff, then every subsequent diff, in order, through the exact same
 *      update rule steady-state uses (`tryApplyDiffChecked`): a diff entirely covered by the
 *      snapshot (or by what's already been replayed) is discarded; a diff whose firstUpdateId is
 *      ahead of lastAppliedUpdateId + 1 is a real gap and invalidates the book; a diff that applies
 *      but leaves the reconstructed book crossed also invalidates it; otherwise it applies
 *      normally -- including the first diff to apply after the snapshot, which in a healthy
 *      stream will bracket the snapshot's lastUpdateId by construction, but is not special-cased
 *      here: it is just the first call to the same rule as every other diff.
 *   5. The outcome reported is derived from the resulting book's own status, never asserted
 *      independently of it: `"synced"` only when the book actually ended up SYNCED; `"invalid"`
 *      when replay hit a real gap, a crossed book, or mismatched lineage and left it INVALID. This
 *      is what makes the strong invariant on `StartupSyncResult.outcome` hold structurally rather
 *      than by convention.
 */
export function synchronizeDepthBookStartup(
  bufferedDiffs: DepthDiffEvent[],
  snapshot: DepthSnapshotEvent,
  subsequentDiffs: DepthDiffEvent[] = [],
): StartupSyncResult {
  const issues: DepthBookIssue[] = [];
  const allDiffs = [...bufferedDiffs, ...subsequentDiffs];

  const mismatch = findLineageMismatch(snapshot.venue, snapshot.symbol, allDiffs);
  if (mismatch) {
    issues.push({
      venue: snapshot.venue,
      symbol: snapshot.symbol,
      kind: "lineage_mismatch",
      reason: `diff venue/symbol (${mismatch.venue}/${mismatch.symbol}) does not match the snapshot's (${snapshot.venue}/${snapshot.symbol}); refusing to apply any mutation from mismatched lineage`,
      event: mismatch,
    });
    return { outcome: "invalid", issues };
  }

  if (bufferedDiffs.length > 0) {
    const firstBufferedUpdateId = BigInt(bufferedDiffs[0].firstUpdateId);
    const snapshotLastUpdateId = BigInt(snapshot.lastUpdateId);
    if (snapshotLastUpdateId < firstBufferedUpdateId) {
      issues.push({
        venue: snapshot.venue,
        symbol: snapshot.symbol,
        kind: "snapshot_too_old",
        reason: `snapshot lastUpdateId ${snapshot.lastUpdateId} is older than the first buffered diff's firstUpdateId ${bufferedDiffs[0].firstUpdateId}; a newer snapshot is required before syncing can proceed`,
        event: snapshot,
      });
      return { outcome: "snapshot_too_old", issues };
    }
  }

  const { internal, issue: seedIssue } = seedFromSnapshotChecked(snapshot);
  if (seedIssue) issues.push(seedIssue);
  for (const diff of allDiffs) {
    if (internal.status === "INVALID") {
      issues.push({ venue: diff.venue, symbol: diff.symbol, kind: "diff_ignored_while_invalid", reason: "book is invalidated pending resync; diff ignored", event: diff });
      continue;
    }
    const issue = tryApplyDiffChecked(internal, diff);
    if (issue) issues.push(issue);
  }

  const state = toDepthBookState(snapshot.venue, snapshot.symbol, internal);
  return { outcome: state.status === "SYNCED" ? "synced" : "invalid", state, issues };
}

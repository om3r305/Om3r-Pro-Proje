// Behavioral tests for the Reflex/L2 event-state foundation (brian-2026 issue #32), rewritten per
// GPT-5.6 Sol's review on PR #37. Fixture payloads for Binance are the exact example shapes from
// Binance's own official Spot API documentation (bookTicker, diff-depth, REST depth snapshot) --
// not invented field names. Every test is pure/fixture-driven: no network, no Supabase, no
// wall-clock dependency, since every timestamp is supplied explicitly.

import { assert, assertEquals, assertNotEquals } from "jsr:@std/assert@^1.0.0";
import {
  binanceBookTickerAdapter,
  binanceDepthDiffAdapter,
  binanceDepthSnapshotAdapter,
  canonicalDecimalKey,
  compareDecimalStrings,
  computeFreshness,
  FIXTURE_VENUE,
  fixtureTopOfBookAdapter,
  isDepthBookTrustworthy,
  isSafeIntegerIdInput,
  isValidDecimalString,
  isValidEpochMillis,
  isValidIntegerString,
  isValidIsoTimestamp,
  isZeroQuantity,
  keyOf,
  reconstructDepthBook,
  reduceTopOfBookEvents,
  synchronizeDepthBookStartup,
  type BinanceRawDepthDiff,
  type BinanceRawDepthSnapshot,
  type DepthDiffEvent,
  type DepthSnapshotEvent,
  type NormalizeContext,
  type TopOfBookEvent,
} from "./l2_book.ts";

function ctx(overrides: Partial<NormalizeContext> = {}): NormalizeContext {
  return { collectorReceivedAt: "2026-09-04T00:00:00.000Z", ingestAt: "2026-09-04T00:00:00.050Z", ...overrides };
}

// -------------------------------------------------------------------------------------------
// Exact-decimal helpers
// -------------------------------------------------------------------------------------------

Deno.test("isValidDecimalString: accepts venue-style decimal strings, rejects garbage", () => {
  assert(isValidDecimalString("25.35190000"));
  assert(isValidDecimalString("0"));
  assert(isValidDecimalString("100"));
  assert(!isValidDecimalString("25.35.19"));
  assert(!isValidDecimalString("-25.35"));
  assert(!isValidDecimalString("abc"));
  assert(!isValidDecimalString(""));
  assert(!isValidDecimalString(25.35));
  assert(!isValidDecimalString(null));
  assert(!isValidDecimalString(undefined));
});

Deno.test("isValidIntegerString: accepts non-negative integer strings only", () => {
  assert(isValidIntegerString("400900217"));
  assert(isValidIntegerString("0"));
  assert(!isValidIntegerString("400900217.0"));
  assert(!isValidIntegerString("-5"));
  assert(!isValidIntegerString("abc"));
  assert(!isValidIntegerString(400900217));
});

Deno.test("isSafeIntegerIdInput: accepts a lossless string of any magnitude, but rejects an unsafe JS number", () => {
  assert(isSafeIntegerIdInput("400900217"));
  assert(isSafeIntegerIdInput(400900217));
  // Binance ids are int64; a number already above Number.MAX_SAFE_INTEGER may have been rounded
  // by JSON.parse before this code ever sees it -- reject it rather than launder that precision
  // loss into a confident-looking string.
  assert(!isSafeIntegerIdInput(Number.MAX_SAFE_INTEGER + 10));
  // The same magnitude arriving as an exact string is still lossless and must be accepted.
  assert(isSafeIntegerIdInput("9223372036854775807"));
  assert(!isSafeIntegerIdInput(-5));
  assert(!isSafeIntegerIdInput(1.5));
  assert(!isSafeIntegerIdInput("abc"));
  assert(!isSafeIntegerIdInput(null));
});

Deno.test("isValidEpochMillis: accepts a positive finite number only", () => {
  assert(isValidEpochMillis(1672515782136));
  assert(!isValidEpochMillis(0));
  assert(!isValidEpochMillis(-5));
  assert(!isValidEpochMillis(NaN));
  assert(!isValidEpochMillis(Infinity));
  assert(!isValidEpochMillis("1672515782136"));
});

Deno.test("isValidIsoTimestamp: accepts a parseable timestamp, rejects garbage", () => {
  assert(isValidIsoTimestamp("2026-09-04T00:00:00.000Z"));
  assert(!isValidIsoTimestamp("not-a-date"));
  assert(!isValidIsoTimestamp(""));
  assert(!isValidIsoTimestamp(null));
});

Deno.test("compareDecimalStrings: exact comparison across mismatched decimal precision, never via float", () => {
  assertEquals(compareDecimalStrings("25.1", "25.10"), 0, "25.1 and 25.10 are the same value");
  assertEquals(compareDecimalStrings("25.09", "25.1"), -1);
  assertEquals(compareDecimalStrings("25.35190000", "25.36520000"), -1);
  assertEquals(compareDecimalStrings("100", "99.9999"), 1);
  assertEquals(compareDecimalStrings("0", "0.00000000"), 0);
});

Deno.test("isZeroQuantity: exact-zero detection regardless of trailing zero formatting", () => {
  assert(isZeroQuantity("0"));
  assert(isZeroQuantity("0.00000000"));
  assert(!isZeroQuantity("0.00000001"));
  assert(!isZeroQuantity("10"));
});

Deno.test("canonicalDecimalKey: '25.10' and '25.1' canonicalize to the same key", () => {
  assertEquals(canonicalDecimalKey("25.10"), canonicalDecimalKey("25.1"));
  assertNotEquals(canonicalDecimalKey("25.10"), canonicalDecimalKey("25.11"));
});

Deno.test("computeFreshness: non-negative ageMs is unaffected, clockSkewMs mirrors it", () => {
  const f = computeFreshness("2026-09-04T00:00:00.000Z", "2026-09-04T00:00:00.200Z");
  assertEquals(f.ageMs, 200);
  assertEquals(f.clockSkewMs, 200);
});

Deno.test("computeFreshness: a reference timestamp AHEAD of ingestAt is preserved as negative clockSkewMs, not hidden by clamping", () => {
  // e.g. a reordered message, or a venue clock running ahead of the collector's clock.
  const f = computeFreshness("2026-09-04T00:00:01.000Z", "2026-09-04T00:00:00.500Z");
  assertEquals(f.clockSkewMs, -500, "the skew itself must be observable, unclamped");
  assertEquals(f.ageMs, 0, "age is still a sane non-negative 'how stale' metric for freshness gating");
});

// -------------------------------------------------------------------------------------------
// binanceBookTickerAdapter -- official payload shape: {u,s,b,B,a,A} only, no E, no depth.
// -------------------------------------------------------------------------------------------

function officialBookTicker(overrides: Record<string, unknown> = {}) {
  // Exact example values from Binance's own Individual Symbol Book Ticker Streams documentation.
  return { u: 400900217, s: "BNBUSDT", b: "25.35190000", B: "31.21000000", a: "25.36520000", A: "40.66000000", ...overrides };
}

Deno.test("binanceBookTickerAdapter: normalizes the official example payload exactly", () => {
  const result = binanceBookTickerAdapter.normalize(officialBookTicker(), ctx());
  assert(result.ok);
  if (!result.ok) return;
  assertEquals(result.event.kind, "top_of_book");
  assertEquals(result.event.venue, "binance");
  assertEquals(result.event.symbol, "BNBUSDT");
  assertEquals(result.event.updateId, "400900217");
  assertEquals(result.event.bestBid, { price: "25.35190000", size: "31.21000000" }, "decimal strings must be preserved exactly, not parsed to number");
  assertEquals(result.event.bestAsk, { price: "25.36520000", size: "40.66000000" });
  assertEquals(result.event.exchangeEventAt, null, "real bookTicker has no event time field");
  assertEquals(result.event.shadowOnly, true);
  assertEquals(result.event.evidenceClass, "PROSPECTIVE_DEVELOPMENT_SHADOW");
});

Deno.test("binanceBookTickerAdapter: rejects a crossed/inverted book", () => {
  const result = binanceBookTickerAdapter.normalize(officialBookTicker({ b: "25.40", a: "25.30" }), ctx());
  assertEquals(result.ok, false);
});

Deno.test("binanceBookTickerAdapter: fails closed on a malformed decimal field instead of coercing to zero", () => {
  const result = binanceBookTickerAdapter.normalize(officialBookTicker({ b: "not-a-number" }), ctx());
  assertEquals(result.ok, false);
  if (result.ok) return;
  assert(result.reason.includes("b"), `expected the rejection reason to name the bad field, got: ${result.reason}`);
});

Deno.test("binanceBookTickerAdapter: fails closed on a malformed updateId", () => {
  const result = binanceBookTickerAdapter.normalize(officialBookTicker({ u: "not-an-id" }), ctx());
  assertEquals(result.ok, false);
});

Deno.test("binanceBookTickerAdapter: fails closed on an unsafe numeric updateId, preserves a large one losslessly as a string", () => {
  assertEquals(binanceBookTickerAdapter.normalize(officialBookTicker({ u: Number.MAX_SAFE_INTEGER + 10 }), ctx()).ok, false);
  const result = binanceBookTickerAdapter.normalize(officialBookTicker({ u: "9223372036854775807" }), ctx());
  assert(result.ok);
  if (!result.ok) return;
  assertEquals(result.event.updateId, "9223372036854775807", "an int64-range id given as a string must be preserved exactly, not truncated/rounded");
});

Deno.test("binanceBookTickerAdapter: fails closed on an invalid ingestAt timestamp rather than silently defaulting", () => {
  const result = binanceBookTickerAdapter.normalize(officialBookTicker(), ctx({ ingestAt: "not-a-date" }));
  assertEquals(result.ok, false);
});

// -------------------------------------------------------------------------------------------
// Required regression: large normal jumps in bookTicker.u must NOT be treated as a gap.
// -------------------------------------------------------------------------------------------

Deno.test("reduceTopOfBookEvents: a real Binance bookTicker with a large jump in u is accepted, not flagged as a gap", () => {
  const first = binanceBookTickerAdapter.normalize(officialBookTicker({ u: 100 }), ctx());
  const second = binanceBookTickerAdapter.normalize(officialBookTicker({ u: 400900217, b: "25.40000000", a: "25.41000000" }), ctx());
  assert(first.ok && second.ok);
  if (!first.ok || !second.ok) return;

  const { states, issues } = reduceTopOfBookEvents([first.event, second.event]);
  assertEquals(issues.length, 0, "a large jump in bookTicker.u is normal traffic (emitted on price/qty change, not a fixed cadence) -- it must never raise a gap");
  const state = states.get(keyOf("binance", "BNBUSDT"))!;
  assertEquals(state.lastUpdateId, 400900217n);
  assertEquals(state.appliedCount, 2);
});

Deno.test("reduceTopOfBookEvents: a duplicate updateId is flagged and does not change state", () => {
  const a = binanceBookTickerAdapter.normalize(officialBookTicker({ u: 100 }), ctx());
  const b = binanceBookTickerAdapter.normalize(officialBookTicker({ u: 100, b: "99.00000000", a: "99.10000000" }), ctx());
  assert(a.ok && b.ok);
  if (!a.ok || !b.ok) return;
  const { states, issues } = reduceTopOfBookEvents([a.event, b.event]);
  const state = states.get(keyOf("binance", "BNBUSDT"))!;
  assertEquals(state.event.bestBid.price, "25.35190000", "the duplicate must not overwrite the already-applied state");
  assertEquals(issues.length, 1);
  assertEquals(issues[0].kind, "duplicate");
});

Deno.test("reduceTopOfBookEvents: an updateId behind the last applied one is flagged out_of_order and does not change state", () => {
  const a = binanceBookTickerAdapter.normalize(officialBookTicker({ u: 200 }), ctx());
  const b = binanceBookTickerAdapter.normalize(officialBookTicker({ u: 100, b: "99.00000000", a: "99.10000000" }), ctx());
  assert(a.ok && b.ok);
  if (!a.ok || !b.ok) return;
  const { states, issues } = reduceTopOfBookEvents([a.event, b.event]);
  const state = states.get(keyOf("binance", "BNBUSDT"))!;
  assertEquals(state.lastUpdateId, 200n);
  assertEquals(issues.length, 1);
  assertEquals(issues[0].kind, "out_of_order");
});

Deno.test("reduceTopOfBookEvents: an event with no updateId is always applied and does not reset the id baseline", () => {
  const a = binanceBookTickerAdapter.normalize(officialBookTicker({ u: 100 }), ctx());
  assert(a.ok);
  if (!a.ok) return;
  const noId: TopOfBookEvent = { ...a.event, updateId: null };
  const c = binanceBookTickerAdapter.normalize(officialBookTicker({ u: 101 }), ctx());
  assert(c.ok);
  if (!c.ok) return;

  const { states, issues } = reduceTopOfBookEvents([a.event, noId, c.event]);
  assertEquals(issues.length, 0, "101 following 100 must still be accepted even after an unsequenced event in between");
  const state = states.get(keyOf("binance", "BNBUSDT"))!;
  assertEquals(state.appliedCount, 3);
  assertEquals(state.lastUpdateId, 101n);
});

Deno.test("reduceTopOfBookEvents: bid/ask inversion is rejected as defense-in-depth", () => {
  const a = binanceBookTickerAdapter.normalize(officialBookTicker({ u: 100 }), ctx());
  assert(a.ok);
  if (!a.ok) return;
  const inverted: TopOfBookEvent = { ...a.event, updateId: "101", bestBid: { price: "30", size: "1" }, bestAsk: { price: "20", size: "1" } };
  const { states, issues } = reduceTopOfBookEvents([a.event, inverted]);
  const state = states.get(keyOf("binance", "BNBUSDT"))!;
  assertEquals(state.appliedCount, 1);
  assertEquals(issues.length, 1);
  assertEquals(issues[0].kind, "invalid");
});

// -------------------------------------------------------------------------------------------
// binanceDepthSnapshotAdapter -- official payload shape: {lastUpdateId,bids,asks}, no symbol.
// -------------------------------------------------------------------------------------------

function officialDepthSnapshot(overrides: Partial<BinanceRawDepthSnapshot> = {}): BinanceRawDepthSnapshot {
  // Exact example values from Binance's Order Book (REST) documentation.
  return { lastUpdateId: 1027024, bids: [["4.00000000", "431.00000000"]], asks: [["4.00000200", "12.00000000"]], ...overrides };
}

Deno.test("binanceDepthSnapshotAdapter: normalizes the official example payload exactly", () => {
  const result = binanceDepthSnapshotAdapter.normalize(officialDepthSnapshot(), ctx({ symbol: "BNBBTC" }));
  assert(result.ok);
  if (!result.ok) return;
  assertEquals(result.event.kind, "depth_snapshot");
  assertEquals(result.event.lastUpdateId, "1027024");
  assertEquals(result.event.bids, [{ price: "4.00000000", size: "431.00000000" }]);
  assertEquals(result.event.asks, [{ price: "4.00000200", size: "12.00000000" }]);
});

Deno.test("binanceDepthSnapshotAdapter: requires context.symbol, since the REST payload does not carry one", () => {
  const result = binanceDepthSnapshotAdapter.normalize(officialDepthSnapshot(), ctx());
  assertEquals(result.ok, false);
});

Deno.test("binanceDepthSnapshotAdapter: fails closed on a malformed level entry instead of coercing it", () => {
  const result = binanceDepthSnapshotAdapter.normalize(officialDepthSnapshot({ bids: [["not-a-price", "1"]] }), ctx({ symbol: "BNBBTC" }));
  assertEquals(result.ok, false);
});

Deno.test("binanceDepthSnapshotAdapter: fails closed on an unsafe numeric lastUpdateId", () => {
  const result = binanceDepthSnapshotAdapter.normalize(officialDepthSnapshot({ lastUpdateId: Number.MAX_SAFE_INTEGER + 10 }), ctx({ symbol: "BNBBTC" }));
  assertEquals(result.ok, false);
});

// -------------------------------------------------------------------------------------------
// binanceDepthDiffAdapter -- official payload shape: {e,E,s,U,u,b,a}.
// -------------------------------------------------------------------------------------------

function officialDepthDiff(overrides: Partial<BinanceRawDepthDiff> = {}): BinanceRawDepthDiff {
  // Exact example values from Binance's Diff. Depth Stream documentation.
  return {
    e: "depthUpdate",
    E: 1672515782136,
    s: "BNBBTC",
    U: 157,
    u: 160,
    b: [["0.0024", "10"]],
    a: [["0.0026", "100"]],
    ...overrides,
  };
}

Deno.test("binanceDepthDiffAdapter: normalizes the official example payload exactly", () => {
  const result = binanceDepthDiffAdapter.normalize(officialDepthDiff(), ctx());
  assert(result.ok);
  if (!result.ok) return;
  assertEquals(result.event.kind, "depth_diff");
  assertEquals(result.event.firstUpdateId, "157");
  assertEquals(result.event.finalUpdateId, "160");
  assertEquals(result.event.bidMutations, [{ price: "0.0024", size: "10" }]);
  assertEquals(result.event.askMutations, [{ price: "0.0026", size: "100" }]);
  assertEquals(result.event.exchangeEventAt, new Date(1672515782136).toISOString());
});

Deno.test("binanceDepthDiffAdapter: rejects an unexpected event type", () => {
  const result = binanceDepthDiffAdapter.normalize(officialDepthDiff({ e: "trade" as "depthUpdate" }), ctx());
  assertEquals(result.ok, false);
});

Deno.test("binanceDepthDiffAdapter: fails closed on an invalid event time instead of throwing or defaulting", () => {
  const result = binanceDepthDiffAdapter.normalize(officialDepthDiff({ E: -1 }), ctx());
  assertEquals(result.ok, false);
});

Deno.test("binanceDepthDiffAdapter: fails closed on invalid U/u", () => {
  assertEquals(binanceDepthDiffAdapter.normalize(officialDepthDiff({ U: -1 }), ctx()).ok, false);
  assertEquals(binanceDepthDiffAdapter.normalize(officialDepthDiff({ u: "abc" }), ctx()).ok, false);
});

Deno.test("binanceDepthDiffAdapter: fails closed on an unsafe numeric U or u", () => {
  assertEquals(binanceDepthDiffAdapter.normalize(officialDepthDiff({ U: Number.MAX_SAFE_INTEGER + 10 }), ctx()).ok, false);
  assertEquals(binanceDepthDiffAdapter.normalize(officialDepthDiff({ u: Number.MAX_SAFE_INTEGER + 10 }), ctx()).ok, false);
});

Deno.test("binanceDepthDiffAdapter: fails closed when U > u", () => {
  const result = binanceDepthDiffAdapter.normalize(officialDepthDiff({ U: 200, u: 160 }), ctx());
  assertEquals(result.ok, false);
  if (result.ok) return;
  assert(result.reason.includes("U"), `expected the rejection to name U/u, got: ${result.reason}`);
});

// -------------------------------------------------------------------------------------------
// fixtureTopOfBookAdapter -- explicitly synthetic, must never claim a real venue's identity.
// -------------------------------------------------------------------------------------------

Deno.test("fixtureTopOfBookAdapter: is not, and does not claim to be, a real exchange", () => {
  assertEquals(fixtureTopOfBookAdapter.venue, FIXTURE_VENUE);
  assertNotEquals(fixtureTopOfBookAdapter.venue, "coinbase");
  assertNotEquals(fixtureTopOfBookAdapter.venue, "binance");
  assert(FIXTURE_VENUE.toLowerCase().includes("fixture"), "the venue id itself must self-disclose that it is synthetic");
});

Deno.test("fixtureTopOfBookAdapter: normalizes a structurally different raw shape into the same TopOfBookEvent contract", () => {
  const raw = {
    product: "XYZ-TEST",
    time: "2026-09-04T00:00:00.030Z",
    bestBid: { price: "60000.10", size: "0.8" },
    bestAsk: { price: "60000.60", size: "1.2" },
    sequence: 555,
  };
  const result = fixtureTopOfBookAdapter.normalize(raw, ctx({ ingestAt: "2026-09-04T00:00:00.080Z" }));
  assert(result.ok);
  if (!result.ok) return;
  assertEquals(result.event.venue, FIXTURE_VENUE);
  assertEquals(result.event.symbol, "XYZ-TEST");
  assertEquals(result.event.updateId, "555");
  assertEquals(result.event.ageMs, 50);
});

Deno.test("fixtureTopOfBookAdapter: fails closed on an unsafe numeric sequence", () => {
  const raw = {
    product: "XYZ-TEST",
    time: "2026-09-04T00:00:00.030Z",
    bestBid: { price: "60000.10", size: "0.8" },
    bestAsk: { price: "60000.60", size: "1.2" },
    sequence: Number.MAX_SAFE_INTEGER + 10,
  };
  assertEquals(fixtureTopOfBookAdapter.normalize(raw, ctx()).ok, false);
});

Deno.test("fixtureTopOfBookAdapter: rejects a crossed/inverted book just like the Binance adapter", () => {
  const raw = {
    product: "XYZ-TEST",
    time: "2026-09-04T00:00:00.030Z",
    bestBid: { price: "60001.00", size: "0.8" },
    bestAsk: { price: "60000.00", size: "1.2" },
  };
  assertEquals(fixtureTopOfBookAdapter.normalize(raw, ctx()).ok, false);
});

// -------------------------------------------------------------------------------------------
// reconstructDepthBook -- the plain ordered-stream reducer. It seeds from a snapshot immediately
// and applies diffs via the single Binance update rule; it does NOT buffer pre-snapshot diffs
// (see synchronizeDepthBookStartup below for that).
// -------------------------------------------------------------------------------------------

function snapshotEvent(lastUpdateId: number, bids: [string, string][], asks: [string, string][]): DepthSnapshotEvent {
  const result = binanceDepthSnapshotAdapter.normalize({ lastUpdateId, bids, asks }, ctx({ symbol: "BNBBTC" }));
  if (!result.ok) throw new Error(`test fixture setup failed: ${result.reason}`);
  return result.event;
}

function diffEvent(U: number, u: number, b: [string, string][] = [], a: [string, string][] = []): DepthDiffEvent {
  const result = binanceDepthDiffAdapter.normalize(officialDepthDiff({ U, u, b, a }), ctx());
  if (!result.ok) throw new Error(`test fixture setup failed: ${result.reason}`);
  return result.event;
}

Deno.test("reconstructDepthBook: a diff arriving before any snapshot is discarded (honestly, not claimed as buffered)", () => {
  const diff = diffEvent(1, 5);
  const { states, issues } = reconstructDepthBook([diff]);
  const state = states.get(keyOf("binance", "BNBBTC"))!;
  assertEquals(state.status, "UNSYNCED");
  assertEquals(state.bids.length, 0);
  assertEquals(issues[0].kind, "discarded_before_snapshot");
  assertEquals(isDepthBookTrustworthy(state), false);
});

Deno.test("reconstructDepthBook: a snapshot seeds the book immediately, then a covered diff is discarded and the next diff applies", () => {
  const snapshot = snapshotEvent(1000, [["10.00", "1"]], [["10.10", "1"]]);
  const coveredDiff = diffEvent(990, 1000, [["9.00", "5"]]); // entirely covered by the snapshot
  const nextDiff = diffEvent(1001, 1001, [["10.00", "2"]]); // firstUpdateId === lastAppliedUpdateId + 1

  const { states, issues } = reconstructDepthBook([snapshot, coveredDiff, nextDiff]);
  const state = states.get(keyOf("binance", "BNBBTC"))!;
  assertEquals(state.status, "SYNCED");
  assertEquals(state.lastAppliedUpdateId, 1001n);
  assertEquals(state.bids[0], { price: "10.00", size: "2" }, "the next diff's mutation must have been applied");
  const kinds = issues.map((i) => i.kind);
  assert(kinds.includes("stale_diff_discarded"), "the diff covered by the snapshot must be discarded, not applied");
});

Deno.test("reconstructDepthBook: an overlapping-but-covering diff is accepted, not falsely invalidated", () => {
  // Blocker from GPT-5.6 Sol's follow-up review: requiring firstUpdateId === lastApplied + 1
  // exactly is too strict. Binance's documented rule only fails closed when firstUpdateId is
  // STRICTLY AHEAD of lastApplied + 1; a diff whose range starts at or before what's already
  // applied, but still extends past it, must be accepted.
  const snapshot = snapshotEvent(1000, [["10.00", "1"]], [["10.10", "1"]]);
  const d1 = diffEvent(1001, 1005, [["10.00", "2"]]); // lastApplied -> 1005
  const overlapping = diffEvent(1002, 1010, [["10.00", "9"]]); // U=1002 <= 1005, but u=1010 > 1005

  const { states, issues } = reconstructDepthBook([snapshot, d1, overlapping]);
  const state = states.get(keyOf("binance", "BNBBTC"))!;
  assertEquals(state.status, "SYNCED", "an overlapping-but-covering diff must never be falsely invalidated");
  assertEquals(state.lastAppliedUpdateId, 1010n);
  assertEquals(state.bids[0], { price: "10.00", size: "9" }, "the overlapping diff's mutation must have been applied");
  assertEquals(issues.filter((i) => i.kind === "gap_invalidated_book").length, 0);
});

Deno.test("reconstructDepthBook: continuity holds across consecutive aligned diffs", () => {
  const snapshot = snapshotEvent(1000, [["10.00", "1"]], [["10.10", "1"]]);
  const d1 = diffEvent(1001, 1001, [["10.00", "2"]]);
  const d2 = diffEvent(1002, 1005, [["10.00", "3"]]); // firstUpdateId 1002 === lastApplied(1001)+1

  const { states, issues } = reconstructDepthBook([snapshot, d1, d2]);
  const state = states.get(keyOf("binance", "BNBBTC"))!;
  assertEquals(state.status, "SYNCED");
  assertEquals(state.lastAppliedUpdateId, 1005n);
  assertEquals(state.bids[0], { price: "10.00", size: "3" });
  assertEquals(issues.length, 0);
});

Deno.test("reconstructDepthBook: a duplicate/old diff while SYNCED is discarded and does not disturb state", () => {
  const snapshot = snapshotEvent(1000, [["10.00", "1"]], [["10.10", "1"]]);
  const d1 = diffEvent(1001, 1001, [["10.00", "2"]]);
  const replay = diffEvent(1001, 1001, [["10.00", "999"]]); // a stale replay of an already-applied range

  const { states, issues } = reconstructDepthBook([snapshot, d1, replay]);
  const state = states.get(keyOf("binance", "BNBBTC"))!;
  assertEquals(state.bids[0], { price: "10.00", size: "2" }, "the replayed diff must not overwrite already-applied state");
  assertEquals(state.status, "SYNCED");
  const kinds = issues.map((i) => i.kind);
  assert(kinds.includes("stale_diff_discarded"));
});

Deno.test("reconstructDepthBook: a real sequence gap invalidates the book and the post-gap diff is NOT applied", () => {
  const snapshot = snapshotEvent(1000, [["10.00", "1"]], [["10.10", "1"]]);
  const d1 = diffEvent(1001, 1005, [["10.00", "2"]]); // lastApplied -> 1005
  const gapDiff = diffEvent(1010, 1012, [["10.00", "999999"]]); // expected firstUpdateId 1006, got 1010

  const { states, issues } = reconstructDepthBook([snapshot, d1, gapDiff]);
  const state = states.get(keyOf("binance", "BNBBTC"))!;
  assertEquals(state.status, "INVALID");
  assertEquals(isDepthBookTrustworthy(state), false, "a caller must be able to cheaply tell this state is not usable");
  assertEquals(state.bids[0], { price: "10.00", size: "2" }, "the post-gap diff's mutation must never be applied -- the frozen pre-gap state is preserved, not corrupted");
  assertEquals(state.lastAppliedUpdateId, 1005n, "lastAppliedUpdateId must not advance past the gap");
  const kinds = issues.map((i) => i.kind);
  assert(kinds.includes("gap_invalidated_book"));
});

Deno.test("reconstructDepthBook: once INVALID, further diffs are ignored until a fresh snapshot resyncs", () => {
  const snapshot = snapshotEvent(1000, [["10.00", "1"]], [["10.10", "1"]]);
  const d1 = diffEvent(1001, 1005, [["10.00", "2"]]);
  const gapDiff = diffEvent(1010, 1012, [["10.00", "999999"]]);
  const anotherPostGapDiff = diffEvent(1013, 1013, [["10.00", "888888"]]);

  const { states: statesBeforeResync, issues } = reconstructDepthBook([snapshot, d1, gapDiff, anotherPostGapDiff]);
  const beforeResync = statesBeforeResync.get(keyOf("binance", "BNBBTC"))!;
  assertEquals(beforeResync.status, "INVALID");
  assertEquals(beforeResync.bids[0], { price: "10.00", size: "2" }, "ignored post-invalidation diffs must never be applied");
  assertEquals(issues.filter((i) => i.kind === "diff_ignored_while_invalid").length, 1);

  const freshSnapshot = snapshotEvent(2000, [["11.00", "5"]], [["11.10", "5"]]);
  const { states: statesAfterResync } = reconstructDepthBook([snapshot, d1, gapDiff, anotherPostGapDiff, freshSnapshot]);
  const afterResync = statesAfterResync.get(keyOf("binance", "BNBBTC"))!;
  assertEquals(afterResync.status, "SYNCED", "a fresh snapshot must reset synchronization from scratch and seed immediately");
  assertEquals(afterResync.bids[0], { price: "11.00", size: "5" });
});

Deno.test("reconstructDepthBook: a quantity of exactly '0' on a mutation deletes that price level", () => {
  const snapshot = snapshotEvent(1000, [["10.00", "1"], ["9.99", "2"]], [["10.10", "1"]]);
  const del = diffEvent(1001, 1001, [["9.99", "0"]]); // exact-zero deletion, not a parse fallback
  const { states } = reconstructDepthBook([snapshot, del]);
  const state = states.get(keyOf("binance", "BNBBTC"))!;
  assertEquals(state.bids.map((l) => l.price), ["10.00"], "the zero-quantity level must be removed entirely");
});

Deno.test("reconstructDepthBook: a trailing-zero-formatted zero quantity ('0.00000000') still deletes the level", () => {
  const snapshot = snapshotEvent(1000, [["10.00", "1"]], [["10.10", "1"]]);
  const del = diffEvent(1001, 1001, [["10.00", "0.00000000"]]);
  const { states } = reconstructDepthBook([snapshot, del]);
  const state = states.get(keyOf("binance", "BNBBTC"))!;
  assertEquals(state.bids.length, 0);
});

Deno.test("reconstructDepthBook: '25.10' and '25.1' address the same price level", () => {
  const snapshot = snapshotEvent(1000, [["25.10", "1"]], [["30.00", "1"]]);
  const update = diffEvent(1001, 1001, [["25.1", "7"]]); // same level, differently formatted
  const { states } = reconstructDepthBook([snapshot, update]);
  const state = states.get(keyOf("binance", "BNBBTC"))!;
  assertEquals(state.bids.length, 1, "must be recognized as the same level, not a second one");
  assertEquals(state.bids[0].size, "7");
});

// -------------------------------------------------------------------------------------------
// synchronizeDepthBookStartup -- the explicit pure primitive for Binance's real startup
// buffering procedure (fetch a snapshot, discard what it already covers, replay the rest),
// distinct from reconstructDepthBook's plain ordered-stream reducer above.
// -------------------------------------------------------------------------------------------

Deno.test("synchronizeDepthBookStartup: a snapshot older than the first buffered diff's U is rejected, requesting a refetch", () => {
  const buffered = [diffEvent(1001, 1005)];
  const snapshot = snapshotEvent(999, [["10.00", "1"]], [["10.10", "1"]]); // 999 < 1001
  const result = synchronizeDepthBookStartup(buffered, snapshot);
  assertEquals(result.outcome, "snapshot_too_old");
  assertEquals(result.state, undefined, "no book may be seeded from a snapshot that's too old to align");
  assertEquals(result.issues[0].kind, "snapshot_too_old");
});

Deno.test("synchronizeDepthBookStartup: buffered diffs entirely covered by the snapshot are discarded before replay", () => {
  const buffered = [
    diffEvent(990, 1000, [["9.00", "5"]]), // finalUpdateId 1000 <= snapshot.lastUpdateId 1000 -- covered
    diffEvent(1001, 1005, [["10.00", "2"]]), // the real first remaining event: U=1001 <= 1000+1, u=1005 > 1000
  ];
  const snapshot = snapshotEvent(1000, [["10.00", "1"]], [["10.10", "1"]]);
  const result = synchronizeDepthBookStartup(buffered, snapshot);
  assertEquals(result.outcome, "synced");
  assert(result.state);
  assertEquals(result.state.status, "SYNCED");
  assertEquals(result.state.lastAppliedUpdateId, 1005n);
  assertEquals(result.state.bids[0], { price: "10.00", size: "2" });
  assertEquals(result.issues.map((i) => i.kind), ["stale_diff_discarded"]);
});

Deno.test("synchronizeDepthBookStartup: the first remaining buffered event brackets the snapshot's lastUpdateId, then normal update rules apply to the rest", () => {
  // By construction (per Binance's documented procedure): after discarding u <= lastUpdateId,
  // the first remaining event's [U,u] range contains lastUpdateId. Here lastUpdateId=1000 and the
  // first remaining diff is U=1000,u=1002 -- U <= 1000 <= u holds. Once that's seeded, the rest
  // of the buffered stream (and any subsequent live diffs) replay through the same rules as
  // steady-state reconstructDepthBook.
  const firstRemaining = diffEvent(1000, 1002, [["10.00", "2"]]);
  const buffered = [firstRemaining];
  const snapshot = snapshotEvent(1000, [["10.00", "1"]], [["10.10", "1"]]);
  assert(BigInt(firstRemaining.firstUpdateId) <= 1000n && 1000n <= BigInt(firstRemaining.finalUpdateId), "test fixture precondition: U <= snapshot.lastUpdateId <= u");

  const subsequent = [diffEvent(1003, 1003, [["10.00", "3"]])];
  const result = synchronizeDepthBookStartup(buffered, snapshot, subsequent);
  assertEquals(result.outcome, "synced");
  assert(result.state);
  assertEquals(result.state.lastAppliedUpdateId, 1003n, "then normal update rules apply to subsequent diffs");
  assertEquals(result.state.bids[0], { price: "10.00", size: "3" });
  assertEquals(result.issues.length, 0);
});

Deno.test("synchronizeDepthBookStartup: a real gap among the buffered or subsequent diffs still invalidates the book", () => {
  const buffered = [diffEvent(999, 1001, [["10.00", "2"]])]; // U=999 <= snapshot.lastUpdateId+1, u=1001 > lastUpdateId -- applies
  const snapshot = snapshotEvent(1000, [["10.00", "1"]], [["10.10", "1"]]);
  const subsequent = [diffEvent(1010, 1012, [["10.00", "999"]])]; // gap: expected firstUpdateId 1002
  const result = synchronizeDepthBookStartup(buffered, snapshot, subsequent);
  assertEquals(result.outcome, "synced", "synchronizeDepthBookStartup itself still succeeds -- the resulting book is what's invalidated");
  assert(result.state);
  assertEquals(result.state.status, "INVALID");
  assertEquals(isDepthBookTrustworthy(result.state), false);
  assertEquals(result.state.bids[0], { price: "10.00", size: "2" }, "the post-gap diff must never be applied");
  assert(result.issues.some((i) => i.kind === "gap_invalidated_book"));
});

Deno.test("synchronizeDepthBookStartup: no buffered diffs at all -- the snapshot alone seeds the book", () => {
  const snapshot = snapshotEvent(1000, [["10.00", "1"]], [["10.10", "1"]]);
  const result = synchronizeDepthBookStartup([], snapshot);
  assertEquals(result.outcome, "synced");
  assert(result.state);
  assertEquals(result.state.status, "SYNCED");
  assertEquals(result.state.lastAppliedUpdateId, 1000n);
  assertEquals(result.issues.length, 0);
});

Deno.test("reconstructDepthBook: deterministic reconstruction of best bid/ask + top-N from snapshot + diffs, sorted correctly", () => {
  const fixtureStream = [
    snapshotEvent(1000, [["10.00", "1"], ["9.99", "1"], ["9.98", "1"]], [["10.10", "1"], ["10.11", "1"], ["10.12", "1"]]),
    diffEvent(1001, 1001, [["10.01", "3"]], [["10.09", "2"]]), // new best bid and best ask
    diffEvent(1002, 1002, [["9.98", "0"]], []), // delete a bid level
  ];

  const run1 = reconstructDepthBook(fixtureStream);
  const run2 = reconstructDepthBook([...fixtureStream]);

  for (const run of [run1, run2]) {
    const state = run.states.get(keyOf("binance", "BNBBTC"))!;
    assertEquals(state.status, "SYNCED");
    assertEquals(state.bids.map((l) => l.price), ["10.01", "10.00", "9.99"], "bids must be sorted best-first (descending)");
    assertEquals(state.asks.map((l) => l.price), ["10.09", "10.10", "10.11", "10.12"], "asks must be sorted best-first (ascending)");
    assertEquals(state.bids[0], { price: "10.01", size: "3" }, "best bid");
    assertEquals(state.asks[0], { price: "10.09", size: "2" }, "best ask");
  }
});

// Behavioral tests for the Reflex/L2 event-state foundation (brian-2026 issue #32). Every test
// here is pure/fixture-driven -- no network, no Supabase, no wall-clock dependency, since every
// timestamp is supplied explicitly by the fixture. This is what makes the "deterministic rebuild"
// test meaningful: the same fixture stream must reduce to the same state every time it's run.

import { assert, assertEquals } from "jsr:@std/assert@^1.0.0";
import {
  binanceBookTickerAdapter,
  classifyFreshness,
  classifySequence,
  keyOf,
  reduceBookEvents,
  secondVenueTickerAdapter,
  validateBookEvent,
  type NormalizedBookEvent,
  type NormalizeContext,
} from "./l2_book.ts";

function ctx(overrides: Partial<NormalizeContext> = {}): NormalizeContext {
  return { collectorReceivedAt: "2026-09-04T00:00:00.000Z", ingestAt: "2026-09-04T00:00:00.050Z", ...overrides };
}

function binanceRaw(overrides: Record<string, unknown> = {}) {
  return { u: 100, s: "BTCUSDT", b: "60000.00", B: "1.5", a: "60000.50", A: "2.0", E: 1_772_841_600_000, ...overrides };
}

// -------------------------------------------------------------------------------------------
// Adapter contract: primary venue (Binance)
// -------------------------------------------------------------------------------------------

Deno.test("binanceBookTickerAdapter: normalizes a well-formed bookTicker message", () => {
  const result = binanceBookTickerAdapter.normalize(binanceRaw(), ctx());
  assert(result.ok, "a valid, non-inverted book must normalize successfully");
  if (!result.ok) return;
  assertEquals(result.event.venue, "binance");
  assertEquals(result.event.symbol, "BTCUSDT");
  assertEquals(result.event.bestBid, 60000.00);
  assertEquals(result.event.bestAsk, 60000.50);
  assertEquals(result.event.midPrice, 60000.25);
  assertEquals(result.event.spread, 0.5);
  assertEquals(result.event.sourceSequence, "100");
  assertEquals(result.event.shadowOnly, true);
  assertEquals(result.event.evidenceClass, "PROSPECTIVE_DEVELOPMENT_SHADOW");
  assertEquals(result.event.bids, [{ price: 60000.00, size: 1.5 }], "with no explicit depth array, best bid/qty must still populate a 1-level depth");
  assertEquals(result.event.asks, [{ price: 60000.50, size: 2.0 }]);
});

Deno.test("binanceBookTickerAdapter: uses explicit depth levels when the raw message carries them, capped to depthN", () => {
  const raw = binanceRaw({
    bids: [["60000.00", "1.5"], ["59999.50", "3.0"], ["59999.00", "4.0"]],
    asks: [["60000.50", "2.0"], ["60001.00", "1.0"], ["60001.50", "5.0"]],
  });
  const result = binanceBookTickerAdapter.normalize(raw, ctx({ depthN: 2 }));
  assert(result.ok);
  if (!result.ok) return;
  assertEquals(result.event.depthN, 2);
  assertEquals(result.event.bids.length, 2);
  assertEquals(result.event.asks.length, 2);
  assertEquals(result.event.bids[0], { price: 60000.00, size: 1.5 });
});

Deno.test("binanceBookTickerAdapter: rejects a crossed/inverted book", () => {
  const result = binanceBookTickerAdapter.normalize(binanceRaw({ b: "60001.00", a: "60000.00" }), ctx());
  assertEquals(result.ok, false);
  if (result.ok) return;
  assert(result.reason.includes("inversion"), `expected an inversion rejection reason, got: ${result.reason}`);
});

Deno.test("binanceBookTickerAdapter: freshnessMs uses the exchange event time when available", () => {
  const raw = binanceRaw({ E: Date.parse("2026-09-04T00:00:00.000Z") });
  const result = binanceBookTickerAdapter.normalize(raw, ctx({ ingestAt: "2026-09-04T00:00:00.200Z" }));
  assert(result.ok);
  if (!result.ok) return;
  assertEquals(result.event.freshnessMs, 200);
});

Deno.test("binanceBookTickerAdapter: falls back to collectorReceivedAt for freshness when the venue gives no event time", () => {
  const raw = binanceRaw({ E: undefined });
  const result = binanceBookTickerAdapter.normalize(raw, ctx({
    collectorReceivedAt: "2026-09-04T00:00:00.000Z",
    ingestAt: "2026-09-04T00:00:00.075Z",
  }));
  assert(result.ok);
  if (!result.ok) return;
  assertEquals(result.event.exchangeEventAt, null);
  assertEquals(result.event.freshnessMs, 75);
});

// -------------------------------------------------------------------------------------------
// Adapter contract: second venue -- proves the boundary is a real seam, not Binance-only.
// -------------------------------------------------------------------------------------------

Deno.test("secondVenueTickerAdapter: normalizes a structurally different raw shape into the same contract", () => {
  const raw = {
    product: "BTC-USD",
    time: "2026-09-04T00:00:00.030Z",
    bestBid: { price: "60000.10", size: "0.8" },
    bestAsk: { price: "60000.60", size: "1.2" },
    sequence: 555,
  };
  const result = secondVenueTickerAdapter.normalize(raw, ctx({ collectorReceivedAt: "2026-09-04T00:00:00.000Z", ingestAt: "2026-09-04T00:00:00.080Z" }));
  assert(result.ok);
  if (!result.ok) return;
  assertEquals(result.event.venue, "coinbase");
  assertEquals(result.event.symbol, "BTC-USD");
  assertEquals(result.event.sourceSequence, "555");
  assertEquals(result.event.bestBid, 60000.10);
  assertEquals(result.event.bestAsk, 60000.60);
  assertEquals(result.event.freshnessMs, 50, "freshness must be measured from the venue's own exchange event time, not collectorReceivedAt");
});

Deno.test("secondVenueTickerAdapter: rejects a crossed/inverted book just like the primary venue adapter", () => {
  const raw = {
    product: "BTC-USD",
    time: "2026-09-04T00:00:00.030Z",
    bestBid: { price: "60001.00", size: "0.8" },
    bestAsk: { price: "60000.00", size: "1.2" },
  };
  const result = secondVenueTickerAdapter.normalize(raw, ctx());
  assertEquals(result.ok, false);
});

// -------------------------------------------------------------------------------------------
// validateBookEvent
// -------------------------------------------------------------------------------------------

Deno.test("validateBookEvent: a normal book with bestBid < bestAsk is valid", () => {
  assertEquals(validateBookEvent({ bestBid: 10, bestAsk: 10.1 }).valid, true);
});

Deno.test("validateBookEvent: bestBid == bestAsk is an inversion (a crossed/locked book must not be trusted as current)", () => {
  const result = validateBookEvent({ bestBid: 10, bestAsk: 10 });
  assertEquals(result.valid, false);
  assert(result.reason?.includes("inversion"));
});

Deno.test("validateBookEvent: non-positive prices are rejected", () => {
  assertEquals(validateBookEvent({ bestBid: 0, bestAsk: 10 }).valid, false);
  assertEquals(validateBookEvent({ bestBid: -5, bestAsk: 10 }).valid, false);
});

// -------------------------------------------------------------------------------------------
// classifySequence / classifyFreshness
// -------------------------------------------------------------------------------------------

Deno.test("classifySequence: the first event for a key (no prior sequence) is 'expected'", () => {
  assertEquals(classifySequence(null, 100n), "expected");
});

Deno.test("classifySequence: consecutive sequence is 'expected'", () => {
  assertEquals(classifySequence(100n, 101n), "expected");
});

Deno.test("classifySequence: an equal sequence is 'duplicate'", () => {
  assertEquals(classifySequence(100n, 100n), "duplicate");
});

Deno.test("classifySequence: a lower sequence than the last applied is 'out_of_order'", () => {
  assertEquals(classifySequence(100n, 99n), "out_of_order");
});

Deno.test("classifySequence: a jump of more than one is a 'gap'", () => {
  assertEquals(classifySequence(100n, 105n), "gap");
});

Deno.test("classifySequence: no current sequence at all is 'no_sequence', regardless of prior", () => {
  assertEquals(classifySequence(100n, null), "no_sequence");
  assertEquals(classifySequence(null, null), "no_sequence");
});

Deno.test("classifyFreshness: at or under the threshold is FRESH, strictly over it is STALE", () => {
  assertEquals(classifyFreshness(5000, 5000), "FRESH");
  assertEquals(classifyFreshness(5001, 5000), "STALE");
  assertEquals(classifyFreshness(0, 5000), "FRESH");
});

// -------------------------------------------------------------------------------------------
// reduceBookEvents: the required behavioral scenarios
// -------------------------------------------------------------------------------------------

function event(overrides: Partial<NormalizedBookEvent> = {}): NormalizedBookEvent {
  return {
    venue: "binance",
    symbol: "BTCUSDT",
    exchangeEventAt: "2026-09-04T00:00:00.000Z",
    collectorReceivedAt: "2026-09-04T00:00:00.010Z",
    ingestAt: "2026-09-04T00:00:00.020Z",
    sourceSequence: "1",
    bestBid: 60000,
    bestAsk: 60000.5,
    midPrice: 60000.25,
    spread: 0.5,
    bids: [{ price: 60000, size: 1 }],
    asks: [{ price: 60000.5, size: 1 }],
    depthN: 5,
    sourceLineage: {},
    freshnessMs: 20,
    evidenceClass: "PROSPECTIVE_DEVELOPMENT_SHADOW",
    shadowOnly: true,
    ...overrides,
  };
}

Deno.test("reduceBookEvents: ordered updates advance state to the latest event, with no issues", () => {
  const events = [
    event({ sourceSequence: "1", bestBid: 100, bestAsk: 100.1 }),
    event({ sourceSequence: "2", bestBid: 101, bestAsk: 101.1 }),
    event({ sourceSequence: "3", bestBid: 102, bestAsk: 102.1 }),
  ];
  const { states, issues } = reduceBookEvents(events);
  assertEquals(issues.length, 0);
  const state = states.get(keyOf("binance", "BTCUSDT"));
  assert(state);
  assertEquals(state!.event.sourceSequence, "3");
  assertEquals(state!.event.bestBid, 102);
  assertEquals(state!.appliedCount, 3);
  assertEquals(state!.lastSequence, 3n);
});

Deno.test("reduceBookEvents: a duplicate update id is flagged and does not change state", () => {
  const events = [
    event({ sourceSequence: "1", bestBid: 100 }),
    event({ sourceSequence: "2", bestBid: 101 }),
    event({ sourceSequence: "2", bestBid: 999 }), // stale replay of an already-applied id
  ];
  const { states, issues } = reduceBookEvents(events);
  const state = states.get(keyOf("binance", "BTCUSDT"))!;
  assertEquals(state.event.bestBid, 101, "the duplicate must not overwrite the already-applied state");
  assertEquals(state.appliedCount, 2);
  assertEquals(issues.length, 1);
  assertEquals(issues[0].kind, "duplicate");
});

Deno.test("reduceBookEvents: an out-of-order update (behind the last applied sequence) is flagged and does not change state", () => {
  const events = [
    event({ sourceSequence: "1", bestBid: 100 }),
    event({ sourceSequence: "2", bestBid: 101 }),
    event({ sourceSequence: "1", bestBid: 999 }), // arrives late, behind sequence 2 which already applied -- and not equal to it, so it's out_of_order rather than a duplicate
  ];
  const { states, issues } = reduceBookEvents(events);
  const state = states.get(keyOf("binance", "BTCUSDT"))!;
  assertEquals(state.event.bestBid, 101, "a straggler must not regress state that a newer sequence already advanced");
  assertEquals(state.appliedCount, 2);
  assertEquals(issues.length, 1);
  assertEquals(issues[0].kind, "out_of_order");
});

Deno.test("reduceBookEvents: a sequence gap is flagged but the event is still applied (state keeps moving forward)", () => {
  const events = [
    event({ sourceSequence: "1", bestBid: 100 }),
    event({ sourceSequence: "5", bestBid: 105 }), // jumped from 1 -> 5
  ];
  const { states, issues } = reduceBookEvents(events);
  const state = states.get(keyOf("binance", "BTCUSDT"))!;
  assertEquals(state.event.sourceSequence, "5", "a gap must not freeze state -- the newer event is still the current book");
  assertEquals(state.appliedCount, 2);
  assertEquals(issues.length, 1);
  assertEquals(issues[0].kind, "gap");
});

Deno.test("reduceBookEvents: freshness is classified per event and reflected on the resulting state", () => {
  const events = [
    event({ sourceSequence: "1", freshnessMs: 100 }),
    event({ sourceSequence: "2", freshnessMs: 9000 }), // beyond the default 5000ms threshold
  ];
  const { states } = reduceBookEvents(events);
  const state = states.get(keyOf("binance", "BTCUSDT"))!;
  assertEquals(state.freshness, "STALE");
});

Deno.test("reduceBookEvents: a custom staleAfterMs threshold is honored", () => {
  const events = [event({ sourceSequence: "1", freshnessMs: 200 })];
  const { states } = reduceBookEvents(events, { staleAfterMs: 100 });
  const state = states.get(keyOf("binance", "BTCUSDT"))!;
  assertEquals(state.freshness, "STALE");
});

Deno.test("reduceBookEvents: bid/ask inversion is rejected as defense-in-depth, even if it somehow reaches the reducer directly", () => {
  const events = [
    event({ sourceSequence: "1", bestBid: 100, bestAsk: 100.1 }),
    event({ sourceSequence: "2", bestBid: 101, bestAsk: 100.5 }), // inverted: bid > ask
  ];
  const { states, issues } = reduceBookEvents(events);
  const state = states.get(keyOf("binance", "BTCUSDT"))!;
  assertEquals(state.event.sourceSequence, "1", "an inverted event must never become current state");
  assertEquals(state.appliedCount, 1);
  assertEquals(issues.length, 1);
  assertEquals(issues[0].kind, "invalid");
});

Deno.test("reduceBookEvents: an event with no sourceSequence is always applied and does not reset the sequence baseline", () => {
  const events = [
    event({ sourceSequence: "1", bestBid: 100 }),
    event({ sourceSequence: null, bestBid: 150 }), // e.g. a venue message with no update id
    event({ sourceSequence: "2", bestBid: 101 }), // must still be compared against sequence 1, not against null
  ];
  const { states, issues } = reduceBookEvents(events);
  const state = states.get(keyOf("binance", "BTCUSDT"))!;
  assertEquals(state.event.bestBid, 101);
  assertEquals(state.appliedCount, 3, "the no-sequence event must be applied, not ignored");
  assertEquals(issues.length, 0, "sequence 2 following sequence 1 must still read as 'expected', not a gap, even after an unsequenced event in between");
});

Deno.test("reduceBookEvents: distinct (venue, symbol) keys are tracked independently", () => {
  const events = [
    event({ venue: "binance", symbol: "BTCUSDT", sourceSequence: "1", bestBid: 100 }),
    event({ venue: "coinbase", symbol: "BTC-USD", sourceSequence: "1", bestBid: 200 }),
    event({ venue: "binance", symbol: "BTCUSDT", sourceSequence: "2", bestBid: 101 }),
  ];
  const { states, issues } = reduceBookEvents(events);
  assertEquals(issues.length, 0);
  assertEquals(states.get(keyOf("binance", "BTCUSDT"))!.event.bestBid, 101);
  assertEquals(states.get(keyOf("coinbase", "BTC-USD"))!.event.bestBid, 200);
});

Deno.test("reduceBookEvents: deterministic rebuild -- the same fixture stream reduces to the same state every time", () => {
  const fixtureStream: NormalizedBookEvent[] = [
    event({ venue: "binance", symbol: "BTCUSDT", sourceSequence: "10", bestBid: 100, freshnessMs: 50 }),
    event({ venue: "binance", symbol: "BTCUSDT", sourceSequence: "11", bestBid: 101, freshnessMs: 40 }),
    event({ venue: "binance", symbol: "BTCUSDT", sourceSequence: "11", bestBid: 999, freshnessMs: 40 }), // duplicate, must be ignored
    event({ venue: "binance", symbol: "BTCUSDT", sourceSequence: "15", bestBid: 105, freshnessMs: 30 }), // gap
    event({ venue: "binance", symbol: "BTCUSDT", sourceSequence: "13", bestBid: 999, freshnessMs: 30 }), // out of order (behind 15)
    event({ venue: "binance", symbol: "BTCUSDT", sourceSequence: "16", bestBid: 106, bestAsk: 106.05, freshnessMs: 9000 }), // stale
    event({ venue: "coinbase", symbol: "BTC-USD", sourceSequence: "1", bestBid: 200, freshnessMs: 10 }),
  ];

  const run1 = reduceBookEvents(fixtureStream);
  const run2 = reduceBookEvents([...fixtureStream]); // fresh array, same contents -- proves no shared mutable state leaks between calls

  for (const run of [run1, run2]) {
    const btc = run.states.get(keyOf("binance", "BTCUSDT"))!;
    assertEquals(btc.event.sourceSequence, "16");
    assertEquals(btc.event.bestBid, 106);
    assertEquals(btc.appliedCount, 4, "10, 11, 15 (gap), 16 applied; the duplicate 11 and out-of-order 13 are not");
    assertEquals(btc.freshness, "STALE");
    assertEquals(btc.lastSequence, 16n);

    const coinbase = run.states.get(keyOf("coinbase", "BTC-USD"))!;
    assertEquals(coinbase.event.bestBid, 200);
    assertEquals(coinbase.appliedCount, 1);

    const kinds = run.issues.map((i) => i.kind).sort();
    assertEquals(kinds, ["duplicate", "gap", "out_of_order"]);
  }

  assertEquals(run1.states.size, run2.states.size);
});

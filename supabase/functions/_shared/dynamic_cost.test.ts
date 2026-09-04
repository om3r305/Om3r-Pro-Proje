import { assert, assertEquals, assertThrows } from "jsr:@std/assert@1";
import { compileDegradedTopOfBookCost, compileL2Cost } from "./dynamic_cost.ts";

Deno.test("dynamic cost: single ask level has zero depth slippage", () => {
  const q = compileL2Cost({
    side: "BUY", notionalUsd: 50, feeBps: 10,
    bids: [{ price: "99.90", size: "10" }],
    asks: [{ price: "100.10", size: "10" }],
  });
  assert(q.fillable);
  assertEquals(q.levelsConsumed, 1);
  assert(Math.abs((q.vwap ?? 0) - 100.10) < 1e-10);
  assert(Math.abs(q.depthSlippageBps) < 1e-10);
  assert(q.oneWayCostBps > 10);
  assertEquals(q.quality, "L2_OBSERVED");
});

Deno.test("dynamic cost: BUY walks multiple ask levels and charges depth slippage", () => {
  const q = compileL2Cost({
    side: "BUY", notionalUsd: 250, feeBps: 10,
    bids: [{ price: "99.90", size: "10" }],
    asks: [
      { price: "100.10", size: "1" },
      { price: "100.20", size: "1" },
      { price: "100.40", size: "10" },
    ],
  });
  assert(q.fillable);
  assertEquals(q.levelsConsumed, 3);
  assert((q.vwap ?? 0) > 100.10);
  assert(q.depthSlippageBps > 0);
  assert(q.oneWayCostBps > q.feeBps + q.halfSpreadBps);
});

Deno.test("dynamic cost: SELL walks bids in descending price order", () => {
  const q = compileL2Cost({
    side: "SELL", notionalUsd: 180, feeBps: 8,
    bids: [
      { price: "99.70", size: "10" },
      { price: "99.90", size: "1" },
      { price: "99.80", size: "1" },
    ],
    asks: [{ price: "100.10", size: "10" }],
  });
  assert(q.fillable);
  assertEquals(q.levelsConsumed, 2);
  assert((q.vwap ?? 0) < 99.90);
  assert(q.depthSlippageBps > 0);
});

Deno.test("dynamic cost: insufficient visible depth fails closed", () => {
  const q = compileL2Cost({
    side: "BUY", notionalUsd: 1000, feeBps: 10,
    bids: [{ price: "99", size: "1" }],
    asks: [{ price: "101", size: "1" }],
  });
  assert(!q.fillable);
  assert(q.fillRatio > 0 && q.fillRatio < 1);
  assertEquals(q.reason, "visible L2 depth is insufficient for requested notional");
});

Deno.test("dynamic cost: crossed book is rejected", () => {
  assertThrows(() => compileL2Cost({
    side: "BUY", notionalUsd: 100, feeBps: 10,
    bids: [{ price: "101", size: "1" }],
    asks: [{ price: "100", size: "1" }],
  }), Error, "crossed L2 book");
});

Deno.test("dynamic cost: malformed/non-string venue decimals are rejected", () => {
  assertThrows(() => compileL2Cost({
    side: "BUY", notionalUsd: 100, feeBps: 10,
    bids: [{ price: "100", size: "1" }],
    asks: [{ price: "bad", size: "1" }],
  }), Error, "finite positive");

  assertThrows(() => compileL2Cost({
    side: "BUY", notionalUsd: 100, feeBps: 10,
    bids: [{ price: "100", size: "1" }],
    asks: [{ price: 101 as unknown as string, size: "1" }],
  }), Error, "preserve venue decimal strings");
});

Deno.test("dynamic cost: degraded fallback is explicit and conservative", () => {
  const q = compileDegradedTopOfBookCost({
    side: "BUY", notionalUsd: 500, feeBps: 10, spreadBps: 4, assumedSlippageBps: 3, midPrice: 100,
  });
  assertEquals(q.quality, "DEGRADED_TOP_OF_BOOK");
  assertEquals(q.oneWayCostBps, 15);
  assertEquals(q.estimatedRoundTripCostBps, 30);
  assert(q.fillable);
});

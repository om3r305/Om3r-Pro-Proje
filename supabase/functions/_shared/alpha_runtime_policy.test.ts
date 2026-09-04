import { assert, assertEquals } from "jsr:@std/assert@1";
import { compileL2Cost } from "./dynamic_cost.ts";
import { selectAlphaRuntimeCost } from "./alpha_runtime_policy.ts";

Deno.test("invalid observed L2 can never degrade into fillable top-of-book", () => {
  const quote = selectAlphaRuntimeCost({
    l2Status: "INVALID",
    referenceBook: { mid: 100, spreadBps: 2 },
    side: "BUY",
    notionalUsd: 10,
    feeBps: 10,
    fallbackSlippageBps: 1,
  });
  assertEquals(quote, null);
});

Deno.test("transport-unavailable L2 may use explicit degraded top-of-book", () => {
  const quote = selectAlphaRuntimeCost({
    l2Status: "UNAVAILABLE",
    referenceBook: { mid: 100, spreadBps: 2 },
    side: "BUY",
    notionalUsd: 10,
    feeBps: 10,
    fallbackSlippageBps: 1,
  });
  assert(quote);
  assertEquals(quote.quality, "DEGRADED_TOP_OF_BOOK");
  assertEquals(quote.fillable, true);
});

Deno.test("observed thin L2 remains unfillable and is never replaced by fallback", () => {
  const observed = compileL2Cost({
    side: "BUY",
    notionalUsd: 100,
    feeBps: 10,
    bids: [{ price: "99", size: "10" }],
    asks: [{ price: "101", size: "0.1" }],
  });
  assertEquals(observed.fillable, false);
  const selected = selectAlphaRuntimeCost({
    l2Status: "OBSERVED",
    observedL2Quote: observed,
    referenceBook: { mid: 100, spreadBps: 2 },
    side: "BUY",
    notionalUsd: 100,
    feeBps: 10,
    fallbackSlippageBps: 1,
  });
  assert(selected);
  assertEquals(selected.quality, "L2_OBSERVED");
  assertEquals(selected.fillable, false);
});

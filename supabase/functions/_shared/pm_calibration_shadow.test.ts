import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import { calibratedYes, decideEntry, fromRawBook, marketFeeRate, PM_CALIBRATION_RULES, pmCalibrationRulesFingerprint, returnPerDollar } from "./pm_calibration_shadow.ts";
import { classifyPolymarketDomain } from "./polymarket_domain.ts";

Deno.test("calibration map stretches prices away from 50% (slope > 1) and is monotone", () => {
  assert(calibratedYes(0.2) < 0.2 + 0.02 && calibratedYes(0.8) > 0.8);
  let prev = 0;
  for (let p = 0.01; p <= 0.99; p += 0.01) { const q = calibratedYes(p); assert(q > prev); prev = q; }
  assertAlmostEquals(calibratedYes(0.5), 1 / (1 + Math.exp(-PM_CALIBRATION_RULES.intercept)), 1e-12);
  assertEquals(calibratedYes(0), calibratedYes(0.01));
});

Deno.test("entry buys the side whose calibrated value beats its ask by the frozen margin", () => {
  // p = 0.80 -> q ~ 0.85: YES at 0.81 has edge ~0.04, NO at 0.21 is negative.
  const yes = { bid: 0.79, ask: 0.81, askShares: 50 }, no = { bid: 0.19, ask: 0.21, askShares: 50 };
  const d = decideEntry(yes, no, 0);
  assertEquals(d.action, "BUY_YES");
  if (d.action === "BUY_YES") { assert(d.edge >= PM_CALIBRATION_RULES.minEdge); assertEquals(d.price, 0.81); assertEquals(d.fee, 0); }
  // Low price: q < p, so NO is the cheap side.
  const d2 = decideEntry({ bid: 0.14, ask: 0.15, askShares: 50 }, { bid: 0.85, ask: 0.86, askShares: 50 }, 0);
  assert(d2.action === "BUY_NO" || d2.action === "SKIP");
});

Deno.test("fails closed: unknown fee, one-sided or wide book, thin ask, edge below margin", () => {
  const good = { bid: 0.79, ask: 0.81, askShares: 50 }, no = { bid: 0.19, ask: 0.21, askShares: 50 };
  assertEquals(decideEntry(good, no, null).action, "SKIP");
  assertEquals((decideEntry({ bid: null, ask: 0.81, askShares: 50 }, no, 0) as { reason: string }).reason, "YES_BOOK_ONE_SIDED");
  assertEquals((decideEntry({ bid: 0.6, ask: 0.81, askShares: 50 }, no, 0) as { reason: string }).reason, "SPREAD_TOO_WIDE");
  assertEquals((decideEntry({ ...good, askShares: 1 }, { ...no, askShares: 1 }, 0) as { reason: string }).reason, "NO_USABLE_ASK");
  // p ~ 0.30 is (almost) the fixed point of the frozen map: neither side clears the margin.
  assertEquals((decideEntry({ bid: 0.29, ask: 0.31, askShares: 50 }, { bid: 0.69, ask: 0.71, askShares: 50 }, 0) as { reason: string }).reason, "EDGE_BELOW_MIN");
  // A fee large enough removes the edge.
  assertEquals(decideEntry(good, no, 0.9).action, "SKIP");
});

Deno.test("return per dollar: win pays 1/(price+fee)-1, loss is -1", () => {
  assertAlmostEquals(returnPerDollar("BUY_YES", 0.8, 0, 1), 0.25, 1e-12);
  assertEquals(returnPerDollar("BUY_YES", 0.8, 0, 0), -1);
  assertAlmostEquals(returnPerDollar("BUY_NO", 0.2, 0.01, 0), 1 / 0.21 - 1, 1e-12);
});

Deno.test("book parser takes best levels and total size at the best ask; fee only from the schedule", () => {
  const top = fromRawBook({ bids: [{ price: "0.40", size: "10" }, { price: "0.41", size: "3" }], asks: [{ price: "0.45", size: "7" }, { price: "0.43", size: "2" }, { price: "0.43", size: "4" }, { price: "x", size: "1" }] });
  assertEquals(top, { bid: 0.41, ask: 0.43, askShares: 6 });
  assertEquals(fromRawBook(null), { bid: null, ask: null, askShares: null });
  assertEquals(marketFeeRate({ feesEnabled: false }), 0);
  assertEquals(marketFeeRate({ feesEnabled: true, feeSchedule: { rate: 0.05 } }), 0.05);
  assertEquals(marketFeeRate({ feesEnabled: true }), null);
  assertEquals(marketFeeRate({}), null);
});

Deno.test("politics classifier (generated) matches the paper's first-match order", () => {
  assertEquals(classifyPolymarketDomain("Will Trump win the 2028 election?"), "Politics");
  assertEquals(classifyPolymarketDomain("Lakers vs Celtics: who wins game 3?"), "Sports");
  assertEquals(classifyPolymarketDomain(""), "Other");
});

Deno.test("rules fingerprint is stable and 64 hex", async () => {
  const a = await pmCalibrationRulesFingerprint(), b = await pmCalibrationRulesFingerprint();
  assertEquals(a, b);
  assert(/^[0-9a-f]{64}$/.test(a));
});

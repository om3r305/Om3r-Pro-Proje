import { assertAlmostEquals, assertEquals, assertThrows } from "jsr:@std/assert@1";
import {
  assetSignal,
  isRebalanceDay,
  MAX_ASSET_WEIGHT,
  MIN_HISTORY,
  realisedVol,
  sma,
  step,
  targetWeights,
  ZERO_WEIGHTS,
} from "./trend_shadow.ts";

// Deterministic series: geometric trend with an alternating +/- wiggle so vol > 0.
function series(n: number, drift: number, wiggle = 0.01, start = 100): number[] {
  const out = [start];
  for (let i = 1; i < n; i++) out.push(out[i - 1] * (1 + drift + (i % 2 ? wiggle : -wiggle)));
  return out;
}

Deno.test("sma averages the last L closes", () => {
  assertEquals(sma([1, 2, 3, 4, 5], 2), 4.5);
  assertThrows(() => sma([1, 2], 3));
});

Deno.test("realisedVol matches pandas sample std of returns, annualised", () => {
  // returns alternate +10% / -10% -> mean 0, sample std = 0.1 * sqrt(n/(n-1))
  const closes = [100];
  for (let i = 1; i <= 30; i++) closes.push(closes[i - 1] * (i % 2 ? 1.1 : 0.9));
  const expected = Math.sqrt((30 * 0.01) / 29) * Math.sqrt(365);
  assertAlmostEquals(realisedVol(closes), expected, 1e-12);
});

Deno.test("uptrend is on for every lookback and weight is the vol cap", () => {
  const s = assetSignal(series(MIN_HISTORY + 10, 0.003));
  assertEquals(Object.values(s.on), [true, true, true]);
  assertAlmostEquals(s.weight, s.cap, 1e-12);
  assertEquals(s.cap <= MAX_ASSET_WEIGHT, true);
});

Deno.test("downtrend holds no weight", () => {
  const s = assetSignal(series(MIN_HISTORY + 10, -0.003));
  assertEquals(Object.values(s.on), [false, false, false]);
  assertEquals(s.weight, 0);
});

Deno.test("high volatility scales weight below the per-asset cap", () => {
  const calm = assetSignal(series(MIN_HISTORY + 10, 0.003, 0.002));
  const wild = assetSignal(series(MIN_HISTORY + 10, 0.003, 0.06));
  assertEquals(calm.cap, MAX_ASSET_WEIGHT);
  assertEquals(wild.cap < MAX_ASSET_WEIGHT, true);
  assertAlmostEquals(wild.cap, MAX_ASSET_WEIGHT * 0.40 / wild.vol, 1e-12);
});

Deno.test("rejects short or invalid history", () => {
  assertThrows(() => assetSignal(series(MIN_HISTORY - 1, 0.001)));
  const bad = series(MIN_HISTORY, 0.001);
  bad[5] = 0;
  assertThrows(() => assetSignal(bad));
});

Deno.test("rebalance day is Monday UTC", () => {
  assertEquals(isRebalanceDay("2026-09-28"), true); // Monday
  assertEquals(isRebalanceDay("2026-09-29"), false);
});

Deno.test("first step rebalances from cash and pays entry cost only", () => {
  const closes = { BTCUSDT: series(MIN_HISTORY + 5, 0.003), ETHUSDT: series(MIN_HISTORY + 5, -0.003) };
  const s = step("2026-09-29", null, closes);
  assertEquals(s.rebalanced, true);
  assertEquals(s.day_return, 0);
  assertEquals(s.target.ETHUSDT, 0);
  assertAlmostEquals(s.cost, s.target.BTCUSDT * 0.0015, 1e-15);
  assertAlmostEquals(s.nav, 10_000 * (1 - s.cost), 1e-9);
});

Deno.test("non-Monday step holds weights, earns held return, no cost", () => {
  const closes = { BTCUSDT: series(MIN_HISTORY + 5, 0.003), ETHUSDT: series(MIN_HISTORY + 5, 0.002) };
  const held = { BTCUSDT: 0.3, ETHUSDT: 0.2 };
  const s = step("2026-09-30", { held, nav: 12_000 }, closes);
  assertEquals(s.rebalanced, false);
  assertEquals(s.target, held);
  assertEquals(s.cost, 0);
  const b = closes.BTCUSDT, e = closes.ETHUSDT;
  const r = 0.3 * (b.at(-1)! / b.at(-2)! - 1) + 0.2 * (e.at(-1)! / e.at(-2)! - 1);
  assertAlmostEquals(s.day_return, r, 1e-15);
  assertAlmostEquals(s.nav, 12_000 * (1 + r), 1e-9);
});

Deno.test("Monday step moves to new target and charges turnover", () => {
  const closes = { BTCUSDT: series(MIN_HISTORY + 5, -0.003), ETHUSDT: series(MIN_HISTORY + 5, -0.003) };
  const s = step("2026-10-05", { held: { BTCUSDT: 0.4, ETHUSDT: 0.1 }, nav: 10_000 }, closes);
  assertEquals(s.rebalanced, true);
  assertEquals(s.target, ZERO_WEIGHTS);
  assertAlmostEquals(s.turnover, 0.5, 1e-15);
  assertAlmostEquals(s.cost, 0.5 * 0.0015, 1e-15);
});

Deno.test("targetWeights covers both assets", () => {
  const { weights } = targetWeights({ BTCUSDT: series(MIN_HISTORY, 0.002), ETHUSDT: series(MIN_HISTORY, 0.002) });
  assertEquals(Object.keys(weights).sort(), ["BTCUSDT", "ETHUSDT"]);
});

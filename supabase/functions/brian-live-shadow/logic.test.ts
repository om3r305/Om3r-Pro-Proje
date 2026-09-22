// Behavioral tests for the actual deployed brian-live-shadow feature/allocation/accounting logic.
// Run with: deno test --allow-read supabase/functions/brian-live-shadow

import { assertAlmostEquals, assertEquals, assertThrows } from "jsr:@std/assert@^1.0.0";
import {
  alignedFrames, type AllocationConfig, type Bar, type Book, chooseAllocation, clip, dot,
  type DrawdownConfig, executionCost, featureMap, finite, GYM, grossBudget, markAndDrift, mean,
  type ModelWeights, safeLogRatio, std, SYMBOLS, type SymbolName, type TickState,
} from "./logic.ts";

function bar(closeTime: number, close: number, opts: Partial<Bar> = {}): Bar {
  return { closeTime, open: close, high: close * 1.01, low: close * 0.99, close, volume: 100, ...opts };
}
function allSymbolBars(closeTime: number, close: number, overrides: Partial<Record<SymbolName, Bar>> = {}): Record<SymbolName, Bar> {
  const out = {} as Record<SymbolName, Bar>;
  for (const s of SYMBOLS) out[s] = overrides[s] ?? bar(closeTime, close);
  return out;
}

Deno.test("clip: clamps to a custom [low, high] range", () => {
  assertEquals(clip(-5, 0, 10), 0);
  assertEquals(clip(5, 0, 10), 5);
  assertEquals(clip(15, 0, 10), 10);
});

Deno.test("mean/std: empty and single-element inputs do not divide by zero", () => {
  assertEquals(mean([]), 0);
  assertEquals(std([]), 0);
  assertEquals(std([3]), 0);
});

Deno.test("safeLogRatio: requires both sides strictly positive", () => {
  assertAlmostEquals(safeLogRatio(110, 100), Math.log(1.1));
  assertThrows(() => safeLogRatio(0, 100));
  assertThrows(() => safeLogRatio(100, -1));
});

Deno.test("finite: rejects non-finite market values instead of coercing", () => {
  assertEquals(finite("42.5"), 42.5);
  assertThrows(() => finite("not a number"));
  assertThrows(() => finite(undefined));
});

Deno.test("dot: standard dot product", () => {
  assertEquals(dot([1, 2, 3], [4, 5, 6]), 32);
});

Deno.test("alignedFrames: keeps only closeTimes common to every symbol, sorted ascending", () => {
  const bars = {} as Record<SymbolName, Bar[]>;
  for (const s of SYMBOLS) bars[s] = [bar(200, 100), bar(100, 100), bar(300, 100)];
  // BTCUSDT is missing closeTime 300, so only 100 and 200 are common to all symbols.
  bars.BTCUSDT = [bar(100, 100), bar(200, 100)];
  const frames = alignedFrames(bars);
  assertEquals(frames.map((f) => f.closeTime), [100, 200]);
});

Deno.test("featureMap: requires at least two visible frames", () => {
  const frames = [{ closeTime: 100, bars: allSymbolBars(100, 100) }];
  assertThrows(() => featureMap(frames, 5));
});

Deno.test("featureMap: bias term is 1, and a symbol that outperforms the market gets a positive relative feature", () => {
  const frames = [0, 1, 2, 3, 4].map((i) => ({ closeTime: i, bars: allSymbolBars(i, 100) }));
  // Only BTCUSDT jumps on the final (current) frame; every other symbol stays flat at 100.
  frames.push({
    closeTime: 5,
    bars: { ...allSymbolBars(5, 100), BTCUSDT: bar(5, 110) },
  });
  const features = featureMap(frames, 5);
  for (const s of SYMBOLS) assertEquals(features[s].length, 10);
  for (const s of SYMBOLS) assertEquals(features[s][0], 1);
  assertEquals(features.BTCUSDT[6] > 0, true); // outperformed the market average
  assertEquals(features.ETHUSDT[6] < 0, true); // flat, so underperformed the BTC-pulled average
});

const DD_CFG: DrawdownConfig = { drawdown_flatten: 0.5, drawdown_throttle_2: 0.3, drawdown_throttle_1: 0.15, max_gross_exposure: 0.75 };

Deno.test("grossBudget: steps down in three stages as drawdown from starting equity deepens, then flattens", () => {
  assertEquals(grossBudget(GYM.starting_equity, DD_CFG), 0.75); // no drawdown
  assertEquals(grossBudget(400, DD_CFG), 0.375); // dd=0.2 -> throttle_1 -> 0.5x
  assertEquals(grossBudget(325, DD_CFG), 0.1875); // dd=0.35 -> throttle_2 -> 0.25x
  assertEquals(grossBudget(200, DD_CFG), 0); // dd=0.6 -> flatten
});

const ALLOC_CFG: AllocationConfig = {
  ...DD_CFG, max_label_abs: 0.25, min_uncertainty: 0.0015, risk_aversion: 0.75,
  min_weighted_samples_per_asset: 12, min_abs_edge: 0.0015, turnover_penalty_bps: 15,
  max_positions: 3, max_asset_weight: 0.25,
};

function zeroFeatures(): number[] {
  return new Array(10).fill(0);
}
function pickFirstModel(): ModelWeights {
  return { weights: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], weighted_samples: 50, error_ewma: 0.01 };
}
function passiveModel(): ModelWeights {
  return { weights: zeroFeatures(), weighted_samples: 50, error_ewma: 0 };
}
function allSymbolFeatures(overrides: Partial<Record<SymbolName, number[]>> = {}): Record<SymbolName, number[]> {
  const out = {} as Record<SymbolName, number[]>;
  for (const s of SYMBOLS) out[s] = overrides[s] ?? zeroFeatures();
  return out;
}
function allSymbolModels(overrides: Partial<Record<SymbolName, ModelWeights>> = {}): Record<SymbolName, ModelWeights> {
  const out = {} as Record<SymbolName, ModelWeights>;
  for (const s of SYMBOLS) out[s] = overrides[s] ?? passiveModel();
  return out;
}

Deno.test("chooseAllocation: a strong, well-supported prediction opens a full-strength, budget-respecting position", () => {
  const features = allSymbolFeatures({ BTCUSDT: [0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0] });
  const models = allSymbolModels({ BTCUSDT: pickFirstModel() });
  const allocation = chooseAllocation("NATIVE", features, {}, GYM.starting_equity, models, ALLOC_CFG);
  assertAlmostEquals(allocation.weights.BTCUSDT, 0.25);
  assertEquals(Object.keys(allocation.weights).length, 1);
});

Deno.test("chooseAllocation: below min_weighted_samples_per_asset, even a strong prediction is excluded", () => {
  const features = allSymbolFeatures({ BTCUSDT: [0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0] });
  const models = allSymbolModels({ BTCUSDT: { ...pickFirstModel(), weighted_samples: 5 } });
  const allocation = chooseAllocation("NATIVE", features, {}, GYM.starting_equity, models, ALLOC_CFG);
  assertEquals(allocation.weights, {});
});

Deno.test("chooseAllocation: deep drawdown zeroes the tradeable budget regardless of edge strength", () => {
  const features = allSymbolFeatures({ BTCUSDT: [0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0] });
  const models = allSymbolModels({ BTCUSDT: pickFirstModel() });
  const deeplyDrawndownEquity = GYM.starting_equity * (1 - 0.6); // dd=0.6 >= drawdown_flatten
  const allocation = chooseAllocation("NATIVE", features, {}, deeplyDrawndownEquity, models, ALLOC_CFG);
  assertEquals(allocation.weights, {});
});

Deno.test("chooseAllocation: PROFIT never opens a larger position than NATIVE would for the same raw edge (cost-aware sizing)", () => {
  const features = allSymbolFeatures({ BTCUSDT: [0.01, 0, 0, 0, 0, 0, 0, 0, 0, 0] });
  const models = allSymbolModels({ BTCUSDT: { weights: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], weighted_samples: 50, error_ewma: 0.0015 } });
  const native = chooseAllocation("NATIVE", features, {}, GYM.starting_equity, models, ALLOC_CFG);
  const profit = chooseAllocation("PROFIT", features, {}, GYM.starting_equity, models, ALLOC_CFG);
  const nativeMagnitude = Math.abs(native.weights.BTCUSDT ?? 0);
  const profitMagnitude = Math.abs(profit.weights.BTCUSDT ?? 0);
  assertEquals(profitMagnitude <= nativeMagnitude, true);
});

Deno.test("markAndDrift: no prior tick starts flat at the configured starting equity", () => {
  const result = markAndDrift(null, {} as Record<SymbolName, Book>);
  assertEquals(result, { equityAfterMark: GYM.starting_equity, drifted: {}, periodPnl: 0 });
});

Deno.test("markAndDrift: marks a prior position to the current book and computes the period P&L", () => {
  const previous: TickState = {
    observed_at: "2026-09-19T00:00:00.000Z", equity_after_costs: 500, peak_equity_after: 500,
    max_drawdown_pct_after: 0, target_weights: { BTCUSDT: 0.5 }, observed_mid_prices: { BTCUSDT: 100 },
  };
  const books = { BTCUSDT: { bid: 109, ask: 111, mid: 110, spreadBps: 10 } } as Record<SymbolName, Book>;
  const result = markAndDrift(previous, books);
  assertAlmostEquals(result.equityAfterMark, 525);
  assertAlmostEquals(result.periodPnl, 25);
  assertAlmostEquals(result.drifted.BTCUSDT, (0.5 * 500 * 1.1) / 525);
});

Deno.test("markAndDrift: throws rather than silently dropping a position with no current mark price", () => {
  const previous: TickState = {
    observed_at: "2026-09-19T00:00:00.000Z", equity_after_costs: 500, peak_equity_after: 500,
    max_drawdown_pct_after: 0, target_weights: { BTCUSDT: 0.5 }, observed_mid_prices: { BTCUSDT: 100 },
  };
  assertThrows(() => markAndDrift(previous, {} as Record<SymbolName, Book>));
});

Deno.test("executionCost: fee + slippage + half-spread cost, proportional to weight turnover and equity", () => {
  const books = { BTCUSDT: { bid: 99.9, ask: 100.1, mid: 100, spreadBps: 10 } } as Record<SymbolName, Book>;
  const result = executionCost(500, {}, { BTCUSDT: 0.25 }, books);
  assertAlmostEquals(result.turnoverNotional, 125);
  assertAlmostEquals(result.cost, 0.2);
});

Deno.test("executionCost: zero-delta assets are skipped, so a missing spread there does not throw", () => {
  const result = executionCost(500, { ETHUSDT: 0.1 }, { ETHUSDT: 0.1 }, {} as Record<SymbolName, Book>);
  assertEquals(result, { turnoverNotional: 0, cost: 0 });
});

Deno.test("executionCost: a moved asset with no book entry fails closed instead of pricing it as free", () => {
  assertThrows(() => executionCost(500, {}, { BTCUSDT: 0.1 }, {} as Record<SymbolName, Book>));
});

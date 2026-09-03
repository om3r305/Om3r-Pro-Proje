// Behavioral tests that execute the actual deployed TS logic (not text/grep assertions).
// Run with: deno test --allow-read supabase/functions/brian-intrabar-eye
//
// The consensus/late-chase cases below are loaded from tests/fixtures/intrabar_consensus_cases.json,
// which is also consumed by tests/test_brian2026_phase40_intrabar_reaction.py so the Python reference
// model and this deployed TS logic are checked against the same scenarios. Cases marked
// shared_semantics:false are the one currently-known, documented divergence (brian-2026 issue #32
// Sec 3.4) -- they assert each implementation's real, current behavior, not a reconciled target.

import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@^1.0.0";
import {
  type AggTrade,
  type Bar,
  buildIntrabarConsensus,
  clip,
  computeExtensionSigma,
  FEE_BPS,
  type IntrabarSignalRow,
  type MarketRow,
  microTick,
  signalBreakout,
  signalReclaim,
  signalTakerFlow,
  signalVelocity,
  signalVolumeBurst,
  SLIPPAGE_BPS,
} from "./logic.ts";

const SCORE_TOLERANCE = 1e-9;

interface FixtureObservation {
  independent_group: string;
  direction: number;
  strength: number;
  confidence: number;
  reliability: number;
}
interface FixtureExpectation {
  direction: number;
  eligible: boolean;
  late_chase?: boolean;
  status?: string;
  score?: number;
  support_groups?: string[];
  conflict_groups?: string[];
}
interface FixtureCase {
  name: string;
  shared_semantics: boolean;
  observations: FixtureObservation[];
  extension_sigma: number;
  decelerating: boolean;
  expected?: FixtureExpectation;
  ts_expected?: FixtureExpectation;
}
interface FixtureFile {
  cases: FixtureCase[];
}

async function loadFixtures(): Promise<FixtureFile> {
  const url = new URL("../../../tests/fixtures/intrabar_consensus_cases.json", import.meta.url);
  const text = await Deno.readTextFile(url);
  return JSON.parse(text) as FixtureFile;
}

function toSignalRows(observations: FixtureObservation[]): IntrabarSignalRow[] {
  return observations.map((o) => ({
    independentGroup: o.independent_group,
    direction: o.direction,
    strength: o.strength,
    confidence: o.confidence,
    reliability: o.reliability,
  }));
}

function assertOutcomeMatches(outcome: ReturnType<typeof buildIntrabarConsensus>, expected: FixtureExpectation, caseName: string) {
  assertEquals(outcome.direction, expected.direction, `${caseName}: direction`);
  assertEquals(outcome.eligible, expected.eligible, `${caseName}: eligible`);
  if (expected.late_chase !== undefined) assertEquals(outcome.lateChase, expected.late_chase, `${caseName}: late_chase`);
  if (expected.status !== undefined) assertEquals(outcome.status, expected.status, `${caseName}: status`);
  if (expected.score !== undefined) assertAlmostEquals(outcome.score, expected.score, SCORE_TOLERANCE, `${caseName}: score`);
  if (expected.support_groups !== undefined) assertEquals(outcome.supportGroups, expected.support_groups, `${caseName}: support_groups`);
  if (expected.conflict_groups !== undefined) assertEquals(outcome.conflictGroups, expected.conflict_groups, `${caseName}: conflict_groups`);
}

const CONSENSUS_PARAMS_BASE = { minSupportGroups: 2, minConsensusScore: 0.18, overextensionSigma: 3.5 };

Deno.test("buildIntrabarConsensus matches shared fixtures (Python + TS)", async (t) => {
  const fixtures = await loadFixtures();
  for (const testCase of fixtures.cases) {
    await t.step(testCase.name, () => {
      const rows = toSignalRows(testCase.observations);
      const outcome = buildIntrabarConsensus(rows, {
        ...CONSENSUS_PARAMS_BASE,
        extensionSigma: testCase.extension_sigma,
        decelerating: testCase.decelerating,
        spreadBps: 4,
      });
      const expected = testCase.shared_semantics ? testCase.expected : testCase.ts_expected;
      assert(expected, `${testCase.name}: fixture is missing its expected block for this implementation`);
      assertOutcomeMatches(outcome, expected, testCase.name);
    });
  }
});

Deno.test("buildIntrabarConsensus: empty evidence is a neutral WATCH, not a throw", () => {
  const outcome = buildIntrabarConsensus([], { ...CONSENSUS_PARAMS_BASE, extensionSigma: 0, decelerating: false, spreadBps: 4 });
  assertEquals(outcome.direction, 0);
  assertEquals(outcome.eligible, false);
  assertEquals(outcome.status, "WATCH");
});

Deno.test("computeExtensionSigma matches the documented five-bar realized-vol-normalized formula", () => {
  const closes = [100, 100, 100, 100, 100, 100];
  const baseline: Bar[] = closes.map((close, i) => ({
    openTime: i, closeTime: i, open: close, high: close, low: close, close,
    baseVolume: 0, quoteVolume: 0, trades: 0, takerBuyQuote: 0,
  }));
  const sigma = 0.01;
  const mid = 105;
  const expected = Math.abs(Math.log(mid / baseline.at(-5)!.close)) / (sigma * Math.sqrt(5));
  assertAlmostEquals(computeExtensionSigma(baseline, mid, sigma), expected, 1e-12);
});

Deno.test("computeExtensionSigma falls back to the first bar when fewer than five are available", () => {
  const baseline: Bar[] = [100, 101].map((close, i) => ({
    openTime: i, closeTime: i, open: close, high: close, low: close, close,
    baseVolume: 0, quoteVolume: 0, trades: 0, takerBuyQuote: 0,
  }));
  const expected = Math.abs(Math.log(103 / 100)) / (0.02 * Math.sqrt(5));
  assertAlmostEquals(computeExtensionSigma(baseline, 103, 0.02), expected, 1e-12);
});

function makeBar(overrides: Partial<Bar> = {}): Bar {
  return { openTime: 0, closeTime: 60_000, open: 100, high: 100, low: 100, close: 100, baseVolume: 0, quoteVolume: 0, trades: 0, takerBuyQuote: 0, ...overrides };
}

Deno.test("signalTakerFlow: strong buy imbalance with sufficient pace is directional", () => {
  const row = {
    current: makeBar({ quoteVolume: 100_000, takerBuyQuote: 70_000 }),
    elapsedFraction: 1,
    medianQuoteVolume: 50_000,
  } as unknown as MarketRow;
  const signal = signalTakerFlow(row);
  assertEquals(signal.direction, 1);
  assertAlmostEquals(signal.strength, 0.4, 1e-9);
});

Deno.test("signalTakerFlow: balanced buy/sell share produces no signal even at high pace", () => {
  const row = {
    current: makeBar({ quoteVolume: 100_000, takerBuyQuote: 50_000 }),
    elapsedFraction: 1,
    medianQuoteVolume: 50_000,
  } as unknown as MarketRow;
  const signal = signalTakerFlow(row);
  assertEquals(signal.direction, 0);
  assertEquals(signal.strength, 0);
});

Deno.test("FEE_BPS/SLIPPAGE_BPS: today's production cost constants are pinned", () => {
  assertEquals(FEE_BPS, 10.0);
  assertEquals(SLIPPAGE_BPS, 1.0);
});

Deno.test("microTick: cost-aware two-tick equity/drawdown sequence", () => {
  const spreadBps = 4;
  const oneWay = (FEE_BPS + SLIPPAGE_BPS + spreadBps / 2) / 10000;

  const tick1 = microTick({
    eyeId: "eye-1", templateId: "velocity-micro", assetId: "crypto:BTCUSDT", observedAt: "2026-01-01T00:00:00.000Z",
    mid: 100, spreadBps, strength: 0.8, confidence: 0.9, targetDirection: 1, startingTicket: 5, rawCaptureId: "cap-1",
    metadata: {},
  });
  assert(tick1);
  const expectedCost1 = 5 * 1 * oneWay;
  assertAlmostEquals(tick1.starting_equity as number, 5, 1e-9);
  assertAlmostEquals(tick1.equity_before as number, 5, 1e-9);
  assertAlmostEquals(tick1.period_pnl as number, 0, 1e-9);
  assertAlmostEquals(tick1.trading_cost as number, expectedCost1, 1e-9);
  assertAlmostEquals(tick1.equity_after as number, 5 - expectedCost1, 1e-9);
  assertAlmostEquals(tick1.peak_equity_after as number, 5, 1e-9);
  assertAlmostEquals(tick1.max_drawdown_pct_after as number, 100 * expectedCost1 / 5, 1e-9);
  assertEquals(tick1.target_direction, 1);

  const tick2 = microTick({
    eyeId: "eye-1", templateId: "velocity-micro", assetId: "crypto:BTCUSDT", observedAt: "2026-01-01T00:01:00.000Z",
    mid: 101, spreadBps, strength: 0.8, confidence: 0.9, targetDirection: 1, startingTicket: 5, rawCaptureId: "cap-2",
    // deno-lint-ignore no-explicit-any
    prior: tick1 as any,
    metadata: {},
  });
  assert(tick2);
  const equityBefore2 = tick1.equity_after as number;
  const expectedPnl2 = equityBefore2 * 1 * (101 / 100 - 1);
  const marked2 = equityBefore2 + expectedPnl2;
  assertAlmostEquals(tick2.starting_equity as number, 5, 1e-9);
  assertAlmostEquals(tick2.equity_before as number, equityBefore2, 1e-9);
  assertAlmostEquals(tick2.period_pnl as number, expectedPnl2, 1e-9);
  assertAlmostEquals(tick2.trading_cost as number, 0, 1e-9);
  assertAlmostEquals(tick2.equity_after as number, marked2, 1e-9);
  assertAlmostEquals(tick2.peak_equity_after as number, marked2, 1e-9);
  assertAlmostEquals(tick2.max_drawdown_pct_after as number, tick1.max_drawdown_pct_after as number, 1e-9);
  assertEquals(tick2.prior_direction, 1);
});

Deno.test("microTick: no prior and flat target produces no tick", () => {
  const tick = microTick({
    eyeId: "eye-1", templateId: "velocity-micro", assetId: "crypto:BTCUSDT", observedAt: "2026-01-01T00:00:00.000Z",
    mid: 100, spreadBps: 4, strength: 0, confidence: 0.9, targetDirection: 0, startingTicket: 5, rawCaptureId: "cap-1",
    metadata: {},
  });
  assertEquals(tick, null);
});

Deno.test("microTick: exit from a held long position charges full (turnover=1) cost", () => {
  const spreadBps = 4;
  const oneWay = (FEE_BPS + SLIPPAGE_BPS + spreadBps / 2) / 10000;

  const enter = microTick({
    eyeId: "eye-2", templateId: "velocity-micro", assetId: "crypto:BTCUSDT", observedAt: "2026-01-01T00:00:00.000Z",
    mid: 100, spreadBps, strength: 0.8, confidence: 0.9, targetDirection: 1, startingTicket: 5, rawCaptureId: "cap-1",
    metadata: {},
  });
  assert(enter);

  const exit = microTick({
    eyeId: "eye-2", templateId: "velocity-micro", assetId: "crypto:BTCUSDT", observedAt: "2026-01-01T00:01:00.000Z",
    mid: 99, spreadBps, strength: 0, confidence: 0.9, targetDirection: 0, startingTicket: 5, rawCaptureId: "cap-2",
    // deno-lint-ignore no-explicit-any
    prior: enter as any,
    metadata: {},
  });
  assert(exit);
  const equityBefore = enter.equity_after as number;
  const expectedPnl = equityBefore * 1 * (99 / 100 - 1);
  const marked = equityBefore + expectedPnl;
  const expectedCost = marked * 1 * oneWay; // turnover = |0 - 1| = 1
  assertAlmostEquals(exit.equity_before as number, equityBefore, 1e-9);
  assertAlmostEquals(exit.period_pnl as number, expectedPnl, 1e-9);
  assertAlmostEquals(exit.trading_cost as number, expectedCost, 1e-9);
  assertAlmostEquals(exit.equity_after as number, Math.max(0, marked - expectedCost), 1e-9);
  assertEquals(exit.target_direction, 0);
  assertEquals(exit.prior_direction, 1);

  const stayFlat = microTick({
    eyeId: "eye-2", templateId: "velocity-micro", assetId: "crypto:BTCUSDT", observedAt: "2026-01-01T00:02:00.000Z",
    mid: 99, spreadBps, strength: 0, confidence: 0.9, targetDirection: 0, startingTicket: 5, rawCaptureId: "cap-3",
    // deno-lint-ignore no-explicit-any
    prior: exit as any,
    metadata: {},
  });
  assertEquals(stayFlat, null, "once flat, staying flat must not produce another tick");
});

Deno.test("microTick: +1 -> -1 flip charges full flip (turnover=2) cost", () => {
  const spreadBps = 4;
  const oneWay = (FEE_BPS + SLIPPAGE_BPS + spreadBps / 2) / 10000;

  const enter = microTick({
    eyeId: "eye-3", templateId: "velocity-micro", assetId: "crypto:BTCUSDT", observedAt: "2026-01-01T00:00:00.000Z",
    mid: 100, spreadBps, strength: 0.8, confidence: 0.9, targetDirection: 1, startingTicket: 5, rawCaptureId: "cap-1",
    metadata: {},
  });
  assert(enter);

  const flip = microTick({
    eyeId: "eye-3", templateId: "velocity-micro", assetId: "crypto:BTCUSDT", observedAt: "2026-01-01T00:01:00.000Z",
    mid: 102, spreadBps, strength: 0.8, confidence: 0.9, targetDirection: -1, startingTicket: 5, rawCaptureId: "cap-2",
    // deno-lint-ignore no-explicit-any
    prior: enter as any,
    metadata: {},
  });
  assert(flip);
  const equityBefore = enter.equity_after as number;
  const expectedPnl = equityBefore * 1 * (102 / 100 - 1);
  const marked = equityBefore + expectedPnl;
  const expectedCost = marked * 2 * oneWay; // turnover = |-1 - 1| = 2
  assertAlmostEquals(flip.equity_before as number, equityBefore, 1e-9);
  assertAlmostEquals(flip.period_pnl as number, expectedPnl, 1e-9);
  assertAlmostEquals(flip.trading_cost as number, expectedCost, 1e-9);
  assertAlmostEquals(flip.equity_after as number, Math.max(0, marked - expectedCost), 1e-9);
  assertEquals(flip.target_direction, -1);
  assertEquals(flip.prior_direction, 1);
});

// --- signalVelocity ---------------------------------------------------------

function makeTrade(ts: number, price: number): AggTrade {
  return { price, qty: 1, quote: price, ts, buyerMaker: false };
}

const OBSERVED_MS = 1_700_000_000_000;

Deno.test("signalVelocity: insufficient coverage is neutral regardless of trades", () => {
  const row = {
    velocityCoverageSeconds: 2,
    trades: [makeTrade(OBSERVED_MS - 1_000, 100), makeTrade(OBSERVED_MS, 105)],
    sigma: 0.001,
  } as unknown as MarketRow;
  const signal = signalVelocity(row, OBSERVED_MS);
  assertEquals(signal.direction, 0);
  assertEquals(signal.strength, 0);
});

Deno.test("signalVelocity: move below the adaptive threshold is neutral", () => {
  const sigma = 0.01;
  const coverage = 30;
  const trades = [makeTrade(OBSERVED_MS - 20_000, 100), makeTrade(OBSERVED_MS - 1_000, 100.05)];
  const live = Math.log(100.05 / 100);
  const adaptive = Math.max(0.00010, sigma * Math.sqrt(30 / 60) * 0.70);
  assert(Math.abs(live) < adaptive, "fixture must land strictly below the adaptive threshold");
  const row = { velocityCoverageSeconds: coverage, trades, sigma } as unknown as MarketRow;
  const signal = signalVelocity(row, OBSERVED_MS);
  assertEquals(signal.direction, 0);
  assertEquals(signal.strength, 0);
});

Deno.test("signalVelocity: clear positive impulse is directional with strength in (0,1]", () => {
  const sigma = 0.01;
  const coverage = 30;
  const trades = [makeTrade(OBSERVED_MS - 20_000, 100), makeTrade(OBSERVED_MS - 1_000, 101)];
  const live = Math.log(101 / 100);
  const adaptive = Math.max(0.00010, sigma * Math.sqrt(30 / 60) * 0.70);
  assert(Math.abs(live) >= adaptive, "fixture must clear the adaptive threshold");
  const row = { velocityCoverageSeconds: coverage, trades, sigma } as unknown as MarketRow;
  const signal = signalVelocity(row, OBSERVED_MS);
  const expectedStrength = clip(Math.abs(live) / (adaptive * 3));
  assertEquals(signal.direction, 1);
  assert(signal.strength > 0 && signal.strength <= 1, "strength must be bounded to (0,1]");
  assertAlmostEquals(signal.strength, expectedStrength, 1e-9);
});

Deno.test("signalVelocity: clear negative impulse is directional with strength in (0,1]", () => {
  const sigma = 0.01;
  const coverage = 30;
  const trades = [makeTrade(OBSERVED_MS - 20_000, 100), makeTrade(OBSERVED_MS - 1_000, 99)];
  const live = Math.log(99 / 100);
  const adaptive = Math.max(0.00010, sigma * Math.sqrt(30 / 60) * 0.70);
  assert(Math.abs(live) >= adaptive, "fixture must clear the adaptive threshold");
  const row = { velocityCoverageSeconds: coverage, trades, sigma } as unknown as MarketRow;
  const signal = signalVelocity(row, OBSERVED_MS);
  const expectedStrength = clip(Math.abs(live) / (adaptive * 3));
  assertEquals(signal.direction, -1);
  assert(signal.strength > 0 && signal.strength <= 1, "strength must be bounded to (0,1]");
  assertAlmostEquals(signal.strength, expectedStrength, 1e-9);
});

// --- signalVolumeBurst -------------------------------------------------------

Deno.test("signalVolumeBurst: low pace and flat body is not impulsive", () => {
  const row = {
    current: makeBar({ quoteVolume: 100_000, open: 100 }),
    elapsedFraction: 1,
    medianQuoteVolume: 100_000,
    book: { mid: 100, spreadBps: 4 },
    sigma: 0.001,
  } as unknown as MarketRow;
  const pace = 100_000 / Math.max(1, 0.15) / Math.max(100_000, 1e-12);
  assert(pace < 1.8, "fixture must stay below the pace gate");
  const signal = signalVolumeBurst(row);
  assertEquals(signal.direction, 0);
  assertEquals(signal.strength, 0);
});

Deno.test("signalVolumeBurst: relative-volume pace plus body impulse is directional", () => {
  const sigma = 0.001;
  const row = {
    current: makeBar({ quoteVolume: 250_000, open: 100 }),
    elapsedFraction: 1,
    medianQuoteVolume: 100_000,
    book: { mid: 102, spreadBps: 4 },
    sigma,
  } as unknown as MarketRow;
  const pace = 250_000 / Math.max(1, 0.15) / Math.max(100_000, 1e-12);
  const body = Math.log(102 / 100);
  const threshold = Math.max(0.00015, sigma * 0.45);
  assert(pace >= 1.8 && Math.abs(body) >= threshold, "fixture must clear both the pace and body gates");
  const signal = signalVolumeBurst(row);
  const expectedStrength = clip(0.55 * clip(Math.abs(body) / (threshold * 3)) + 0.45 * clip((pace - 1) / 4));
  assertEquals(signal.direction, 1);
  assertAlmostEquals(signal.strength, expectedStrength, 1e-9);
});

// --- signalBreakout -----------------------------------------------------------

function makeBaseline(n: number, high: number, low: number, close: number): Bar[] {
  return Array.from({ length: n }, (_, i) => makeBar({ openTime: i, closeTime: i, high, low, close, open: close }));
}

Deno.test("signalBreakout: mid inside the prior range produces no break", () => {
  const row = {
    baseline: makeBaseline(10, 101, 99, 100),
    book: { mid: 100, spreadBps: 4 },
    sigma: 0.001,
  } as unknown as MarketRow;
  const signal = signalBreakout(row);
  assertEquals(signal.direction, 0);
  assertEquals(signal.strength, 0);
});

Deno.test("signalBreakout: mid above the buffered prior high is a bullish break", () => {
  const sigma = 0.001;
  const spreadBps = 4;
  const mid = 101.05;
  const row = { baseline: makeBaseline(10, 101, 99, 100), book: { mid, spreadBps }, sigma } as unknown as MarketRow;
  const oneWay = (FEE_BPS + SLIPPAGE_BPS + spreadBps / 2) / 10000;
  const buffer = Math.max(0.00012, oneWay * 0.30, sigma * 0.30);
  assert(mid > 101 * (1 + buffer), "fixture must clear the buffered prior high");
  const signal = signalBreakout(row);
  const dist = Math.log(mid / 101);
  const expectedStrength = clip(dist / Math.max(buffer * 4, sigma * 2));
  assertEquals(signal.direction, 1);
  assertAlmostEquals(signal.strength, expectedStrength, 1e-9);
});

Deno.test("signalBreakout: mid below the buffered prior low is a bearish break", () => {
  const sigma = 0.001;
  const spreadBps = 4;
  const mid = 98.9;
  const row = { baseline: makeBaseline(10, 101, 99, 100), book: { mid, spreadBps }, sigma } as unknown as MarketRow;
  const oneWay = (FEE_BPS + SLIPPAGE_BPS + spreadBps / 2) / 10000;
  const buffer = Math.max(0.00012, oneWay * 0.30, sigma * 0.30);
  assert(mid < 99 * (1 - buffer), "fixture must clear the buffered prior low");
  const signal = signalBreakout(row);
  const dist = Math.log(99 / mid);
  const expectedStrength = clip(dist / Math.max(buffer * 4, sigma * 2));
  assertEquals(signal.direction, -1);
  assertAlmostEquals(signal.strength, expectedStrength, 1e-9);
});

// --- signalReclaim --------------------------------------------------------------

function makeReclaimBars(n: number, low: number, high: number, close: number): Bar[] {
  return Array.from({ length: n }, (_, i) => makeBar({ openTime: i, closeTime: i, low, high, close, open: close }));
}

Deno.test("signalReclaim: flat range with no sweep is neutral", () => {
  const row = {
    bars: makeReclaimBars(13, 99, 101, 100),
    current: makeBar({ high: 100.2, low: 99.8 }),
    book: { mid: 100, spreadBps: 4 },
    sigma: 0.001,
  } as unknown as MarketRow;
  const signal = signalReclaim(row);
  assertEquals(signal.direction, 0);
  assertEquals(signal.strength, 0);
});

Deno.test("signalReclaim: same-minute downside sweep reclaimed is bullish", () => {
  const sigma = 0.001;
  const bars = makeReclaimBars(13, 99, 101, 100); // context = bars.slice(-13,-1) -> priorLow=99, priorHigh=101
  const buffer = Math.max(0.00010, sigma * 0.20);
  const currentLow = 99 * (1 - buffer) - 0.05;
  const currentHigh = 99.5;
  const mid = 99.8;
  const currentMidpoint = (currentHigh + currentLow) / 2;
  assert(currentLow < 99 * (1 - buffer), "fixture must sweep below the buffered prior low");
  assert(mid > 99 && mid > currentMidpoint, "fixture must reclaim above the prior low and the current bar's midpoint");
  const row = {
    bars, current: makeBar({ high: currentHigh, low: currentLow }), book: { mid, spreadBps: 4 }, sigma,
  } as unknown as MarketRow;
  const signal = signalReclaim(row);
  const sweep = Math.log(99 / currentLow);
  const reclaim = Math.log(mid / 99);
  const expectedStrength = clip((sweep + reclaim) / Math.max(sigma * 3, buffer * 5));
  assertEquals(signal.direction, 1);
  assertAlmostEquals(signal.strength, expectedStrength, 1e-9);
});

Deno.test("signalReclaim: same-minute upside sweep rejected is bearish", () => {
  const sigma = 0.001;
  const bars = makeReclaimBars(13, 99, 101, 100);
  const buffer = Math.max(0.00010, sigma * 0.20);
  const currentHigh = 101 * (1 + buffer) + 0.05;
  const currentLow = 100.8;
  const mid = 100.85;
  const currentMidpoint = (currentHigh + currentLow) / 2;
  assert(currentLow >= 99 * (1 - buffer), "fixture must not also qualify as a downside sweep");
  assert(currentHigh > 101 * (1 + buffer), "fixture must sweep above the buffered prior high");
  assert(mid < 101 && mid < currentMidpoint, "fixture must reject below the prior high and the current bar's midpoint");
  const row = {
    bars, current: makeBar({ high: currentHigh, low: currentLow }), book: { mid, spreadBps: 4 }, sigma,
  } as unknown as MarketRow;
  const signal = signalReclaim(row);
  const sweep = Math.log(currentHigh / 101);
  const reclaim = Math.log(101 / mid);
  const expectedStrength = clip((sweep + reclaim) / Math.max(sigma * 3, buffer * 5));
  assertEquals(signal.direction, -1);
  assertAlmostEquals(signal.strength, expectedStrength, 1e-9);
});

Deno.test("signalReclaim: prior-minute downside sweep reclaimed is bullish", () => {
  const sigma = 0.005;
  const buffer = Math.max(0.00010, sigma * 0.20);
  const beforePrevious = makeReclaimBars(11, 99, 101, 100);
  const previous = makeBar({ low: 98.85, high: 100, close: 99.5 });
  const currentPlaceholder = makeBar({ low: 99.6, high: 100.2, close: 100 });
  const bars = [...beforePrevious, previous, currentPlaceholder]; // length 13
  const mid = 99.6;
  const current = makeBar({ high: 100.5, low: 99.5 }); // must not itself trip the same-minute branches
  const row = { bars, current, book: { mid, spreadBps: 4 }, sigma } as unknown as MarketRow;

  // Branch order matters: confirm the same-minute (context = bars.slice(-13,-1)) branches are skipped first.
  const contextPriorLow = Math.min(99, previous.low);
  const contextPriorHigh = Math.max(101, previous.high);
  assert(!(current.low < contextPriorLow * (1 - buffer)), "current bar must not trip the same-minute downside branch");
  assert(!(current.high > contextPriorHigh * (1 + buffer)), "current bar must not trip the same-minute upside branch");

  assert(previous.low < 99 * (1 - buffer), "fixture must sweep below the beforePrevious low");
  assert(mid > 99 && mid > previous.close, "fixture must reclaim above the beforePrevious low and the previous bar's close");

  const signal = signalReclaim(row);
  const expectedStrength = clip(Math.log(mid / Math.max(previous.low, 1e-12)) / Math.max(sigma * 4, buffer * 6));
  assertEquals(signal.direction, 1);
  assertAlmostEquals(signal.strength, expectedStrength, 1e-9);
});

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
  type Bar,
  buildIntrabarConsensus,
  computeExtensionSigma,
  FEE_BPS,
  type IntrabarSignalRow,
  type MarketRow,
  microTick,
  SLIPPAGE_BPS,
  signalTakerFlow,
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

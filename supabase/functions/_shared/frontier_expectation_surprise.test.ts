import {
  assessIncorporation,
  assessResolvedSurprise,
  rankHorizonCandidates,
  type FrozenExpectationSnapshot,
  type LearnedIncorporationCurve,
} from "./frontier_expectation_surprise.ts";
import type { ExpectationDistribution } from "./frontier_market_belief.ts";

function belief(mean: number, at: string): ExpectationDistribution {
  return {
    metric: "event_probability",
    units: "probability",
    horizon: "HOURS",
    mean,
    uncertainty: 0.1,
    generatedAt: at,
    evidenceKinds: ["PREDICTION_MARKET", "OPTIONS"],
    evidenceRefs: ["belief:1"],
  };
}

function frozen(): FrozenExpectationSnapshot {
  return {
    eventId: "event:1",
    eventFamily: "GEOPOLITICAL_ESCALATION",
    instrumentId: "BTCUSDT",
    frozenAt: "2026-09-12T08:59:00Z",
    brianBeliefP: belief(0.7, "2026-09-12T08:58:30Z"),
    marketBeliefQ: belief(0.45, "2026-09-12T08:58:40Z"),
    sourceRefs: ["primary:1"],
    positioningRefs: ["positioning:1"],
    liquidityRefs: ["liquidity:1"],
  };
}

Deno.test("Frontier surprise keeps P-Q separate from realized X-star", () => {
  const result = assessResolvedSurprise(frozen(), {
    metric: "event_probability",
    units: "probability",
    value: 1,
    resolvedAt: "2026-09-12T09:00:00Z",
    resolutionRef: "resolution:1",
  });
  if (!result.legal) throw new Error(JSON.stringify(result));
  if (Math.abs((result.preEventBeliefResidual ?? 0) - 0.25) > 1e-9) throw new Error(JSON.stringify(result));
  if (Math.abs((result.marketSurprise ?? 0) - 0.55) > 1e-9) throw new Error(JSON.stringify(result));
  if (Math.abs((result.brianForecastError ?? 0) - 0.3) > 1e-9) throw new Error(JSON.stringify(result));
});

Deno.test("Frontier surprise rejects Q generated after event resolution", () => {
  const input = frozen();
  input.marketBeliefQ = belief(0.9, "2026-09-12T09:00:10Z");
  input.frozenAt = "2026-09-12T09:00:10Z";
  const result = assessResolvedSurprise(input, {
    metric: "event_probability",
    units: "probability",
    value: 1,
    resolvedAt: "2026-09-12T09:00:00Z",
    resolutionRef: "resolution:1",
  });
  if (result.legal) throw new Error(JSON.stringify(result));
  if (!result.reasons.includes("EXPECTATION_SNAPSHOT_AFTER_RESOLUTION")) throw new Error(JSON.stringify(result));
});

function curve(): LearnedIncorporationCurve {
  return {
    eventFamily: "GEOPOLITICAL_ESCALATION",
    instrumentId: "BTCUSDT",
    horizon: "HOURS",
    learnedAt: "2026-09-12T08:00:00Z",
    medianLagSeconds: 300,
    p90LagSeconds: 1200,
    expectedMoveAbsBps: 100,
    sampleSize: 120,
    regimeRef: "regime:1",
  };
}

Deno.test("Frontier flags a partially absorbed linked market inside the learned lag window", () => {
  const result = assessIncorporation(curve(), {
    eventId: "event:1",
    instrumentId: "BTCUSDT",
    eventResolvedAt: "2026-09-12T09:00:00Z",
    observedAt: "2026-09-12T09:05:00Z",
    observedMoveAbsBps: 40,
    marketBeliefShiftAbs: 0.2,
  });
  if (!result.legal || !result.lagWindowOpen || !result.potentiallyUnreacted) throw new Error(JSON.stringify(result));
  if (Math.abs((result.reactionFraction ?? 0) - 0.4) > 1e-9) throw new Error(JSON.stringify(result));
});

Deno.test("Frontier rejects incorporation model learned after the event", () => {
  const model = curve();
  model.learnedAt = "2026-09-12T09:01:00Z";
  const result = assessIncorporation(model, {
    eventId: "event:1",
    instrumentId: "BTCUSDT",
    eventResolvedAt: "2026-09-12T09:00:00Z",
    observedAt: "2026-09-12T09:05:00Z",
    observedMoveAbsBps: 40,
    marketBeliefShiftAbs: 0.2,
  });
  if (result.legal) throw new Error(JSON.stringify(result));
  if (!result.reasons.includes("INCORPORATION_MODEL_LEARNED_AFTER_EVENT")) throw new Error(JSON.stringify(result));
});

Deno.test("Horizon router rejects edge that decays before operational latency", () => {
  const result = rankHorizonCandidates([{
    horizon: "MINUTES",
    expectedResidualBps: 40,
    estimatedRoundTripCostBps: 10,
    operationalLatencySeconds: 180,
    expectedIncorporationSeconds: 120,
    evidenceMaturity: 0.9,
  }])[0];
  if (result.feasible) throw new Error(JSON.stringify(result));
  if (!result.reasons.includes("EDGE_DECAYS_BEFORE_OPERATIONAL_LATENCY")) throw new Error(JSON.stringify(result));
});

Deno.test("Horizon router ranks after-cost residual only among feasible horizons", () => {
  const ranked = rankHorizonCandidates([
    {
      horizon: "MINUTES",
      expectedResidualBps: 50,
      estimatedRoundTripCostBps: 20,
      operationalLatencySeconds: 60,
      expectedIncorporationSeconds: 600,
      evidenceMaturity: 0.8,
    },
    {
      horizon: "HOURS",
      expectedResidualBps: 45,
      estimatedRoundTripCostBps: 10,
      operationalLatencySeconds: 60,
      expectedIncorporationSeconds: 7200,
      evidenceMaturity: 0.9,
    },
    {
      horizon: "SECONDS",
      expectedResidualBps: 8,
      estimatedRoundTripCostBps: 12,
      operationalLatencySeconds: 2,
      expectedIncorporationSeconds: 20,
      evidenceMaturity: 1,
    },
  ]);
  if (ranked[0].horizon !== "HOURS" || !ranked[0].feasible) throw new Error(JSON.stringify(ranked));
  const seconds = ranked.find((row) => row.horizon === "SECONDS");
  if (!seconds || seconds.feasible || !seconds.reasons.includes("NO_POSITIVE_AFTER_COST_RESIDUAL")) throw new Error(JSON.stringify(ranked));
});

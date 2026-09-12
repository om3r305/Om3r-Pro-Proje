import {
  assessFrontierOpportunity,
  type CrowdState,
  type ExpectationDistribution,
  type FrontierOpportunityContext,
} from "./frontier_market_belief.ts";

function dist(
  mean: number,
  generatedAt: string,
  evidenceKinds: ExpectationDistribution["evidenceKinds"],
): ExpectationDistribution {
  return {
    metric: "event_probability",
    units: "probability",
    horizon: "HOURS",
    mean,
    uncertainty: 0.08,
    generatedAt,
    evidenceKinds,
    evidenceRefs: ["evidence:1"],
  };
}

function crowd(kind: CrowdState["observations"][number]["kind"]): CrowdState {
  return {
    fear: 0.75,
    uncertainty: 0.82,
    fomo: 0.18,
    capitulation: 0.31,
    euphoria: 0.05,
    crowding: 0.66,
    reflexivity: 0.58,
    forcedFlowPressure: 0.61,
    observations: [{
      observedAt: "2026-09-12T08:59:30Z",
      participant: "RETAIL",
      kind,
      strength: 0.8,
      evidenceRef: "behavior:1",
    }],
  };
}

function base(): FrontierOpportunityContext {
  return {
    decisionAt: "2026-09-12T09:00:00Z",
    eventFamily: "GEOPOLITICAL_ESCALATION",
    entityId: "EU-RU",
    instrumentId: "BTCUSDT",
    brianBeliefP: dist(0.7, "2026-09-12T08:59:00Z", ["PRIMARY_CLAIM", "BRIAN_SENSOR"]),
    marketBeliefQ: dist(0.45, "2026-09-12T08:59:20Z", ["PREDICTION_MARKET", "OPTIONS"]),
    realizedX: null,
    crowd: crowd("FLOW"),
    earliestKnowable: {
      firstPrimaryWorldAt: "2026-09-12T08:57:00Z",
      firstSourcePublishedAt: "2026-09-12T08:57:10Z",
      firstBrianObservedAt: "2026-09-12T08:57:11Z",
      firstMarketImpliedMoveAt: "2026-09-12T08:58:40Z",
      firstLegalActionableAt: "2026-09-12T08:59:25Z",
    },
    regimeRef: "regime:shadow",
    independentEvidenceGroups: 3,
    safety: {
      shadowOnly: true,
      liveExecution: false,
      canonicalAlphaMutation: false,
      dipControlled: false,
    },
  };
}

Deno.test("Frontier separates Brian belief P from market belief Q and reports residual", () => {
  const result = assessFrontierOpportunity(base());
  if (!result.eligibleForResearchSignal) throw new Error(JSON.stringify(result));
  if (Math.abs((result.beliefResidual ?? 0) - 0.25) > 1e-9) throw new Error(JSON.stringify(result));
});

Deno.test("Frontier rejects text-only psychology as a capital-relevant research signal", () => {
  const input = base();
  input.crowd = crowd("TEXT_SENTIMENT");
  const result = assessFrontierOpportunity(input);
  if (result.eligibleForResearchSignal) throw new Error(JSON.stringify(result));
  if (!result.doNotTradeReasons.includes("PSYCHOLOGY_NOT_GROUNDED_IN_MARKET_BEHAVIOR")) throw new Error(JSON.stringify(result));
});

Deno.test("Frontier rejects Q that is only stated consensus with no market anchor", () => {
  const input = base();
  input.marketBeliefQ = dist(0.45, "2026-09-12T08:59:20Z", ["STATED_CONSENSUS"]);
  const result = assessFrontierOpportunity(input);
  if (result.eligibleForResearchSignal) throw new Error(JSON.stringify(result));
  if (!result.doNotTradeReasons.includes("Q_HAS_NO_MARKET_ANCHOR")) throw new Error(JSON.stringify(result));
});

Deno.test("Frontier rejects future evidence to preserve point-in-time legality", () => {
  const input = base();
  input.brianBeliefP = dist(0.7, "2026-09-12T09:00:06Z", ["PRIMARY_CLAIM", "BRIAN_SENSOR"]);
  const result = assessFrontierOpportunity(input);
  if (result.eligibleForResearchSignal) throw new Error(JSON.stringify(result));
  if (!result.doNotTradeReasons.includes("POINT_IN_TIME_VIOLATION")) throw new Error(JSON.stringify(result));
});

Deno.test("Frontier does not combine P and Q with different horizons", () => {
  const input = base();
  input.marketBeliefQ = { ...input.marketBeliefQ, horizon: "DAYS" };
  const result = assessFrontierOpportunity(input);
  if (result.eligibleForResearchSignal) throw new Error(JSON.stringify(result));
  if (!result.doNotTradeReasons.includes("P_Q_NOT_COMPARABLE")) throw new Error(JSON.stringify(result));
});

Deno.test("Frontier requires independent evidence before psychology can become research-eligible", () => {
  const input = base();
  input.independentEvidenceGroups = 1;
  const result = assessFrontierOpportunity(input);
  if (result.eligibleForResearchSignal) throw new Error(JSON.stringify(result));
  if (!result.doNotTradeReasons.includes("INSUFFICIENT_INDEPENDENT_EVIDENCE")) throw new Error(JSON.stringify(result));
});

Deno.test("Frontier safety envelope refuses DIP control or live execution", () => {
  const input = base() as FrontierOpportunityContext & { safety: Record<string, boolean> };
  input.safety = {
    shadowOnly: true,
    liveExecution: true,
    canonicalAlphaMutation: false,
    dipControlled: false,
  };
  let threw = false;
  try {
    assessFrontierOpportunity(input as FrontierOpportunityContext);
  } catch {
    threw = true;
  }
  if (!threw) throw new Error("unsafe Frontier envelope should throw");
});

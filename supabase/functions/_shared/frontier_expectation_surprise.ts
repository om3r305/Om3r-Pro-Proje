import {
  type ExpectationDistribution,
  type FrontierHorizon,
} from "./frontier_market_belief.ts";

export const BRIAN_FRONTIER_EXPECTATION_VERSION = "brian.frontier-expectation-surprise.v0";

export interface FrozenExpectationSnapshot {
  eventId: string;
  eventFamily: string;
  instrumentId: string;
  frozenAt: string;
  brianBeliefP: ExpectationDistribution;
  marketBeliefQ: ExpectationDistribution;
  sourceRefs: string[];
  positioningRefs: string[];
  liquidityRefs: string[];
}

export interface RealizedEventState {
  metric: string;
  units: string;
  value: number;
  resolvedAt: string;
  resolutionRef: string;
}

export interface SurpriseAssessment {
  version: string;
  legal: boolean;
  reasons: string[];
  brianForecastError: number | null;
  marketSurprise: number | null;
  preEventBeliefResidual: number | null;
  surpriseSigmaVsMarket: number | null;
}

export interface LearnedIncorporationCurve {
  eventFamily: string;
  instrumentId: string;
  horizon: FrontierHorizon;
  learnedAt: string;
  medianLagSeconds: number;
  p90LagSeconds: number;
  expectedMoveAbsBps: number;
  sampleSize: number;
  regimeRef: string | null;
}

export interface IncorporationObservation {
  eventId: string;
  instrumentId: string;
  eventResolvedAt: string;
  observedAt: string;
  observedMoveAbsBps: number;
  marketBeliefShiftAbs: number;
}

export interface IncorporationAssessment {
  legal: boolean;
  reasons: string[];
  elapsedSeconds: number | null;
  lagWindowOpen: boolean;
  reactionFraction: number | null;
  potentiallyUnreacted: boolean;
}

export interface HorizonCandidate {
  horizon: FrontierHorizon;
  expectedResidualBps: number;
  estimatedRoundTripCostBps: number;
  operationalLatencySeconds: number;
  expectedIncorporationSeconds: number;
  evidenceMaturity: number;
}

export interface HorizonRanking {
  horizon: FrontierHorizon;
  feasible: boolean;
  expectedAfterCostResidualBps: number;
  score: number;
  reasons: string[];
}

const FUTURE_SKEW_MS = 5_000;

function ts(value: string): number | null {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function distributionsComparable(a: ExpectationDistribution, b: ExpectationDistribution): boolean {
  return a.metric === b.metric && a.units === b.units && a.horizon === b.horizon;
}

export function assessResolvedSurprise(
  snapshot: FrozenExpectationSnapshot,
  actual: RealizedEventState,
): SurpriseAssessment {
  const reasons: string[] = [];
  const frozenAt = ts(snapshot.frozenAt);
  const pAt = ts(snapshot.brianBeliefP.generatedAt);
  const qAt = ts(snapshot.marketBeliefQ.generatedAt);
  const resolvedAt = ts(actual.resolvedAt);

  if (frozenAt == null || pAt == null || qAt == null || resolvedAt == null) reasons.push("INVALID_TIMESTAMP");
  if (resolvedAt != null && frozenAt != null && frozenAt > resolvedAt + FUTURE_SKEW_MS) reasons.push("EXPECTATION_SNAPSHOT_AFTER_RESOLUTION");
  if (frozenAt != null && pAt != null && pAt > frozenAt + FUTURE_SKEW_MS) reasons.push("P_GENERATED_AFTER_FREEZE");
  if (frozenAt != null && qAt != null && qAt > frozenAt + FUTURE_SKEW_MS) reasons.push("Q_GENERATED_AFTER_FREEZE");
  if (!distributionsComparable(snapshot.brianBeliefP, snapshot.marketBeliefQ)) reasons.push("P_Q_NOT_COMPARABLE");
  if (actual.metric !== snapshot.marketBeliefQ.metric || actual.units !== snapshot.marketBeliefQ.units) reasons.push("ACTUAL_NOT_COMPARABLE");
  if (!Number.isFinite(actual.value)) reasons.push("ACTUAL_VALUE_INVALID");
  if (!actual.resolutionRef) reasons.push("ACTUAL_RESOLUTION_UNSOURCED");
  if (!snapshot.sourceRefs.length) reasons.push("EXPECTATION_SNAPSHOT_UNSOURCED");

  const legal = reasons.length === 0;
  const brianForecastError = legal ? actual.value - snapshot.brianBeliefP.mean : null;
  const marketSurprise = legal ? actual.value - snapshot.marketBeliefQ.mean : null;
  const preEventBeliefResidual = legal ? snapshot.brianBeliefP.mean - snapshot.marketBeliefQ.mean : null;
  const qUncertainty = snapshot.marketBeliefQ.uncertainty;
  const surpriseSigmaVsMarket = legal && qUncertainty > 0 ? marketSurprise! / qUncertainty : null;

  return {
    version: BRIAN_FRONTIER_EXPECTATION_VERSION,
    legal,
    reasons,
    brianForecastError,
    marketSurprise,
    preEventBeliefResidual,
    surpriseSigmaVsMarket,
  };
}

export function assessIncorporation(
  model: LearnedIncorporationCurve,
  observation: IncorporationObservation,
): IncorporationAssessment {
  const reasons: string[] = [];
  const learnedAt = ts(model.learnedAt);
  const eventAt = ts(observation.eventResolvedAt);
  const observedAt = ts(observation.observedAt);
  if (learnedAt == null || eventAt == null || observedAt == null) reasons.push("INVALID_TIMESTAMP");
  if (learnedAt != null && eventAt != null && learnedAt > eventAt + FUTURE_SKEW_MS) reasons.push("INCORPORATION_MODEL_LEARNED_AFTER_EVENT");
  if (eventAt != null && observedAt != null && observedAt < eventAt - FUTURE_SKEW_MS) reasons.push("OBSERVATION_BEFORE_EVENT");
  if (!Number.isFinite(model.medianLagSeconds) || model.medianLagSeconds < 0) reasons.push("MEDIAN_LAG_INVALID");
  if (!Number.isFinite(model.p90LagSeconds) || model.p90LagSeconds < model.medianLagSeconds) reasons.push("P90_LAG_INVALID");
  if (!Number.isFinite(model.expectedMoveAbsBps) || model.expectedMoveAbsBps < 0) reasons.push("EXPECTED_MOVE_INVALID");
  if (!Number.isInteger(model.sampleSize) || model.sampleSize < 1) reasons.push("INCORPORATION_SAMPLE_INVALID");
  if (!Number.isFinite(observation.observedMoveAbsBps) || observation.observedMoveAbsBps < 0) reasons.push("OBSERVED_MOVE_INVALID");
  if (!Number.isFinite(observation.marketBeliefShiftAbs) || observation.marketBeliefShiftAbs < 0) reasons.push("Q_SHIFT_INVALID");

  if (reasons.length) {
    return { legal: false, reasons, elapsedSeconds: null, lagWindowOpen: false, reactionFraction: null, potentiallyUnreacted: false };
  }

  const elapsedSeconds = Math.max(0, (observedAt! - eventAt!) / 1000);
  const lagWindowOpen = elapsedSeconds <= model.p90LagSeconds;
  const reactionFraction = model.expectedMoveAbsBps > 0
    ? Math.min(1, observation.observedMoveAbsBps / model.expectedMoveAbsBps)
    : null;
  // This is only a research label. It never authorizes capital by itself.
  const potentiallyUnreacted = lagWindowOpen && reactionFraction != null && reactionFraction < 1;
  return { legal: true, reasons, elapsedSeconds, lagWindowOpen, reactionFraction, potentiallyUnreacted };
}

export function rankHorizonCandidates(candidates: HorizonCandidate[]): HorizonRanking[] {
  return candidates.map((candidate) => {
    const reasons: string[] = [];
    if (!Number.isFinite(candidate.expectedResidualBps)) reasons.push("RESIDUAL_INVALID");
    if (!Number.isFinite(candidate.estimatedRoundTripCostBps) || candidate.estimatedRoundTripCostBps < 0) reasons.push("COST_INVALID");
    if (!Number.isFinite(candidate.operationalLatencySeconds) || candidate.operationalLatencySeconds < 0) reasons.push("LATENCY_INVALID");
    if (!Number.isFinite(candidate.expectedIncorporationSeconds) || candidate.expectedIncorporationSeconds < 0) reasons.push("INCORPORATION_INVALID");
    if (!Number.isFinite(candidate.evidenceMaturity) || candidate.evidenceMaturity < 0 || candidate.evidenceMaturity > 1) reasons.push("MATURITY_INVALID");

    const expectedAfterCostResidualBps = reasons.length
      ? Number.NEGATIVE_INFINITY
      : candidate.expectedResidualBps - candidate.estimatedRoundTripCostBps;
    if (!reasons.length && candidate.expectedIncorporationSeconds <= candidate.operationalLatencySeconds) reasons.push("EDGE_DECAYS_BEFORE_OPERATIONAL_LATENCY");
    if (!reasons.length && expectedAfterCostResidualBps <= 0) reasons.push("NO_POSITIVE_AFTER_COST_RESIDUAL");

    const feasible = reasons.length === 0;
    const score = feasible ? expectedAfterCostResidualBps * candidate.evidenceMaturity : Number.NEGATIVE_INFINITY;
    return { horizon: candidate.horizon, feasible, expectedAfterCostResidualBps, score, reasons };
  }).sort((a, b) => b.score - a.score);
}

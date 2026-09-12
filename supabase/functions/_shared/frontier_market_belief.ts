export const BRIAN_FRONTIER_CONTRACT_VERSION = "brian.frontier-market-belief.v0";

export type FrontierHorizon = "SECONDS" | "MINUTES" | "HOURS" | "DAYS" | "WEEKS";
export type ParticipantClass =
  | "RETAIL"
  | "DISCRETIONARY_MACRO"
  | "CTA_TREND"
  | "VOL_CONTROL"
  | "DEALER_MARKET_MAKER"
  | "ETF_FUND_FLOW"
  | "LONG_ONLY"
  | "CORPORATE"
  | "GOVERNMENT_REGULATOR"
  | "CENTRAL_BANK"
  | "UNKNOWN";

export type BeliefEvidenceKind =
  | "OPTIONS"
  | "FUTURES"
  | "PREDICTION_MARKET"
  | "SPOT_CURVE"
  | "POSITIONING"
  | "STATED_CONSENSUS"
  | "PRIMARY_CLAIM"
  | "BRIAN_SENSOR";

export type BehavioralEvidenceKind =
  | "FLOW"
  | "OPTIONS_SKEW"
  | "FUNDING"
  | "LIQUIDATION"
  | "ORDER_BOOK"
  | "PREDICTION_MARKET"
  | "ETF_FLOW"
  | "ATTENTION"
  | "SEARCH"
  | "TEXT_SENTIMENT";

export interface FrontierSafetyEnvelope {
  shadowOnly: true;
  liveExecution: false;
  canonicalAlphaMutation: false;
  dipControlled: false;
}

export interface ExpectationDistribution {
  metric: string;
  units: string;
  horizon: FrontierHorizon;
  mean: number;
  uncertainty: number;
  generatedAt: string;
  evidenceKinds: BeliefEvidenceKind[];
  evidenceRefs: string[];
}

export interface BehavioralObservation {
  observedAt: string;
  participant: ParticipantClass;
  kind: BehavioralEvidenceKind;
  strength: number;
  evidenceRef: string;
}

export interface CrowdState {
  fear: number;
  uncertainty: number;
  fomo: number;
  capitulation: number;
  euphoria: number;
  crowding: number;
  reflexivity: number;
  forcedFlowPressure: number;
  observations: BehavioralObservation[];
}

export interface EarliestKnowableLedger {
  firstPrimaryWorldAt: string | null;
  firstSourcePublishedAt: string | null;
  firstBrianObservedAt: string | null;
  firstMarketImpliedMoveAt: string | null;
  firstLegalActionableAt: string | null;
}

export interface FrontierOpportunityContext {
  decisionAt: string;
  eventFamily: string;
  entityId: string;
  instrumentId: string;
  brianBeliefP: ExpectationDistribution;
  marketBeliefQ: ExpectationDistribution;
  realizedX?: ExpectationDistribution | null;
  crowd: CrowdState;
  earliestKnowable: EarliestKnowableLedger;
  regimeRef: string | null;
  independentEvidenceGroups: number;
  safety: FrontierSafetyEnvelope;
}

export interface FrontierOpportunityAssessment {
  contractVersion: string;
  pointInTimeClear: boolean;
  marketBeliefAnchored: boolean;
  behavioralStateGrounded: boolean;
  comparableBeliefs: boolean;
  beliefResidual: number | null;
  eligibleForResearchSignal: boolean;
  doNotTradeReasons: string[];
}

const FUTURE_SKEW_MS = 5_000;
const MARKET_ANCHORS = new Set<BeliefEvidenceKind>([
  "OPTIONS",
  "FUTURES",
  "PREDICTION_MARKET",
  "SPOT_CURVE",
  "POSITIONING",
]);
const BEHAVIORAL_MARKET_EVIDENCE = new Set<BehavioralEvidenceKind>([
  "FLOW",
  "OPTIONS_SKEW",
  "FUNDING",
  "LIQUIDATION",
  "ORDER_BOOK",
  "PREDICTION_MARKET",
  "ETF_FLOW",
]);

function timestamp(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function inUnitInterval(value: number): boolean {
  return Number.isFinite(value) && value >= 0 && value <= 1;
}

function validDistribution(value: ExpectationDistribution): boolean {
  return Boolean(
    value.metric &&
      value.units &&
      Number.isFinite(value.mean) &&
      Number.isFinite(value.uncertainty) &&
      value.uncertainty >= 0 &&
      timestamp(value.generatedAt) != null &&
      value.evidenceKinds.length > 0 &&
      value.evidenceRefs.length > 0,
  );
}

function assertSafety(envelope: FrontierSafetyEnvelope): void {
  if (
    envelope.shadowOnly !== true ||
    envelope.liveExecution !== false ||
    envelope.canonicalAlphaMutation !== false ||
    envelope.dipControlled !== false
  ) {
    throw new Error("Frontier safety envelope violated");
  }
}

function timestampsArePointInTime(input: FrontierOpportunityContext): boolean {
  const decisionAt = timestamp(input.decisionAt);
  if (decisionAt == null) return false;
  const candidateTimes: Array<string | null | undefined> = [
    input.brianBeliefP.generatedAt,
    input.marketBeliefQ.generatedAt,
    input.earliestKnowable.firstPrimaryWorldAt,
    input.earliestKnowable.firstSourcePublishedAt,
    input.earliestKnowable.firstBrianObservedAt,
    input.earliestKnowable.firstMarketImpliedMoveAt,
    input.earliestKnowable.firstLegalActionableAt,
    ...input.crowd.observations.map((row) => row.observedAt),
  ];
  if (input.realizedX) candidateTimes.push(input.realizedX.generatedAt);
  return candidateTimes.every((value) => {
    if (value == null) return true;
    const parsed = timestamp(value);
    return parsed != null && parsed <= decisionAt + FUTURE_SKEW_MS;
  });
}

function crowdIsWellFormed(crowd: CrowdState): boolean {
  const dimensions = [
    crowd.fear,
    crowd.uncertainty,
    crowd.fomo,
    crowd.capitulation,
    crowd.euphoria,
    crowd.crowding,
    crowd.reflexivity,
    crowd.forcedFlowPressure,
  ];
  if (!dimensions.every(inUnitInterval)) return false;
  return crowd.observations.every((row) =>
    Boolean(
      row.evidenceRef &&
        timestamp(row.observedAt) != null &&
        inUnitInterval(row.strength),
    )
  );
}

export function assessFrontierOpportunity(input: FrontierOpportunityContext): FrontierOpportunityAssessment {
  assertSafety(input.safety);
  const reasons: string[] = [];

  const pointInTimeClear = timestampsArePointInTime(input);
  if (!pointInTimeClear) reasons.push("POINT_IN_TIME_VIOLATION");

  const pValid = validDistribution(input.brianBeliefP);
  const qValid = validDistribution(input.marketBeliefQ);
  if (!pValid) reasons.push("BRIAN_BELIEF_INVALID");
  if (!qValid) reasons.push("MARKET_BELIEF_INVALID");

  const comparableBeliefs = pValid && qValid &&
    input.brianBeliefP.metric === input.marketBeliefQ.metric &&
    input.brianBeliefP.units === input.marketBeliefQ.units &&
    input.brianBeliefP.horizon === input.marketBeliefQ.horizon;
  if (!comparableBeliefs) reasons.push("P_Q_NOT_COMPARABLE");

  const marketBeliefAnchored = input.marketBeliefQ.evidenceKinds.some((kind) => MARKET_ANCHORS.has(kind));
  if (!marketBeliefAnchored) reasons.push("Q_HAS_NO_MARKET_ANCHOR");

  const crowdWellFormed = crowdIsWellFormed(input.crowd);
  if (!crowdWellFormed) reasons.push("CROWD_STATE_INVALID");
  const behavioralStateGrounded = crowdWellFormed && input.crowd.observations.some((row) => BEHAVIORAL_MARKET_EVIDENCE.has(row.kind));
  if (!behavioralStateGrounded) reasons.push("PSYCHOLOGY_NOT_GROUNDED_IN_MARKET_BEHAVIOR");

  if (input.independentEvidenceGroups < 2) reasons.push("INSUFFICIENT_INDEPENDENT_EVIDENCE");

  const beliefResidual = comparableBeliefs ? input.brianBeliefP.mean - input.marketBeliefQ.mean : null;
  const eligibleForResearchSignal = reasons.length === 0;

  return {
    contractVersion: BRIAN_FRONTIER_CONTRACT_VERSION,
    pointInTimeClear,
    marketBeliefAnchored,
    behavioralStateGrounded,
    comparableBeliefs,
    beliefResidual,
    eligibleForResearchSignal,
    doNotTradeReasons: reasons,
  };
}

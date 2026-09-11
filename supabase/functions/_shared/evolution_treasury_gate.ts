import {
  planTreasuryCycle,
  type TreasuryCyclePlan,
  type TreasuryOpportunity,
  type TreasuryState,
} from "./evolution_treasury.ts";

export const BRIAN_TREASURY_GATE_VERSION = "brian.treasury-promotion-gate.v2";
export const TREASURY_PROMOTION_MAX_AGE_SECONDS = 6 * 60 * 60;
const PROMOTION_FUTURE_SKEW_SECONDS = 5;

export interface PromotionGateState {
  authorized: boolean;
  reason: string;
  evidenceRef: string | null;
  decidedAt?: string | null;
}

export interface PromotionGatedTreasuryPlan extends TreasuryCyclePlan {
  promotionGate: PromotionGateState;
  gateVersion: typeof BRIAN_TREASURY_GATE_VERSION;
}

function time(value: string): number | null {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function normalizePromotionGate(
  gate: PromotionGateState,
  observedAt: string,
): PromotionGateState {
  if (!gate.authorized) return { ...gate };
  const now = time(observedAt);
  const decided = gate.decidedAt ? time(gate.decidedAt) : null;
  if (now == null) {
    return { ...gate, authorized: false, reason: "promotion gate cycle timestamp is invalid" };
  }
  if (decided == null) {
    return { ...gate, authorized: false, reason: "promotion gate has no valid decision timestamp" };
  }
  const ageSeconds = (now - decided) / 1000;
  if (ageSeconds < -PROMOTION_FUTURE_SKEW_SECONDS) {
    return { ...gate, authorized: false, reason: "promotion gate decision timestamp is in the future" };
  }
  if (ageSeconds > TREASURY_PROMOTION_MAX_AGE_SECONDS) {
    return {
      ...gate,
      authorized: false,
      reason: `promotion gate expired after ${Math.round(ageSeconds)}s without fresh EXPECTED_EDGE approval`,
    };
  }
  return { ...gate };
}

function latestMarkByAsset(opportunities: TreasuryOpportunity[], observedAt: string): Map<string, TreasuryOpportunity> {
  const now = time(observedAt);
  const out = new Map<string, TreasuryOpportunity>();
  if (now == null) return out;
  for (const opportunity of opportunities) {
    const at = time(opportunity.observedAt);
    if (at == null || at > now + 5_000 || !Number.isFinite(opportunity.referencePrice) || opportunity.referencePrice <= 0) continue;
    const previous = out.get(opportunity.assetId);
    const previousAt = previous ? time(previous.observedAt) ?? -1 : -1;
    if (!previous || at > previousAt) out.set(opportunity.assetId, opportunity);
  }
  return out;
}

function closedGateInvalidations(
  state: TreasuryState,
  opportunities: TreasuryOpportunity[],
  observedAt: string,
  reason: string,
): TreasuryOpportunity[] {
  const latest = latestMarkByAsset(opportunities, observedAt);
  return state.positions.map((position) => {
    const mark = latest.get(position.assetId);
    return {
      assetId: position.assetId,
      direction: position.direction,
      observedAt,
      referencePrice: mark?.referencePrice ?? position.entryPrice,
      expectedNetEdgeBps: 0,
      roundTripCostBps: position.roundTripCostBps,
      reliabilityConfidence: 0.5,
      matureGroupCount: 0,
      pitClear: false,
      recommendation: "PROMOTION_GATE_CLOSED",
      sourceDecisionId: mark?.sourceDecisionId ?? `promotion-gate:${reason}`,
    };
  });
}

export function planPromotionGatedTreasuryCycle(input: {
  state: TreasuryState;
  opportunities: TreasuryOpportunity[];
  observedAt: string;
  promotionGate: PromotionGateState;
  positionIdFor: (opportunity: TreasuryOpportunity) => string;
}): PromotionGatedTreasuryPlan {
  const promotionGate = normalizePromotionGate(input.promotionGate, input.observedAt);
  const effectiveOpportunities = promotionGate.authorized
    ? input.opportunities
    : closedGateInvalidations(input.state, input.opportunities, input.observedAt, promotionGate.reason);

  const planned = planTreasuryCycle({
    state: input.state,
    opportunities: effectiveOpportunities,
    observedAt: input.observedAt,
    positionIdFor: input.positionIdFor,
  });

  const blockedReasons = [...planned.blockedReasons];
  if (!promotionGate.authorized) blockedReasons.unshift(`layer4 promotion gate closed: ${promotionGate.reason}`);

  return {
    ...planned,
    blockedReasons,
    promotionGate,
    gateVersion: BRIAN_TREASURY_GATE_VERSION,
  };
}

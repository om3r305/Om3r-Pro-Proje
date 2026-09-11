import {
  planTreasuryCycle,
  type TreasuryCyclePlan,
  type TreasuryOpportunity,
  type TreasuryState,
} from "./evolution_treasury.ts";

export const BRIAN_TREASURY_GATE_VERSION = "brian.treasury-promotion-gate.v1";

export interface PromotionGateState {
  authorized: boolean;
  reason: string;
  evidenceRef: string | null;
}

export interface PromotionGatedTreasuryPlan extends TreasuryCyclePlan {
  promotionGate: PromotionGateState;
  gateVersion: typeof BRIAN_TREASURY_GATE_VERSION;
}

function time(value: string): number | null {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
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
  const effectiveOpportunities = input.promotionGate.authorized
    ? input.opportunities
    : closedGateInvalidations(input.state, input.opportunities, input.observedAt, input.promotionGate.reason);

  const planned = planTreasuryCycle({
    state: input.state,
    opportunities: effectiveOpportunities,
    observedAt: input.observedAt,
    positionIdFor: input.positionIdFor,
  });

  const blockedReasons = [...planned.blockedReasons];
  if (!input.promotionGate.authorized) blockedReasons.unshift(`layer4 promotion gate closed: ${input.promotionGate.reason}`);

  return {
    ...planned,
    blockedReasons,
    promotionGate: { ...input.promotionGate },
    gateVersion: BRIAN_TREASURY_GATE_VERSION,
  };
}

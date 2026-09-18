import {
  planTreasuryCycle,
  type TreasuryAction,
  type TreasuryCyclePlan,
  type TreasuryOpportunity,
  type TreasuryPosition,
  type TreasuryState,
} from "./evolution_treasury.ts";
import {
  CANONICAL_ALPHA_SHADOW_MAX_TICKET_USD,
  CANONICAL_ALPHA_SHADOW_POSITION_PREFIX,
  isCanonicalAlphaShadowPosition,
  type CanonicalAlphaShadowOpportunity,
} from "./evolution_treasury_alpha_shadow.ts";

export const BRIAN_TREASURY_GATE_VERSION = "brian.treasury-promotion-gate.v4";
export const TREASURY_PROMOTION_MAX_AGE_SECONDS = 6 * 60 * 60;
const PROMOTION_FUTURE_SKEW_SECONDS = 5;
const MAX_SHADOW_POSITIONS = 8;
// Keep the canonical ALPHA fallback active in SHADOW while BIG_MOVE validation runs.
// This creates only bounded virtual positions; live_execution remains false and the
// EXPECTED_EDGE promotion gate still controls the ordinary Treasury allocator.
const CANONICAL_ALPHA_MICRO_ENTRY_ENABLED = true;

export interface PromotionGateState {
  authorized: boolean;
  reason: string;
  evidenceRef: string | null;
  decidedAt?: string | null;
}

export interface PromotionGatedTreasuryPlan extends TreasuryCyclePlan {
  promotionGate: PromotionGateState;
  gateVersion: typeof BRIAN_TREASURY_GATE_VERSION;
  canonicalAlphaShadow: {
    enabled: boolean;
    candidates: number;
    actions: number;
  };
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
  return state.positions.filter((position) => !isCanonicalAlphaShadowPosition(position)).map((position) => {
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

function canonicalShadowInvalidationsWhenPromoted(
  state: TreasuryState,
  opportunities: TreasuryOpportunity[],
  observedAt: string,
): TreasuryOpportunity[] {
  const latest = latestMarkByAsset(opportunities, observedAt);
  return state.positions.filter(isCanonicalAlphaShadowPosition).map((position) => {
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
      recommendation: "CANONICAL_ALPHA_SHADOW_LANE_CLOSED",
      sourceDecisionId: mark?.sourceDecisionId ?? "promotion-gate-open",
    };
  });
}

function latestCanonicalByAsset(rows: CanonicalAlphaShadowOpportunity[]): Map<string, CanonicalAlphaShadowOpportunity> {
  const out = new Map<string, CanonicalAlphaShadowOpportunity>();
  for (const row of rows) {
    const previous = out.get(row.assetId);
    const at = time(row.observedAt) ?? -1;
    const previousAt = previous ? time(previous.observedAt) ?? -1 : -1;
    if (!previous || at > previousAt) out.set(row.assetId, row);
  }
  return out;
}

/**
 * The ordinary allocator requires positive expected edge. Canonical ALPHA's score is not
 * expected-return bps, so it must never be copied into that field. For an already-open
 * canonical shadow position only, this maintenance row is a private control sentinel that
 * lets the existing risk/mark machinery HOLD a refreshed same-direction OPEN signal. It is
 * never persisted as an expected-edge action and can never create a new position by itself.
 */
function canonicalShadowMaintenance(
  state: TreasuryState,
  fallbacks: CanonicalAlphaShadowOpportunity[],
): TreasuryOpportunity[] {
  const latest = latestCanonicalByAsset(fallbacks);
  const out: TreasuryOpportunity[] = [];
  for (const position of state.positions.filter(isCanonicalAlphaShadowPosition)) {
    const signal = latest.get(position.assetId);
    if (!signal) continue;
    const keep = signal.actionable && signal.pitClear && signal.recommendation === "ALLOW_CANONICAL_ALPHA_SHADOW" && signal.direction === position.direction;
    out.push({
      assetId: position.assetId,
      direction: position.direction,
      observedAt: signal.observedAt,
      referencePrice: signal.referencePrice,
      expectedNetEdgeBps: keep ? 3 : 0,
      roundTripCostBps: position.roundTripCostBps,
      reliabilityConfidence: keep ? 0.65 : 0.5,
      matureGroupCount: keep ? Math.max(2, signal.matureGroupCount) : 0,
      pitClear: keep,
      recommendation: keep ? "ALLOW_EDGE" : "CANONICAL_ALPHA_SHADOW_REVOKED",
      sourceDecisionId: signal.sourceDecisionId,
    });
  }
  return out;
}

function halfEntryCostUsd(capitalUsd: number, roundTripCostBps: number): number {
  return capitalUsd * Math.max(0, roundTripCostBps) / 20_000;
}

function alphaActionForAsset(rows: CanonicalAlphaShadowOpportunity[], assetId: string): CanonicalAlphaShadowOpportunity | null {
  return latestCanonicalByAsset(rows).get(assetId) ?? null;
}

function rewriteCanonicalShadowExitReasons(
  actions: TreasuryAction[],
  fallbacks: CanonicalAlphaShadowOpportunity[],
  promoted: boolean,
) {
  for (const action of actions) {
    if (action.kind !== "EXIT" || !String(action.positionId || "").startsWith(CANONICAL_ALPHA_SHADOW_POSITION_PREFIX)) continue;
    if (promoted && action.reason === "EDGE_INVALIDATED") {
      action.reason = "ALPHA_SHADOW_LANE_CLOSED_BY_PROMOTION";
      continue;
    }
    const signal = alphaActionForAsset(fallbacks, action.assetId);
    if (signal && action.reason === "EDGE_INVALIDATED") action.reason = "ALPHA_SIGNAL_REVOKED";
  }
}

function applyCanonicalAlphaShadowOpens(
  plan: TreasuryCyclePlan,
  fallbacks: CanonicalAlphaShadowOpportunity[],
  observedAt: string,
): number {
  if (!CANONICAL_ALPHA_MICRO_ENTRY_ENABLED) return 0;
  const candidates = [...latestCanonicalByAsset(fallbacks).values()]
    .filter((row) => row.actionable && row.pitClear && row.recommendation === "ALLOW_CANONICAL_ALPHA_SHADOW")
    .sort((a, b) => (b.signalScore - a.signalScore) || ((time(b.observedAt) ?? 0) - (time(a.observedAt) ?? 0)));
  let opens = 0;
  for (const row of candidates) {
    if (plan.state.positions.length >= MAX_SHADOW_POSITIONS) break;
    if (plan.state.positions.some((position) => position.assetId === row.assetId)) continue;
    const requested = row.requestedCapitalUsd;
    if (!Number.isFinite(requested) || requested <= 0 || requested > CANONICAL_ALPHA_SHADOW_MAX_TICKET_USD) continue;
    if (!Number.isFinite(row.roundTripCostBps) || row.roundTripCostBps < 0) continue;

    const entryCostRate = Math.max(0, row.roundTripCostBps) / 20_000;
    const maxByCash = plan.state.cashUsd / (1 + entryCostRate);
    const maxByDeployment = Math.max(0, plan.afterEquityUsd - plan.deploymentUsd);
    const capitalUsd = Math.min(requested, maxByCash, maxByDeployment);
    // Fail closed instead of silently changing ALPHA's preregistered virtual ticket.
    if (capitalUsd + 1e-9 < requested) continue;
    const entryCost = halfEntryCostUsd(capitalUsd, row.roundTripCostBps);
    if (plan.state.cashUsd + 1e-9 < capitalUsd + entryCost) continue;

    const positionId = `${CANONICAL_ALPHA_SHADOW_POSITION_PREFIX}${row.sourceDecisionId}`;
    plan.state.cashUsd -= capitalUsd + entryCost;
    plan.state.realizedPnlUsd -= entryCost;
    plan.state.cumulativeCostsUsd += entryCost;
    plan.state.positions.push({
      positionId,
      assetId: row.assetId,
      direction: row.direction,
      openedAt: observedAt,
      entryPrice: row.referencePrice,
      capitalUsd,
      // Unknown by design: canonical evidence_score is not expected-return bps.
      entryExpectedNetEdgeBps: 0,
      latestExpectedNetEdgeBps: 0,
      roundTripCostBps: row.roundTripCostBps,
      sourceDecisionId: row.sourceDecisionId,
      highWaterPnlBps: 0,
    });
    plan.actions.push({
      kind: "OPEN",
      assetId: row.assetId,
      direction: row.direction,
      capitalUsd,
      referencePrice: row.referencePrice,
      costUsd: entryCost,
      expectedNetEdgeBps: 0,
      sourceDecisionId: row.sourceDecisionId,
      reason: "CANONICAL_ALPHA_SHADOW_SIGNAL",
      positionId,
    });
    plan.deploymentUsd += capitalUsd;
    plan.afterEquityUsd = Math.max(1e-9, plan.afterEquityUsd - entryCost);
    const denominator = Math.max(plan.afterEquityUsd, plan.deploymentUsd);
    plan.deploymentPct = denominator > 0 ? plan.deploymentUsd / denominator : 0;
    plan.cashReservePct = plan.afterEquityUsd > 0 ? plan.state.cashUsd / plan.afterEquityUsd : 0;
    opens++;
  }
  return opens;
}

export function planPromotionGatedTreasuryCycle(input: {
  state: TreasuryState;
  opportunities: TreasuryOpportunity[];
  shadowFallbackOpportunities?: CanonicalAlphaShadowOpportunity[];
  observedAt: string;
  promotionGate: PromotionGateState;
  positionIdFor: (opportunity: TreasuryOpportunity) => string;
}): PromotionGatedTreasuryPlan {
  const promotionGate = normalizePromotionGate(input.promotionGate, input.observedAt);
  const fallbackOpportunities = input.shadowFallbackOpportunities ?? [];
  const effectiveOpportunities = promotionGate.authorized
    ? [...input.opportunities, ...canonicalShadowInvalidationsWhenPromoted(input.state, input.opportunities, input.observedAt)]
    : [
      ...closedGateInvalidations(input.state, input.opportunities, input.observedAt, promotionGate.reason),
      ...canonicalShadowMaintenance(input.state, fallbackOpportunities),
    ];

  const planned = planTreasuryCycle({
    state: input.state,
    opportunities: effectiveOpportunities,
    observedAt: input.observedAt,
    positionIdFor: input.positionIdFor,
  });

  rewriteCanonicalShadowExitReasons(planned.actions, fallbackOpportunities, promotionGate.authorized);
  const fallbackOpens = promotionGate.authorized ? 0 : applyCanonicalAlphaShadowOpens(planned, fallbackOpportunities, input.observedAt);
  const fallbackActions = planned.actions.filter((action) =>
    String(action.positionId || "").startsWith(CANONICAL_ALPHA_SHADOW_POSITION_PREFIX) ||
    action.reason === "CANONICAL_ALPHA_SHADOW_SIGNAL" || action.reason === "ALPHA_SIGNAL_REVOKED" ||
    action.reason === "ALPHA_SHADOW_LANE_CLOSED_BY_PROMOTION"
  ).length;

  const blockedReasons = [...planned.blockedReasons];
  if (!promotionGate.authorized) blockedReasons.unshift(`layer4 promotion gate closed: ${promotionGate.reason}`);
  if (!promotionGate.authorized && !CANONICAL_ALPHA_MICRO_ENTRY_ENABLED) {
    blockedReasons.unshift("canonical ALPHA micro-entry paused: BIG_MOVE validation mode; evidence/outcome learning remains active");
  }

  return {
    ...planned,
    blockedReasons,
    promotionGate,
    gateVersion: BRIAN_TREASURY_GATE_VERSION,
    canonicalAlphaShadow: {
      enabled: !promotionGate.authorized && CANONICAL_ALPHA_MICRO_ENTRY_ENABLED,
      candidates: fallbackOpportunities.filter((row) => row.actionable).length,
      actions: Math.max(fallbackActions, fallbackOpens),
    },
  };
}

import { planPromotionGatedTreasuryCycle } from "./evolution_treasury_gate.ts";
import {
  CANONICAL_ALPHA_SHADOW_POSITION_PREFIX,
  buildCanonicalAlphaShadowOpportunities,
  type CanonicalAlphaDecisionRow,
} from "./evolution_treasury_alpha_shadow.ts";
import { initialTreasuryState, type TreasuryPosition } from "./evolution_treasury.ts";

function alpha(overrides: Partial<CanonicalAlphaDecisionRow> = {}): CanonicalAlphaDecisionRow {
  return {
    decisionId: "alpha-short-1",
    observedAt: "2026-09-15T20:11:17Z",
    assetId: "crypto:CRCLBUSDT",
    referencePrice: 85.595,
    action: "OPEN_SHORT",
    direction: -1,
    evidenceScore: 0.204,
    independentGroupCount: 2,
    requestedVirtualNotionalUsd: 3,
    estimatedRoundTripCostBps: 21.168,
    vetoReason: null,
    evidenceClass: "PROSPECTIVE_DEVELOPMENT_SHADOW",
    shadowOnly: true,
    liveExecution: false,
    costQuality: "L2_OBSERVED",
    l2RuntimeStatus: "OBSERVED",
    ...overrides,
  };
}

const closedGate = {
  authorized: false,
  reason: "no active EXPECTED_EDGE promotion; newest verdict is KEEP_EXPERIMENTAL",
  evidenceRef: "gate-keep-experimental",
  decidedAt: "2026-09-15T20:00:00Z",
};

Deno.test("canonical ALPHA remains a research candidate but cannot charge main SHADOW treasury", () => {
  const state = initialTreasuryState("2026-09-15T20:11:00Z", 5000);
  const fallbacks = buildCanonicalAlphaShadowOpportunities([alpha()], state.positions, "2026-09-15T20:12:00Z");
  if (fallbacks.length !== 1 || !fallbacks[0].actionable || fallbacks[0].requestedCapitalUsd !== 3) throw new Error(JSON.stringify(fallbacks));

  const plan = planPromotionGatedTreasuryCycle({
    state,
    opportunities: [],
    shadowFallbackOpportunities: fallbacks,
    observedAt: "2026-09-15T20:12:00Z",
    promotionGate: closedGate,
    positionIdFor: () => "normal-unused",
  });

  if (plan.actions.length !== 0 || plan.state.positions.length !== 0) throw new Error(JSON.stringify(plan));
  if (plan.state.cashUsd !== 5000 || plan.state.realizedPnlUsd !== 0 || plan.state.cumulativeCostsUsd !== 0) throw new Error(JSON.stringify(plan.state));
  if (plan.canonicalAlphaShadow.enabled) throw new Error(JSON.stringify(plan.canonicalAlphaShadow));
  if (!plan.blockedReasons.some((row) => row.includes("canonical ALPHA micro-entry paused"))) throw new Error(JSON.stringify(plan.blockedReasons));
});

Deno.test("an already-open canonical ALPHA probe is flattened once without allowing a reopen", () => {
  const state = initialTreasuryState("2026-09-15T20:11:00Z", 5000);
  const position: TreasuryPosition = {
    positionId: `${CANONICAL_ALPHA_SHADOW_POSITION_PREFIX}alpha-short-1`,
    assetId: "crypto:CRCLBUSDT",
    direction: -1,
    openedAt: "2026-09-15T20:10:00Z",
    entryPrice: 85.5,
    capitalUsd: 3,
    entryExpectedNetEdgeBps: 0,
    latestExpectedNetEdgeBps: 0,
    roundTripCostBps: 21.168,
    sourceDecisionId: "alpha-short-1",
    highWaterPnlBps: 0,
  };
  state.positions=[position];
  state.cashUsd=4996.9968248;

  const fallbacks = buildCanonicalAlphaShadowOpportunities([
    alpha({decisionId:"alpha-short-2",observedAt:"2026-09-15T20:11:50Z",referencePrice:85.55})
  ], state.positions, "2026-09-15T20:12:00Z");

  const plan = planPromotionGatedTreasuryCycle({
    state,
    opportunities: [],
    shadowFallbackOpportunities: fallbacks,
    observedAt: "2026-09-15T20:12:00Z",
    promotionGate: closedGate,
    positionIdFor: () => "normal-unused",
  });

  if (plan.state.positions.length !== 0) throw new Error(JSON.stringify(plan.state.positions));
  if (plan.actions.length !== 1 || plan.actions[0].kind !== "EXIT" || plan.actions[0].reason !== "ALPHA_SIGNAL_REVOKED") throw new Error(JSON.stringify(plan.actions));
});

Deno.test("open EXPECTED_EDGE promotion never allows canonical research fallback to add a position", () => {
  const state = initialTreasuryState("2026-09-15T20:11:00Z", 5000);
  const fallbacks = buildCanonicalAlphaShadowOpportunities([alpha()], state.positions, "2026-09-15T20:12:00Z");
  const plan = planPromotionGatedTreasuryCycle({
    state,
    opportunities: [],
    shadowFallbackOpportunities: fallbacks,
    observedAt: "2026-09-15T20:12:00Z",
    promotionGate: { authorized: true, reason: "EXPECTED_EDGE promoted", evidenceRef: "promotion-1", decidedAt: "2026-09-15T20:10:00Z" },
    positionIdFor: () => "normal-unused",
  });
  if (plan.actions.length || plan.state.positions.length || plan.canonicalAlphaShadow.enabled) throw new Error(JSON.stringify(plan));
});

import { planPromotionGatedTreasuryCycle } from "./evolution_treasury_gate.ts";
import {
  CANONICAL_ALPHA_SHADOW_POSITION_PREFIX,
  buildCanonicalAlphaShadowOpportunities,
  type CanonicalAlphaDecisionRow,
} from "./evolution_treasury_alpha_shadow.ts";
import { initialTreasuryState } from "./evolution_treasury.ts";

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

Deno.test("closed EXPECTED_EDGE gate may open only ALPHA's tiny canonical SHADOW ticket", () => {
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
  const open = plan.actions.find((action) => action.kind === "OPEN");
  if (!open || open.reason !== "CANONICAL_ALPHA_SHADOW_SIGNAL" || open.capitalUsd !== 3) throw new Error(JSON.stringify(plan.actions));
  if (!String(open.positionId).startsWith(CANONICAL_ALPHA_SHADOW_POSITION_PREFIX)) throw new Error(String(open.positionId));
  if (plan.promotionGate.authorized || !plan.blockedReasons.some((row) => row.includes("promotion gate closed"))) throw new Error(JSON.stringify(plan));
  if (plan.state.positions.length !== 1 || plan.state.positions[0].capitalUsd !== 3) throw new Error(JSON.stringify(plan.state.positions));
  if (plan.state.cashUsd >= 4997 || plan.state.cashUsd < 4996.9) throw new Error(`unexpected cash ${plan.state.cashUsd}`);
  if (!plan.canonicalAlphaShadow.enabled || plan.canonicalAlphaShadow.actions < 1) throw new Error(JSON.stringify(plan.canonicalAlphaShadow));
});

Deno.test("fresh same-direction canonical ALPHA refresh holds the tiny shadow position without duplicate OPEN", () => {
  const initial = initialTreasuryState("2026-09-15T20:11:00Z", 5000);
  const firstFallback = buildCanonicalAlphaShadowOpportunities([alpha()], initial.positions, "2026-09-15T20:12:00Z");
  const opened = planPromotionGatedTreasuryCycle({
    state: initial,
    opportunities: [],
    shadowFallbackOpportunities: firstFallback,
    observedAt: "2026-09-15T20:12:00Z",
    promotionGate: closedGate,
    positionIdFor: () => "normal-unused",
  });
  const refresh = alpha({ decisionId: "alpha-short-2", observedAt: "2026-09-15T20:14:41Z", referencePrice: 85.635, independentGroupCount: 3 });
  const refreshedFallback = buildCanonicalAlphaShadowOpportunities([refresh], opened.state.positions, "2026-09-15T20:15:00Z");
  const held = planPromotionGatedTreasuryCycle({
    state: opened.state,
    opportunities: [],
    shadowFallbackOpportunities: refreshedFallback,
    observedAt: "2026-09-15T20:15:00Z",
    promotionGate: closedGate,
    positionIdFor: () => "normal-unused",
  });
  if (held.actions.length !== 0 || held.state.positions.length !== 1 || held.state.positions[0].direction !== -1) throw new Error(JSON.stringify(held));
});

Deno.test("canonical ALPHA WAIT revokes and closes its fallback SHADOW position", () => {
  const initial = initialTreasuryState("2026-09-15T20:11:00Z", 5000);
  const firstFallback = buildCanonicalAlphaShadowOpportunities([alpha()], initial.positions, "2026-09-15T20:12:00Z");
  const opened = planPromotionGatedTreasuryCycle({
    state: initial,
    opportunities: [],
    shadowFallbackOpportunities: firstFallback,
    observedAt: "2026-09-15T20:12:00Z",
    promotionGate: closedGate,
    positionIdFor: () => "normal-unused",
  });
  const wait = alpha({
    decisionId: "alpha-wait-1",
    observedAt: "2026-09-15T20:17:59Z",
    referencePrice: 85.77,
    action: "WAIT",
    direction: -1,
    requestedVirtualNotionalUsd: null,
    estimatedRoundTripCostBps: null,
    costQuality: null,
    l2RuntimeStatus: null,
  });
  const waitFallback = buildCanonicalAlphaShadowOpportunities([wait], opened.state.positions, "2026-09-15T20:18:05Z");
  if (waitFallback.length !== 1 || waitFallback[0].actionable) throw new Error(JSON.stringify(waitFallback));
  const closed = planPromotionGatedTreasuryCycle({
    state: opened.state,
    opportunities: [],
    shadowFallbackOpportunities: waitFallback,
    observedAt: "2026-09-15T20:18:05Z",
    promotionGate: closedGate,
    positionIdFor: () => "normal-unused",
  });
  if (closed.state.positions.length !== 0 || closed.actions[0]?.kind !== "EXIT" || closed.actions[0]?.reason !== "ALPHA_SIGNAL_REVOKED") throw new Error(JSON.stringify(closed.actions));
});

Deno.test("open EXPECTED_EDGE promotion never allows canonical fallback lane to add a position", () => {
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

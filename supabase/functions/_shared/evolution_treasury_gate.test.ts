import { planPromotionGatedTreasuryCycle } from "./evolution_treasury_gate.ts";
import { initialTreasuryState, type TreasuryOpportunity, type TreasuryPosition } from "./evolution_treasury.ts";

function opportunity(overrides: Partial<TreasuryOpportunity> = {}): TreasuryOpportunity {
  return {
    assetId: "crypto:BTCUSDT",
    direction: 1,
    observedAt: "2026-09-11T13:00:00Z",
    referencePrice: 100,
    expectedNetEdgeBps: 24,
    roundTripCostBps: 20,
    reliabilityConfidence: 0.6,
    matureGroupCount: 3,
    pitClear: true,
    recommendation: "ALLOW_EDGE",
    sourceDecisionId: "d-1",
    ...overrides,
  };
}

function position(overrides: Partial<TreasuryPosition> = {}): TreasuryPosition {
  return {
    positionId: "p-1",
    assetId: "crypto:BTCUSDT",
    direction: 1,
    openedAt: "2026-09-11T12:50:00Z",
    entryPrice: 100,
    capitalUsd: 1000,
    entryExpectedNetEdgeBps: 20,
    latestExpectedNetEdgeBps: 20,
    roundTripCostBps: 20,
    sourceDecisionId: "d-old",
    highWaterPnlBps: 0,
    ...overrides,
  };
}

Deno.test("closed Layer-4 promotion gate keeps a fresh treasury entirely in cash", () => {
  const state = initialTreasuryState("2026-09-11T12:59:00Z");
  const plan = planPromotionGatedTreasuryCycle({
    state,
    opportunities: [opportunity()],
    observedAt: "2026-09-11T13:00:00Z",
    promotionGate: { authorized: false, reason: "no promoted EXPECTED_EDGE experiment", evidenceRef: null },
    positionIdFor: (row) => `p-${row.sourceDecisionId}`,
  });
  if (plan.actions.length !== 0 || plan.state.positions.length !== 0 || plan.deploymentUsd !== 0) throw new Error(JSON.stringify(plan));
  if (plan.state.cashUsd !== 10_000 || plan.promotionGate.authorized) throw new Error(JSON.stringify(plan));
  if (!plan.blockedReasons.some((row) => row.includes("promotion gate closed"))) throw new Error(JSON.stringify(plan.blockedReasons));
});

Deno.test("closing Layer-4 promotion gate immediately flattens any existing shadow allocation", () => {
  const state = initialTreasuryState("2026-09-11T12:50:00Z");
  state.cashUsd = 9000;
  state.positions = [position()];
  const plan = planPromotionGatedTreasuryCycle({
    state,
    opportunities: [opportunity({ observedAt: "2026-09-11T13:00:00Z", referencePrice: 101 })],
    observedAt: "2026-09-11T13:00:00Z",
    promotionGate: { authorized: false, reason: "promotion revoked", evidenceRef: "promotion:old" },
    positionIdFor: () => "unused",
  });
  if (plan.state.positions.length !== 0 || plan.deploymentUsd !== 0) throw new Error(JSON.stringify(plan));
  if (plan.actions.length !== 1 || plan.actions[0].kind !== "EXIT" || plan.actions[0].reason !== "EDGE_INVALIDATED") throw new Error(JSON.stringify(plan.actions));
});

Deno.test("open promotion gate delegates to the normal opportunity allocator", () => {
  const state = initialTreasuryState("2026-09-11T12:59:00Z");
  const plan = planPromotionGatedTreasuryCycle({
    state,
    opportunities: [opportunity()],
    observedAt: "2026-09-11T13:00:00Z",
    promotionGate: { authorized: true, reason: "EXPECTED_EDGE promoted in prospective shadow", evidenceRef: "promotion:123" },
    positionIdFor: (row) => `p-${row.sourceDecisionId}`,
  });
  if (plan.actions[0]?.kind !== "OPEN" || plan.state.positions.length !== 1) throw new Error(JSON.stringify(plan));
  if (!plan.promotionGate.authorized || plan.promotionGate.evidenceRef !== "promotion:123") throw new Error(JSON.stringify(plan.promotionGate));
});

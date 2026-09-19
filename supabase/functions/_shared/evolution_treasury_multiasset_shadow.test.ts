import { planPromotionGatedTreasuryCycle } from "./evolution_treasury_gate.ts";
import {
  MULTIASSET_SHADOW_POSITION_PREFIX,
  buildMultiassetShadowOpportunities,
  type MultiassetAlphaDecisionRow,
} from "./evolution_treasury_multiasset_shadow.ts";
import { initialTreasuryState } from "./evolution_treasury.ts";

function decision(overrides: Partial<MultiassetAlphaDecisionRow> = {}): MultiassetAlphaDecisionRow {
  return {
    decisionId: "multi-spx-1",
    observedAt: "2026-09-21T14:10:00Z",
    assetId: "index:SP500",
    assetClass: "index",
    providerTime: "2026-09-21T14:09:55Z",
    referencePrice: 7650,
    action: "OPEN_LONG",
    direction: 1,
    evidenceScore: 0.74,
    independentGroupCount: 2,
    supportGroups: ["event_link", "market_reaction"],
    linkedEventIds: ["event-fed-1"],
    requestedVirtualNotionalUsd: 10,
    estimatedRoundTripCostBps: 10,
    vetoReason: null,
    sessionState: "REGULAR",
    dataLatencySeconds: 5,
    shadowOnly: true,
    liveExecution: false,
    metadata: {
      provider_quality: "PUBLIC_UNOFFICIAL_SHADOW_ONLY",
      direction_source: "OBSERVED_MARKET_REACTION_NOT_HEADLINE_GUESS",
      execution_grade: false,
      shadow_lane: "MULTIASSET_EVENT_REACTION",
    },
    ...overrides,
  };
}

const closedGate = {
  authorized: false,
  reason: "EXPECTED_EDGE not promoted",
  evidenceRef: null,
  decidedAt: null,
};

Deno.test("fresh event + observed market reaction can open only a bounded multiasset SHADOW ticket", () => {
  const state = initialTreasuryState("2026-09-21T14:09:00Z", 5000);
  const multi = buildMultiassetShadowOpportunities([decision()], state.positions, "2026-09-21T14:10:30Z");
  if (multi.length !== 1 || !multi[0].actionable || multi[0].requestedCapitalUsd !== 10) throw new Error(JSON.stringify(multi));

  const plan = planPromotionGatedTreasuryCycle({
    state,
    opportunities: [],
    multiassetShadowOpportunities: multi,
    observedAt: "2026-09-21T14:10:30Z",
    promotionGate: closedGate,
    positionIdFor: () => "ordinary-unused",
  });
  const open = plan.actions.find((row) => row.kind === "OPEN");
  if (!open || open.reason !== "MULTIASSET_SHADOW_SIGNAL" || open.capitalUsd !== 10) throw new Error(JSON.stringify(plan.actions));
  if (!String(open.positionId).startsWith(MULTIASSET_SHADOW_POSITION_PREFIX)) throw new Error(String(open.positionId));
  if (plan.state.positions.length !== 1 || plan.multiassetShadow.candidates !== 1 || plan.multiassetShadow.actions < 1) throw new Error(JSON.stringify(plan));
});

Deno.test("headline alone cannot open a multiasset position without observed market reaction", () => {
  const state = initialTreasuryState("2026-09-21T14:09:00Z", 5000);
  const rows = buildMultiassetShadowOpportunities([
    decision({ independentGroupCount: 1, supportGroups: ["event_link"], evidenceScore: 0.9 }),
  ], state.positions, "2026-09-21T14:10:30Z");
  if (rows.length !== 1 || rows[0].actionable) throw new Error(JSON.stringify(rows));
  const plan = planPromotionGatedTreasuryCycle({
    state, opportunities: [], multiassetShadowOpportunities: rows,
    observedAt: "2026-09-21T14:10:30Z", promotionGate: closedGate,
    positionIdFor: () => "ordinary-unused",
  });
  if (plan.actions.length || plan.state.positions.length) throw new Error(JSON.stringify(plan));
});

Deno.test("execution-grade or live-execution flags never enter the unofficial multiasset SHADOW lane", () => {
  const state = initialTreasuryState("2026-09-21T14:09:00Z", 5000);
  const rows = buildMultiassetShadowOpportunities([
    decision({ metadata: {
      provider_quality: "PUBLIC_UNOFFICIAL_SHADOW_ONLY",
      direction_source: "OBSERVED_MARKET_REACTION_NOT_HEADLINE_GUESS",
      execution_grade: true,
      shadow_lane: "MULTIASSET_EVENT_REACTION",
    }}),
    decision({ decisionId: "multi-spx-live", liveExecution: true }),
  ], state.positions, "2026-09-21T14:10:30Z");
  if (rows.some((row) => row.actionable)) throw new Error(JSON.stringify(rows));
});

Deno.test("closed or stale market revokes an existing multiasset SHADOW position", () => {
  const initial = initialTreasuryState("2026-09-21T14:09:00Z", 5000);
  const first = buildMultiassetShadowOpportunities([decision()], initial.positions, "2026-09-21T14:10:30Z");
  const opened = planPromotionGatedTreasuryCycle({
    state: initial, opportunities: [], multiassetShadowOpportunities: first,
    observedAt: "2026-09-21T14:10:30Z", promotionGate: closedGate,
    positionIdFor: () => "ordinary-unused",
  });

  const wait = decision({
    decisionId: "multi-spx-closed",
    observedAt: "2026-09-21T14:20:00Z",
    providerTime: "2026-09-21T14:05:00Z",
    referencePrice: 7658,
    action: "WAIT",
    direction: 0,
    evidenceScore: 0.4,
    independentGroupCount: 1,
    supportGroups: ["event_link"],
    requestedVirtualNotionalUsd: 0,
    vetoReason: "MARKET_CLOSED_OR_STALE",
    sessionState: "CLOSED",
    dataLatencySeconds: 900,
  });
  const maintenance = buildMultiassetShadowOpportunities([wait], opened.state.positions, "2026-09-21T14:20:10Z");
  if (maintenance.length !== 1 || maintenance[0].actionable) throw new Error(JSON.stringify(maintenance));
  const closed = planPromotionGatedTreasuryCycle({
    state: opened.state, opportunities: [], multiassetShadowOpportunities: maintenance,
    observedAt: "2026-09-21T14:20:10Z", promotionGate: closedGate,
    positionIdFor: () => "ordinary-unused",
  });
  if (closed.state.positions.length !== 0) throw new Error(JSON.stringify(closed.state.positions));
  if (closed.actions[0]?.kind !== "EXIT" || closed.actions[0]?.reason !== "MULTIASSET_SESSION_CLOSED_OR_STALE") throw new Error(JSON.stringify(closed.actions));
});

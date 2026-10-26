import { assert, assertEquals } from "jsr:@std/assert@1";
import { initialTreasuryState, type TreasuryOpportunity, type TreasuryState } from "./evolution_treasury.ts";
import { planPromotionGatedTreasuryCycle, riskGuardState, TREASURY_RISK_GUARD } from "./evolution_treasury_gate.ts";
import { UNIFIED_POSITION_PREFIX } from "./evolution_treasury_unified.ts";

const NOW = "2026-10-26T15:00:00.000Z";
const at = (m: number) => new Date(Date.parse(NOW) - m * 60_000).toISOString();
const gate = { authorized: false, reason: "no EXPECTED_EDGE promotion", evidenceRef: null } as never;
const strong = (assetId: string, id: string): TreasuryOpportunity => ({
  assetId, direction: 1, observedAt: at(1), referencePrice: 100, expectedNetEdgeBps: 10, roundTripCostBps: 10, // moderate: each takes ~22 % of equity
  reliabilityConfidence: 0.65, matureGroupCount: 6, pitClear: true, recommendation: "ALLOW_EDGE", sourceDecisionId: id,
});
const plan = (state: TreasuryState, unified: TreasuryOpportunity[], dailyRisk?: Parameters<typeof riskGuardState>[0]) =>
  planPromotionGatedTreasuryCycle({ state, opportunities: [], multiassetShadowOpportunities: [], unifiedOpportunities: unified,
    unifiedEnabled: true, dailyRisk, observedAt: NOW, promotionGate: gate, positionIdFor: (o) => `treasury:${o.sourceDecisionId}` });

Deno.test("risk guard state: daily loss limit and daily trade limit", () => {
  assertEquals(riskGuardState(undefined), null);
  const loss = riskGuardState({ opensToday: 0, dayStartEquityUsd: 10_000, currentEquityUsd: 9_790 })!;
  assertEquals(loss.openBudget, 0);
  assert(loss.blocked!.startsWith("daily loss limit"));
  const fine = riskGuardState({ opensToday: 5, dayStartEquityUsd: 10_000, currentEquityUsd: 9_850 })!;
  assertEquals([fine.openBudget, fine.blocked], [TREASURY_RISK_GUARD.maxOpensPerDay - 5, null]);
  const busy = riskGuardState({ opensToday: TREASURY_RISK_GUARD.maxOpensPerDay, dayStartEquityUsd: null, currentEquityUsd: null })!;
  assertEquals(busy.openBudget, 0);
  assert(busy.blocked!.startsWith("daily trade limit"));
});

Deno.test("no guard input keeps the previous behaviour; a budget caps the number of new positions", () => {
  const state = initialTreasuryState(at(60), 10_000);
  const opps = [strong("equity:AAA", "a"), strong("equity:BBB", "b"), strong("equity:CCC", "c")];
  const free = plan(state, opps);
  assertEquals(free.actions.filter((a) => a.kind === "OPEN").length, 3);
  assertEquals(free.riskGuard, null);
  const capped = plan(state, opps, { opensToday: TREASURY_RISK_GUARD.maxOpensPerDay - 1, dayStartEquityUsd: 10_000, currentEquityUsd: 10_000 });
  assertEquals(capped.actions.filter((a) => a.kind === "OPEN").length, 1);
});

Deno.test("a blocked day opens nothing but still exits: the risk stop fires", () => {
  const state: TreasuryState = { ...initialTreasuryState(at(120), 10_000), cashUsd: 9_000,
    positions: [{ positionId: `${UNIFIED_POSITION_PREFIX}d0`, assetId: "equity:OLD", direction: 1, openedAt: at(10), entryPrice: 100,
      capitalUsd: 1_000, entryExpectedNetEdgeBps: 20, latestExpectedNetEdgeBps: 20, roundTripCostBps: 10, sourceDecisionId: "d0", highWaterPnlBps: 0 }] };
  const crash = { ...strong("equity:OLD", "m"), referencePrice: 97, pitClear: false, recommendation: "UNIFIED_WAIT_EDGE_BELOW_COST", expectedNetEdgeBps: 0 };
  const p = plan(state, [crash, strong("equity:NEW", "n")], { opensToday: 0, dayStartEquityUsd: 10_000, currentEquityUsd: 9_700 });
  assertEquals(p.actions.filter((a) => a.kind === "OPEN").length, 0);
  assertEquals(p.actions.find((a) => a.kind === "EXIT")?.reason, "RISK_STOP");
  assert(p.blockedReasons[0].startsWith("daily loss limit"));
});

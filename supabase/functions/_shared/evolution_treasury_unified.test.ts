import { assert, assertAlmostEquals, assertEquals } from "jsr:@std/assert@1";
import { initialTreasuryState, planTreasuryCycle, type TreasuryState } from "./evolution_treasury.ts";
import { planPromotionGatedTreasuryCycle } from "./evolution_treasury_gate.ts";
import type { MultiassetAlphaDecisionRow } from "./evolution_treasury_multiasset_shadow.ts";
import type { CanonicalAlphaDecisionRow } from "./evolution_treasury_alpha_shadow.ts";
import {
  buildUnifiedOpportunities,
  newsClassOf,
  parseSegmentEdges,
  segmentVerdict,
  signalFromCanonical,
  signalFromMultiasset,
  UNIFIED_LANES,
  UNIFIED_POSITION_PREFIX,
  UNIFIED_RULES,
  type SegmentEdge,
} from "./evolution_treasury_unified.ts";

const NOW = "2026-10-22T14:00:00.000Z";
const at = (minutesAgo: number) => new Date(Date.parse(NOW) - minutesAgo * 60_000).toISOString();
const lane = UNIFIED_RULES.lane;

function maRow(over: Partial<MultiassetAlphaDecisionRow> = {}): MultiassetAlphaDecisionRow {
  return {
    decisionId: "d1", observedAt: at(2), assetId: "equity:AAA", assetClass: "equity", providerTime: at(4),
    referencePrice: 100, action: "OPEN_LONG", direction: 1, evidenceScore: 0.8, independentGroupCount: 4,
    supportGroups: ["event_link", "market_reaction"], linkedEventIds: [], requestedVirtualNotionalUsd: 20,
    estimatedRoundTripCostBps: 10, vetoReason: null, sessionState: "REGULAR", dataLatencySeconds: 100,
    shadowOnly: true, liveExecution: false, metadata: {}, ...over,
  };
}
const row = (over: Partial<MultiassetAlphaDecisionRow> = {}, news = "NONE") => signalFromMultiasset(maRow(over), news);
const seg = (over: Partial<SegmentEdge> = {}): SegmentEdge =>
  ({ lane, assetClass: "equity", direction: 1, newsClass: "NONE", n: 400, grossBps: 40, seBps: 3, hitRate: 0.62, ...over });
const gate = { authorized: false, reason: "no EXPECTED_EDGE promotion", evidenceRef: null } as never;

Deno.test("segment verdict: shrinks gross toward zero, subtracts one SE and the cost; refuses thin evidence", () => {
  const v = segmentVerdict(seg(), 10);
  assertEquals(v.reason, "OK");
  assertAlmostEquals(v.shrunkGrossBps!, 40 * 400 / 500, 1e-9);
  assertAlmostEquals(v.netEdgeBps!, 32 - 3 - 10, 1e-9);
  assert(v.reliability > 0.5 && v.reliability < 0.62);
  assertEquals(segmentVerdict(seg({ n: 99 }), 10).reason, "TOO_FEW_SAMPLES");
  assertEquals(segmentVerdict(undefined, 10).reason, "NO_SEGMENT");
  assert(segmentVerdict(seg({ grossBps: 7.3 }), 14).netEdgeBps! < 0, "today's best real segment does not clear cost");
});

Deno.test("cost gate: only an entry-grade signal whose segment clears cost + margin is allowed", () => {
  const ok = buildUnifiedOpportunities([row()], [seg()], [], NOW);
  assertEquals(ok.opportunities.length, 1);
  assertEquals(ok.opportunities[0].recommendation, "ALLOW_EDGE");
  assertEquals(ok.opportunities[0].pitClear, true);
  const weak = buildUnifiedOpportunities([row()], [seg({ grossBps: 10 })], [], NOW);
  assertEquals(weak.opportunities.length, 0, "below cost and no open position: nothing to trade or mark");
  assertEquals(weak.verdicts[0].reason, "OK");
  const stale = buildUnifiedOpportunities([row({ observedAt: at(20) })], [seg()], [], NOW);
  assertEquals(stale.opportunities.length, 0, "a 20-min-old signal is not entry grade");
  const vetoed = buildUnifiedOpportunities([row({ vetoReason: "SPREAD" })], [seg()], [], NOW);
  assertEquals(vetoed.opportunities.length, 0);
});

Deno.test("open unified positions are always marked at the newest price, even on a WAIT decision", () => {
  const position = {
    positionId: `${UNIFIED_POSITION_PREFIX}d0`, assetId: "equity:AAA", direction: 1 as const, openedAt: at(30), entryPrice: 100,
    capitalUsd: 1000, entryExpectedNetEdgeBps: 19, latestExpectedNetEdgeBps: 19, roundTripCostBps: 10, sourceDecisionId: "d0", highWaterPnlBps: 0,
  };
  const b = buildUnifiedOpportunities([row({ decisionId: "d2", action: "WAIT", direction: 0, referencePrice: 101 })], [seg()], [position], NOW);
  assertEquals(b.opportunities.length, 1);
  assertEquals(b.opportunities[0].referencePrice, 101);
  assertEquals(b.opportunities[0].direction, 1);
  assertEquals(b.opportunities[0].pitClear, false);
});

Deno.test("anti-churn: a unified position survives a revoked signal until its minimum hold, the risk stop still fires", () => {
  const mk = (openedMinutesAgo: number, entry = 100): TreasuryState => ({
    ...initialTreasuryState(at(120), 10_000), cashUsd: 9_000,
    positions: [{ positionId: `${UNIFIED_POSITION_PREFIX}d0`, assetId: "equity:AAA", direction: 1, openedAt: at(openedMinutesAgo), entryPrice: entry,
      capitalUsd: 1000, entryExpectedNetEdgeBps: 19, latestExpectedNetEdgeBps: 19, roundTripCostBps: 10, sourceDecisionId: "d0", highWaterPnlBps: 0 }],
  });
  const revoked = buildUnifiedOpportunities([row({ decisionId: "d2", action: "WAIT", direction: 0, referencePrice: 100.1 })], [seg()], mk(20).positions, NOW).opportunities;
  const hold = (s: TreasuryState) => planTreasuryCycle({ state: s, opportunities: revoked, observedAt: NOW, positionIdFor: (o) => o.sourceDecisionId,
    minHoldSecondsFor: () => UNIFIED_RULES.minHoldSeconds });
  assertEquals(hold(mk(20)).actions.length, 0, "20 min old: kept");
  assertEquals(hold(mk(61)).actions[0]?.kind, "EXIT", "61 min old: closed at the current price");
  assertEquals(hold(mk(61)).actions[0]?.referencePrice, 100.1);
  const crash = buildUnifiedOpportunities([row({ decisionId: "d3", action: "WAIT", direction: 0, referencePrice: 98 })], [seg()], mk(5).positions, NOW).opportunities;
  const stopped = planTreasuryCycle({ state: mk(5), opportunities: crash, observedAt: NOW, positionIdFor: (o) => o.sourceDecisionId,
    minHoldSecondsFor: () => UNIFIED_RULES.minHoldSeconds });
  assertEquals(stopped.actions[0]?.reason, "RISK_STOP");
});

Deno.test("evidence-based sizing through the gate: strong evidence earns a large share, weak gets less, none gets nothing; no micro tickets", () => {
  const state = initialTreasuryState(at(60), 10_000);
  const size = (s: SegmentEdge) => {
    const u = buildUnifiedOpportunities([row()], [s], [], NOW);
    const plan = planPromotionGatedTreasuryCycle({ state, opportunities: [], multiassetShadowOpportunities: [], unifiedOpportunities: u.opportunities,
      unifiedEnabled: true, observedAt: NOW, promotionGate: gate, positionIdFor: (o) => `treasury:${o.sourceDecisionId}` });
    return plan;
  };
  const strong = size(seg({ grossBps: 120, seBps: 2, hitRate: 0.7, n: 2000 }));
  const weak = size(seg({ grossBps: 25, seBps: 2, hitRate: 0.55, n: 150 }));
  const none = size(seg({ grossBps: 5 }));
  const cap = (p: typeof strong) => p.actions.filter((a) => a.kind === "OPEN").reduce((s, a) => s + a.capitalUsd, 0);
  assert(cap(strong) > 2_000, `strong evidence deploys a large share (got ${cap(strong)})`);
  assert(cap(weak) > 0 && cap(weak) < cap(strong), `weak evidence deploys less (got ${cap(weak)})`);
  assertEquals(cap(none), 0);
  assert(strong.actions.every((a) => String(a.positionId).startsWith(UNIFIED_POSITION_PREFIX)));
  assert(none.blockedReasons[0].includes("holding cash"));
  assertEquals(strong.unified?.enabled, true);
});

Deno.test("unified OFF keeps the previous behaviour (promotion gate wording, micro lane enabled)", () => {
  const plan = planPromotionGatedTreasuryCycle({ state: initialTreasuryState(at(60), 10_000), opportunities: [], multiassetShadowOpportunities: [],
    observedAt: NOW, promotionGate: gate, positionIdFor: (o) => `treasury:${o.sourceDecisionId}` });
  assert(plan.blockedReasons.some((r) => r.startsWith("layer4 promotion gate closed")));
  assertEquals(plan.multiassetShadow.enabled, true);
  assertEquals(plan.unified?.enabled, false);
});

Deno.test("crypto lane: canonical alpha decisions use their own CANONICAL_ALPHA evidence", () => {
  const c = signalFromCanonical({
    decisionId: "c1", observedAt: at(1), assetId: "crypto:BTCUSDT", referencePrice: 60_000, action: "OPEN_SHORT", direction: -1,
    estimatedRoundTripCostBps: 8, independentGroupCount: 3, vetoReason: null, shadowOnly: true, liveExecution: false,
  } as unknown as CanonicalAlphaDecisionRow);
  assertEquals([c.lane, c.assetClass, c.newsClass, c.sessionState], [UNIFIED_LANES.crypto, "crypto", "NONE", "REGULAR"]);
  const cryptoSeg = seg({ lane: UNIFIED_LANES.crypto, assetClass: "crypto", direction: -1 });
  assertEquals(buildUnifiedOpportunities([c], [cryptoSeg], [], NOW).opportunities[0]?.recommendation, "ALLOW_EDGE");
  const wrongLane = buildUnifiedOpportunities([c], [seg({ assetClass: "crypto", direction: -1 })], [], NOW);
  assertEquals(wrongLane.opportunities.length, 0, "multiasset evidence never licenses a crypto trade");
  assertEquals(wrongLane.verdicts[0].reason, "NO_SEGMENT");
});

Deno.test("news reliability: evidence is split by the best trust class of the linked news", () => {
  const trust = new Map([["e1", "OFFICIAL_PRIMARY"], ["e2", "INDEPENDENT_PROFESSIONAL"], ["e3", "SOCIAL"]]);
  assertEquals(newsClassOf([], trust), "NONE");
  assertEquals(newsClassOf(["e3", "e1"], trust), "OFFICIAL");
  assertEquals(newsClassOf(["e2", "e3"], trust), "PROFESSIONAL");
  assertEquals(newsClassOf(["e3"], trust), "UNVERIFIED");
  assertEquals(newsClassOf(["missing"], trust), "UNKNOWN");
  const segs = [seg({ newsClass: "OFFICIAL" }), seg({ newsClass: "UNVERIFIED", grossBps: 5 })];
  assertEquals(buildUnifiedOpportunities([row({}, "OFFICIAL")], segs, [], NOW).opportunities.length, 1);
  assertEquals(buildUnifiedOpportunities([row({}, "UNVERIFIED")], segs, [], NOW).opportunities.length, 0);
  assertEquals(parseSegmentEdges([{ lane, asset_class: "equity", direction: 1, n: 1, gross_bps: 1, se_bps: 1, hit_rate: 0.5 }])[0].newsClass, "NONE");
});

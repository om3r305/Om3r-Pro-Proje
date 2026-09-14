import { compileCostControlAlphaCandidate } from "./cost-control-alpha-expected-edge.ts";

const decisionAt = "2026-09-13T13:00:00Z";
const opportunity = (opportunityId: string, grossEdgeBps = 100) => ({
  opportunityId,
  grossEdgeBps,
  observedAt: "2026-09-13T12:55:00Z",
});
const reliability = (opportunityId: string, groupId: string, value = 0.8) => ({
  opportunityId,
  groupId,
  reliability: value,
  snapshotAt: "2026-09-13T12:58:00Z",
  mature: true,
});
const cost = (extra: Record<string, unknown> = {}) => ({
  asOf: "2026-09-13T12:59:00Z",
  sourceId: "book-a",
  cadenceSeconds: 300,
  spreadBps: 10,
  feeBps: 5,
  depthCostBps: 5,
  fillability: 1,
  ...extra,
});
const input = (extra: Record<string, unknown> = {}) => ({
  opportunities: [opportunity("o1"), opportunity("o2", 80)],
  reliabilitySnapshots: [reliability("o1", "g1"), reliability("o2", "g2")],
  cost: cost(),
  ...extra,
});

Deno.test("ranks net edge after dynamic round-trip cost", () => {
  const result = compileCostControlAlphaCandidate(input(), { decisionAt });
  if (
    result.recommendation !== "ALLOW_EDGE" ||
    result.selectedOpportunityId !== "o1" ||
    result.roundTripCostBps !== 40 ||
    result.rankedOpportunities[0].netEdgeBps !== 40
  ) {
    throw new Error(JSON.stringify(result));
  }
});

Deno.test("rejects a cost-dominated opportunity and enforces strict margin", () => {
  const result = compileCostControlAlphaCandidate(
    input({ opportunities: [opportunity("o1", 50), opportunity("o2", 50)] }),
    { decisionAt, minimumEdgeToCostMargin: 0.25 },
  );
  if (
    result.recommendation !== "DOWNGRADE_TO_WAIT" || result.eligible ||
    !result.reasons.includes("edge-to-cost margin is insufficient")
  ) {
    throw new Error(JSON.stringify(result));
  }
});

Deno.test("fails closed for stale, future, missing, zero, and unsafe cost", () => {
  for (
    const value of [
      null,
      cost({ spreadBps: 0, feeBps: 0, depthCostBps: 0 }),
      cost({ fillability: Number.MIN_VALUE }),
      cost({ asOf: "2026-09-14T00:00:00Z" }),
      cost({ depthCostBps: Number.NaN }),
    ]
  ) {
    const result = compileCostControlAlphaCandidate(input({ cost: value }), {
      decisionAt,
    });
    if (
      result.recommendation !== "COST_UNAVAILABLE" || result.eligible ||
      result.roundTripCostBps !== null
    ) throw new Error(JSON.stringify(result));
  }
});

Deno.test("is deterministic under timestamp normalization and input permutation", () => {
  const baseline = compileCostControlAlphaCandidate(input(), { decisionAt });
  const permuted = compileCostControlAlphaCandidate({
    ...input(),
    opportunities: [opportunity("o2", 80), opportunity("o1")],
    reliabilitySnapshots: [reliability("o2", "g2"), reliability("o1", "g1")],
  }, { decisionAt: Date.parse(decisionAt) });
  if (JSON.stringify(baseline) !== JSON.stringify(permuted)) {
    throw new Error("decision changed under equivalent evidence");
  }
});

Deno.test("future and contradictory reliability remain telemetry or contamination", () => {
  const result = compileCostControlAlphaCandidate(
    input({
      opportunities: [...input().opportunities, {
        ...opportunity("future", 999),
        observedAt: "2026-09-14T00:00:00Z",
      }],
      reliabilitySnapshots: [
        reliability("o1", "g1"),
        reliability("o1", "g1", 0.7),
        reliability("o2", "g2"),
      ],
    }),
    { decisionAt },
  );
  if (
    result.recommendation !== "CONTAMINATED_EVIDENCE" ||
    result.futureTelemetry.futureEvidenceCount !== 1 ||
    result.shadow_only !== true || result.live_execution !== false ||
    result.canonical_mutation !== false || result.promotionReady !== false
  ) {
    throw new Error(JSON.stringify(result));
  }
});

import { compileExpectedEdgeAlphaCandidate } from "./expected-edge-alpha-expected-edge.ts";

const now = "2026-09-13T13:00:00Z";
const observation = (id: string, extra: Record<string, unknown> = {}) => ({
  observationId: id,
  providerId: "provider-a",
  sensorFamily: "momentum",
  horizon: "300s",
  direction: "up",
  observedAt: "2026-09-13T12:55:00Z",
  cadenceSeconds: 300,
  evaluationStartAt: "2026-09-13T12:55:00Z",
  evaluationEndAt: "2026-09-13T13:00:00Z",
  ...extra,
});
const reliability = (
  id: string,
  groupId: string,
  extra: Record<string, unknown> = {},
) => ({
  observationId: id,
  groupId,
  snapshotAt: "2026-09-13T12:58:00Z",
  horizon: "300s",
  cadenceSeconds: 300,
  provenance: "reliability-a",
  expectedMoveBps: 100,
  reliability: 0.8,
  uncertaintyBps: 5,
  mature: true,
  ...extra,
});
const cost = (extra: Record<string, unknown> = {}) => ({
  asOf: "2026-09-13T12:59:00Z",
  spreadBps: 10,
  feeBps: 5,
  slippageBps: 5,
  fillability: 1,
  sourceId: "book-a",
  cadenceSeconds: 300,
  ...extra,
});
const input = (extra: Record<string, unknown> = {}) => ({
  sourceObservations: [observation("o1"), observation("o2")],
  reliabilitySnapshots: [reliability("o1", "g1"), reliability("o2", "g2")],
  cost: cost(),
  ...extra,
});

Deno.test("reconciles gross minus dynamic cost, uncertainty, and decay", () => {
  const result = compileExpectedEdgeAlphaCandidate(input(), {
    decisionAt: now,
    minimumNetMarginBps: 0,
  });
  if (
    result.recommendation !== "ALLOW_EDGE" ||
    result.expectedNetEdgeBps !==
      result.expectedGrossMoveBps! - result.estimatedRoundTripCostBps! -
        result.uncertaintyPenaltyBps - result.eventDecayPenaltyBps -
        0
  ) throw new Error(JSON.stringify(result));
});

Deno.test("fails closed for missing, zero, negative, and non-finite costs", () => {
  for (
    const value of [
      null,
      { ...cost(), spreadBps: 0, feeBps: 0, slippageBps: 0 },
      { ...cost(), feeBps: -1 },
      { ...cost(), feeBps: Infinity },
    ]
  ) {
    const result = compileExpectedEdgeAlphaCandidate(input({ cost: value }), {
      decisionAt: now,
    });
    if (
      result.recommendation !== "COST_UNAVAILABLE" ||
      result.eligible
    ) throw new Error(JSON.stringify(result));
  }
});

Deno.test("requires independent mature groups and exact observation binding", () => {
  const result = compileExpectedEdgeAlphaCandidate(
    input({
      reliabilitySnapshots: [
        reliability("unknown", "g1"),
        reliability("o1", "g1"),
      ],
    }),
    { decisionAt: now },
  );
  if (
    result.recommendation !== "INSUFFICIENT_LAGGED_EVIDENCE" ||
    result.matureIndependentGroupCount !== 1
  ) throw new Error(JSON.stringify(result));
});

Deno.test("future and conflicting evidence cannot become decision features", () => {
  const result = compileExpectedEdgeAlphaCandidate(
    input({
      sourceObservations: [
        observation("o1"),
        observation("future", {
          observedAt: "2026-09-14T00:00:00Z",
        }),
      ],
      reliabilitySnapshots: [
        reliability("o1", "g1"),
        reliability("o1", "g1", { expectedMoveBps: 101 }),
      ],
    }),
    { decisionAt: now },
  );
  if (
    result.futureTelemetry.futureEvidenceCount !== 1 ||
    result.recommendation !== "CONTAMINATED_EVIDENCE" ||
    result.shadow_only !== true || result.live_execution !== false ||
    result.canonical_mutation !== false || result.promotionReady !== false
  ) {
    throw new Error(JSON.stringify(result));
  }
});

Deno.test("input order and stale evidence remain bounded and deterministic", () => {
  const a = compileExpectedEdgeAlphaCandidate(
    input({ eventAt: "2026-09-13T11:00:00Z", eventCadenceSeconds: 300 }),
    {
      decisionAt: now,
      minimumNetMarginBps: 10,
    },
  );
  const b = compileExpectedEdgeAlphaCandidate({
    ...input({ eventAt: "2026-09-13T11:00:00Z", eventCadenceSeconds: 300 }),
    sourceObservations: [observation("o2"), observation("o1")],
    reliabilitySnapshots: [reliability("o2", "g2"), reliability("o1", "g1")],
  }, { decisionAt: now, minimumNetMarginBps: 10 });
  if (
    JSON.stringify(a) !== JSON.stringify(b) ||
    Math.abs(a.expectedNetEdgeBps ?? 0) > 1_000_000
  ) throw new Error("unstable result");
});

Deno.test("preserves a horizon-aligned 24-hour evaluation window", () => {
  const result = compileExpectedEdgeAlphaCandidate(
    input({
      sourceObservations: [observation("long", {
        horizon: "24h",
        observedAt: "2026-09-12T13:00:00Z",
        cadenceSeconds: 3600,
        evaluationStartAt: "2026-09-12T13:00:00Z",
        evaluationEndAt: "2026-09-13T12:00:00Z",
      })],
      reliabilitySnapshots: [
        reliability("long", "g1", {
          horizon: "24h",
          cadenceSeconds: 3600,
          snapshotAt: "2026-09-13T12:30:00Z",
        }),
        reliability("o2", "g2", {
          horizon: "24h",
          cadenceSeconds: 3600,
          snapshotAt: "2026-09-13T12:30:00Z",
        }),
      ],
    }),
    { decisionAt: now },
  );
  if (result.invalidEvidenceCount === 0 || result.eligible) {
    throw new Error("mismatched source horizon was not rejected");
  }
});

Deno.test("rejects missing provenance and stale cadence contracts", () => {
  const result = compileExpectedEdgeAlphaCandidate(
    input({
      reliabilitySnapshots: [
        reliability("o1", "g1", { provenance: undefined }),
      ],
      cost: cost({ asOf: "2026-09-13T10:00:00Z" }),
    }),
    { decisionAt: now },
  );
  if (result.eligible || result.recommendation !== "COST_UNAVAILABLE") {
    throw new Error(JSON.stringify(result));
  }
});

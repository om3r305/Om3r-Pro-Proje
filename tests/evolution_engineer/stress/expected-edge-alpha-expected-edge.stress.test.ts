import { compileExpectedEdgeAlphaCandidate } from "../../../supabase/functions/_shared/evolution_candidates/expected-edge-alpha-expected-edge.ts";

Deno.test("stress input is bounded, deterministic, and shadow-only", () => {
  const sourceObservations = Array.from({ length: 3_000 }, (_, index) => ({
    observationId: `o${index}`,
    providerId: `p${index}`,
    sensorFamily: "stress",
    horizon: "300s",
    direction: "up",
    observedAt: "2026-09-13T12:00:00Z",
  }));
  const result = compileExpectedEdgeAlphaCandidate({
    sourceObservations,
    reliabilitySnapshots: [{
      observationId: "o0",
      groupId: "g1",
      snapshotAt: "2026-09-13T12:30:00Z",
      expectedMoveBps: 1e12,
      reliability: 1,
      uncertaintyBps: 0,
      mature: true,
    }],
    cost: {
      asOf: "2026-09-13T12:45:00Z",
      spreadBps: Number.NaN,
      feeBps: 1,
      slippageBps: 1,
      fillability: 1,
    },
  }, {
    decisionAt: "2026-09-13T13:00:00Z",
    maxInputRows: 100,
    maxContributions: 10,
  });
  if (
    !result.truncated || result.eligible ||
    result.recommendation !== "COST_UNAVAILABLE" ||
    result.contributions.length > 10 || result.shadow_only !== true ||
    result.live_execution !== false || result.canonical_mutation !== false ||
    result.promotionReady !== false
  ) throw new Error(JSON.stringify(result));
});

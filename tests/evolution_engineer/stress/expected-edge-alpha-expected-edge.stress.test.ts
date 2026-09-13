import { compileExpectedEdgeAlphaCandidate } from "../../../supabase/functions/_shared/evolution_candidates/expected-edge-alpha-expected-edge.ts";

Deno.test("stress input is bounded, deterministic, and shadow-only", () => {
  const sourceObservations = Array.from({ length: 3_000 }, (_, index) => ({
    observationId: `o${index}`,
    providerId: `p${index}`,
    sensorFamily: "stress",
    horizon: "300s",
    direction: "up",
    observedAt: "2026-09-13T12:00:00Z",
    cadenceSeconds: 300,
    evaluationStartAt: "2026-09-13T12:00:00Z",
    evaluationEndAt: "2026-09-13T12:05:00Z",
  }));
  const result = compileExpectedEdgeAlphaCandidate({
    sourceObservations,
    reliabilitySnapshots: [{
      observationId: "o0",
      groupId: "g1",
      horizon: "300s",
      cadenceSeconds: 300,
      provenance: "stress-reliability",
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
      sourceId: "stress-book",
      cadenceSeconds: 300,
    },
  }, {
    decisionAt: "2026-09-13T13:00:00Z",
    maxInputRows: 100,
    maxContributions: 10,
  });

  Deno.test("fails closed across malformed, extreme, duplicate, and future evidence", () => {
    const base = {
      sourceObservations: [{
        observationId: "o1",
        providerId: "p1",
        sensorFamily: "stress",
        horizon: "300s",
        direction: "up",
        observedAt: "2026-09-13T12:00:00Z",
        cadenceSeconds: 300,
        evaluationStartAt: "2026-09-13T12:00:00Z",
        evaluationEndAt: "2026-09-13T12:05:00Z",
      }],
      reliabilitySnapshots: [{
        observationId: "o1",
        groupId: "g1",
        horizon: "300s",
        cadenceSeconds: 300,
        provenance: "p",
        snapshotAt: "2026-09-13T12:30:00Z",
        expectedMoveBps: 100,
        reliability: 1,
        uncertaintyBps: 0,
        mature: true,
      }],
      cost: {
        asOf: "2026-09-13T12:45:00Z",
        sourceId: "book",
        cadenceSeconds: 300,
        spreadBps: 1,
        feeBps: 1,
        slippageBps: 1,
        fillability: 1,
      },
    };
    const cases = [
      { ...base, sourceObservations: [null] },
      {
        ...base,
        reliabilitySnapshots: [{
          ...base.reliabilitySnapshots[0],
          uncertaintyBps: Number.MAX_VALUE,
        }],
      },
      { ...base, cost: { ...base.cost, fillability: 0 } },
      {
        ...base,
        sourceObservations: [{
          ...base.sourceObservations[0],
          observedAt: "2026-09-14T00:00:00Z",
        }],
      },
      { ...base, reliabilitySnapshots: [] },
      {
        ...base,
        sourceObservations: [{
          ...base.sourceObservations[0],
          horizon: "24h",
          cadenceSeconds: 3600,
          evaluationEndAt: "2026-09-14T12:00:00Z",
        }],
      },
      { ...base, eventAt: "2026-09-14T00:00:00Z", eventCadenceSeconds: 300 },
      { ...base, cost: { ...base.cost, asOf: "2026-09-14T00:00:00Z" } },
      {
        ...base,
        sourceObservations: [base.sourceObservations[0], {
          ...base.sourceObservations[0],
          observationId: "o2",
        }],
        reliabilitySnapshots: [base.reliabilitySnapshots[0], {
          ...base.reliabilitySnapshots[0],
          observationId: "o2",
          groupId: "g2",
        }],
      },
    ];
    for (const value of cases) {
      const result = compileExpectedEdgeAlphaCandidate(value, {
        decisionAt: "2026-09-13T13:00:00Z",
      });
      if (
        result.eligible || result.shadow_only !== true ||
        result.live_execution !== false ||
        result.canonical_mutation !== false || result.promotionReady !== false
      ) {
        throw new Error(JSON.stringify(result));
      }
    }
  });
  if (
    !result.truncated || result.eligible ||
    result.recommendation !== "COST_UNAVAILABLE" ||
    result.contributions.length > 10 || result.shadow_only !== true ||
    result.live_execution !== false || result.canonical_mutation !== false ||
    result.promotionReady !== false
  ) throw new Error(JSON.stringify(result));
});

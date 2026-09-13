import { compileExpectedEdgeAlphaCandidate } from "../../../supabase/functions/_shared/evolution_candidates/expected-edge-alpha-expected-edge.ts";

const envelope = {
  sourceObservations: [
    {
      observationId: "o1",
      providerId: "p1",
      sensorFamily: "momentum",
      horizon: "300s",
      direction: "up",
      observedAt: "2026-09-13T12:00:00Z",
    },
    {
      observationId: "o2",
      providerId: "p2",
      sensorFamily: "breadth",
      horizon: "300s",
      direction: "up",
      observedAt: "2026-09-13T12:01:00Z",
    },
  ],
  reliabilitySnapshots: [
    {
      observationId: "o1",
      groupId: "g1",
      snapshotAt: "2026-09-13T12:30:00Z",
      expectedMoveBps: 100,
      reliability: .8,
      uncertaintyBps: 5,
      mature: true,
    },
    {
      observationId: "o2",
      groupId: "g2",
      snapshotAt: "2026-09-13T12:30:00Z",
      expectedMoveBps: 100,
      reliability: .8,
      uncertaintyBps: 5,
      mature: true,
    },
  ],
  cost: {
    asOf: "2026-09-13T12:45:00Z",
    spreadBps: 10,
    feeBps: 5,
    slippageBps: 5,
    fillability: 1,
  },
};

Deno.test("replay projection is unchanged by future evidence permutations", () => {
  const options = { decisionAt: "2026-09-13T13:00:00Z" };
  const baseline = compileExpectedEdgeAlphaCandidate(envelope, options);
  const futures = [
    {
      observationId: "future",
      providerId: "p9",
      sensorFamily: "x",
      horizon: "300s",
      direction: "up",
      observedAt: "2026-09-14T00:00:00Z",
    },
    {
      observationId: "future-2",
      providerId: "p9",
      sensorFamily: "x",
      horizon: "300s",
      direction: "up",
      observedAt: "2026-09-15T00:00:00Z",
    },
  ];
  for (const sourceObservations of [futures, [...futures].reverse()]) {
    const result = compileExpectedEdgeAlphaCandidate({
      ...envelope,
      sourceObservations: [
        ...envelope.sourceObservations,
        ...sourceObservations,
      ],
    }, options);
    const project = (
      value: ReturnType<typeof compileExpectedEdgeAlphaCandidate>,
    ) => JSON.stringify({ ...value, futureTelemetry: undefined });
    if (
      project(result) !== project(baseline) ||
      result.futureTelemetry.futureEvidenceCount !== 2
    ) {
      throw new Error("future evidence contaminated replay");
    }
  }
});

import { compileCostControlAlphaCandidate } from "../../../supabase/functions/_shared/evolution_candidates/cost-control-alpha-expected-edge.ts";

const envelope = {
  opportunities: [
    {
      opportunityId: "o1",
      grossEdgeBps: 100,
      observedAt: "2026-09-13T12:55:00Z",
    },
    {
      opportunityId: "o2",
      grossEdgeBps: 80,
      observedAt: "2026-09-13T12:55:00Z",
    },
  ],
  reliabilitySnapshots: [
    {
      opportunityId: "o1",
      groupId: "g1",
      provenance: {
        sourceObservationId: "observation-g1",
        rawIndependentGroup: "g1",
        sensorFamily: "family-g1",
        sensorHorizon: "FAST_5_30M",
        direction: 1,
        snapshotWindowEnd: "2026-09-13T12:58:00Z",
        snapshotGeneratedAt: "2026-09-13T12:58:00Z",
        sourceId: "source-g1",
        lineageId: "lineage-g1",
        independent: true,
      },
      reliability: .8,
      snapshotAt: "2026-09-13T12:58:00Z",
      mature: true,
    },
    {
      opportunityId: "o2",
      groupId: "g2",
      provenance: {
        sourceObservationId: "observation-g2",
        rawIndependentGroup: "g2",
        sensorFamily: "family-g2",
        sensorHorizon: "FAST_5_30M",
        direction: 1,
        snapshotWindowEnd: "2026-09-13T12:58:00Z",
        snapshotGeneratedAt: "2026-09-13T12:58:00Z",
        sourceId: "source-g2",
        lineageId: "lineage-g2",
        independent: true,
      },
      reliability: .8,
      snapshotAt: "2026-09-13T12:58:00Z",
      mature: true,
    },
  ],
  sourceObservations: [
    {
      observationId: "observation-g1",
      opportunityId: "o1",
      providerId: "provider-g1",
      sourceId: "source-g1",
      lineageId: "lineage-g1",
      independentGroup: "g1",
      sensorFamily: "family-g1",
      sensorHorizon: "FAST_5_30M",
      direction: 1,
      observedAt: "2026-09-13T12:55:00Z",
    },
    {
      observationId: "observation-g2",
      opportunityId: "o2",
      providerId: "provider-g2",
      sourceId: "source-g2",
      lineageId: "lineage-g2",
      independentGroup: "g2",
      sensorFamily: "family-g2",
      sensorHorizon: "FAST_5_30M",
      direction: 1,
      observedAt: "2026-09-13T12:55:00Z",
    },
  ],
  cost: {
    asOf: "2026-09-13T12:59:00Z",
    costConvention: "ONE_WAY_COMPONENTS_BPS",
    sourceId: "book-a",
    cadenceSeconds: 300,
    spreadBps: 10,
    feeBps: 5,
    depthCostBps: 5,
    fillability: 1,
  },
};

// Hand-authored replay projection; future outcomes are not compiler inputs.
Deno.test("immutable point-in-time replay is invariant to future telemetry", () => {
  const options = { decisionAt: "2026-09-13T13:00:00Z" };
  const baseline = compileCostControlAlphaCandidate(envelope, options);
  const replay = compileCostControlAlphaCandidate({
    ...envelope,
    opportunities: [...envelope.opportunities].reverse().concat({
      opportunityId: "future",
      grossEdgeBps: 999,
      observedAt: "2026-09-14T00:00:00Z",
    }),
    reliabilitySnapshots: [...envelope.reliabilitySnapshots].reverse().concat({
      opportunityId: "o1",
      groupId: "g1",
      provenance: {
        sourceObservationId: "observation-g1",
        sourceId: "future-source",
        lineageId: "future-lineage",
        rawIndependentGroup: "g1",
        sensorFamily: "family-g1",
        sensorHorizon: "FAST_5_30M",
        direction: 1,
        snapshotWindowEnd: "2026-09-14T00:00:00Z",
        snapshotGeneratedAt: "2026-09-14T00:00:00Z",
        independent: true,
      },
      reliability: 1,
      snapshotAt: "2026-09-14T00:00:00Z",
      mature: true,
    }),
  }, options);
  const project = (
    value: ReturnType<typeof compileCostControlAlphaCandidate>,
  ) =>
    JSON.stringify({
      recommendation: value.recommendation,
      eligible: value.eligible,
      selectedOpportunityId: value.selectedOpportunityId,
      rankedOpportunities: value.rankedOpportunities,
      roundTripCostBps: value.roundTripCostBps,
      costComponentsBps: value.costComponentsBps,
      provenance: value.provenance,
    });
  if (
    project(baseline) !== project(replay) ||
    replay.futureTelemetry.futureEvidenceCount !== 2 ||
    baseline.recommendation !== "ALLOW_EDGE"
  ) {
    throw new Error("replay projection changed");
  }
});

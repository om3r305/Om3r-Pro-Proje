import { compileCostControlAlphaCandidate } from "../../../supabase/functions/_shared/evolution_candidates/cost-control-alpha-expected-edge.ts";

const valid = {
  opportunities: [
    {
      opportunityId: "o0",
      grossEdgeBps: 100,
      observedAt: "2026-09-13T12:55:00Z",
    },
    {
      opportunityId: "o1",
      grossEdgeBps: 90,
      observedAt: "2026-09-13T12:55:00Z",
    },
  ],
  reliabilitySnapshots: [
    {
      opportunityId: "o0",
      groupId: "g0",
      provenance: {
        sourceObservationId: "observation-g0",
        rawIndependentGroup: "g0",
        sensorFamily: "family-g0",
        sensorHorizon: "FAST_5_30M",
        direction: 1,
        snapshotWindowEnd: "2026-09-13T12:58:00Z",
        snapshotGeneratedAt: "2026-09-13T12:58:00Z",
        sourceId: "source-g0",
        lineageId: "lineage-g0",
        independent: true,
      },
      reliability: 1,
      snapshotAt: "2026-09-13T12:58:00Z",
      mature: true,
    },
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
      reliability: 1,
      snapshotAt: "2026-09-13T12:58:00Z",
      mature: true,
    },
  ],
  sourceObservations: [
    {
      observationId: "observation-g0",
      opportunityId: "o0",
      providerId: "provider-g0",
      sourceId: "source-g0",
      lineageId: "lineage-g0",
      independentGroup: "g0",
      sensorFamily: "family-g0",
      sensorHorizon: "FAST_5_30M",
      direction: 1,
      observedAt: "2026-09-13T12:55:00Z",
    },
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
  ],
  cost: {
    asOf: "2026-09-13T12:59:00Z",
    costConvention: "ONE_WAY_COMPONENTS_BPS",
    sourceId: "book",
    cadenceSeconds: 300,
    spreadBps: 1,
    feeBps: 1,
    depthCostBps: 1,
    fillability: 1,
  },
};

Deno.test("bounded stress inputs never emit non-finite edge or unsafe execution flags", () => {
  const oversized = {
    ...valid,
    opportunities: Array.from({ length: 3_000 }, (_, i) => ({
      opportunityId: `o${i}`,
      grossEdgeBps: 100,
      observedAt: "2026-09-13T12:55:00Z",
    })),
  };
  const result = compileCostControlAlphaCandidate(oversized, {
    decisionAt: "2026-09-13T13:00:00Z",
    maxInputRows: 100,
  });

  if (
    !result.truncated || result.eligible ||
    (result.roundTripCostBps !== null &&
      !Number.isFinite(result.roundTripCostBps)) ||
    result.shadow_only !== true || result.live_execution !== false ||
    result.canonical_mutation !== false || result.promotionReady !== false
  ) {
    throw new Error(JSON.stringify(result));
  }
  for (
    const mutation of [
      { ...valid, opportunities: [null] },
      {
        ...valid,
        reliabilitySnapshots: [{
          ...valid.reliabilitySnapshots[0],
          reliability: Number.MAX_VALUE,
        }],
      },
      { ...valid, cost: { ...valid.cost, fillability: 0 } },
      { ...valid, cost: { ...valid.cost, spreadBps: Infinity } },
      {
        ...valid,
        opportunities: [...valid.opportunities, {
          opportunityId: "future",
          grossEdgeBps: 1,
          observedAt: "2026-09-14T00:00:00Z",
        }],
      },
    ]
  ) {
    const value = compileCostControlAlphaCandidate(mutation, {
      decisionAt: "2026-09-13T13:00:00Z",
    });

    if (
      value.roundTripCostBps !== null &&
        !Number.isFinite(value.roundTripCostBps) ||
      value.shadow_only !== true || value.live_execution !== false ||
      value.canonical_mutation !== false || value.promotionReady !== false
    ) {
      throw new Error(JSON.stringify(value));
    }
  }
});

Deno.test("adversarial future evidence cannot change the bounded projection", () => {
  const options = { decisionAt: "2026-09-13T13:00:00Z", maxInputRows: 2 };
  const baseline = compileCostControlAlphaCandidate(valid, options);
  const future = compileCostControlAlphaCandidate({
    ...valid,
    opportunities: [
      ...valid.opportunities,
      {
        opportunityId: "future-high-edge",
        grossEdgeBps: 1_000_000,
        observedAt: "2026-09-14T00:00:00Z",
      },
    ],
    reliabilitySnapshots: [
      ...valid.reliabilitySnapshots,
      {
        opportunityId: "future-high-edge",
        groupId: "future-group",
        provenance: {
          sourceId: "future-source",
          lineageId: "future-lineage",
          independent: true,
        },
        reliability: 1,
        snapshotAt: "2026-09-14T00:00:00Z",
        mature: true,
      },
    ],
    sourceObservations: valid.sourceObservations,
  }, options);
  const project = (
    value: ReturnType<typeof compileCostControlAlphaCandidate>,
  ) =>
    JSON.stringify({
      recommendation: value.recommendation,
      eligible: value.eligible,
      selectedOpportunityId: value.selectedOpportunityId,
      rankedOpportunities: value.rankedOpportunities,
    });
  if (
    project(baseline) !== project(future) ||
    future.futureTelemetry.futureEvidenceCount !== 2 ||
    future.selectedOpportunityId !== "o0"
  ) throw new Error(JSON.stringify(future));
});

Deno.test("large provenance conflicts fail closed without duplicate public rows", () => {
  const conflicting = {
    ...valid,
    reliabilitySnapshots: Array.from({ length: 500 }, (_, index) => ({
      ...valid.reliabilitySnapshots[0],
      reliability: index % 2 ? 0.2 : 1,
      provenance: {
        ...valid.reliabilitySnapshots[0].provenance,
        lineageId: `lineage-conflict-${index}`,
      },
    })).concat(valid.reliabilitySnapshots[1]),
  };
  const result = compileCostControlAlphaCandidate(conflicting, {
    decisionAt: "2026-09-13T13:00:00Z",
    maxInputRows: 600,
  });
  if (
    result.recommendation !== "CONTAMINATED_EVIDENCE" ||
    result.rankedOpportunities.length !== 0 ||
    new Set(result.rankedOpportunities.map((row) => row.opportunityId)).size !==
      result.rankedOpportunities.length ||
    result.shadow_only !== true || result.live_execution !== false
  ) throw new Error(JSON.stringify(result));
});

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
  provenance: {
    sourceObservationId: `observation-${groupId}`,
    sourceId: `source-${groupId}`,
    lineageId: `lineage-${groupId}`,
    rawIndependentGroup: groupId,
    sensorFamily: `family-${groupId}`,
    sensorHorizon: "FAST_5_30M",
    direction: 1,
    snapshotWindowEnd: "2026-09-13T12:58:00Z",
    snapshotGeneratedAt: "2026-09-13T12:58:00Z",
    independent: true,
  },
  reliability: value,
  snapshotAt: "2026-09-13T12:58:00Z",
  mature: true,
});
const cost = (extra: Record<string, unknown> = {}) => ({
  costConvention: "ONE_WAY_COMPONENTS_BPS",
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
  cost: cost(),
  ...extra,
});

Deno.test("ranks net edge after dynamic round-trip cost", () => {
  const result = compileCostControlAlphaCandidate(input(), { decisionAt });
  if (
    result.recommendation !== "ALLOW_EDGE" ||
    result.selectedOpportunityId !== "o1" ||
    result.roundTripCostBps !== 40 ||
    JSON.stringify(result.costComponentsBps) !==
      JSON.stringify({ spread: 20, fee: 10, depth: 10 }) ||
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

Deno.test("future evidence cannot consume bounded decision capacity", () => {
  const baseline = compileCostControlAlphaCandidate(
    input({ opportunities: [opportunity("o1")] }),
    { decisionAt, maxInputRows: 1 },
  );
  const withFuture = compileCostControlAlphaCandidate(
    input({
      opportunities: [{
        ...opportunity("future", 999),
        observedAt: "2026-09-14T00:00:00Z",
      }, opportunity("o1")],
      reliabilitySnapshots: [
        {
          ...reliability("o1", "g1"),
        },
        {
          ...reliability("o1", "future-group", 1),
          snapshotAt: "2026-09-14T00:00:00Z",
        },
      ],
    }),
    { decisionAt, maxInputRows: 1 },
  );
  const project = (
    value: ReturnType<typeof compileCostControlAlphaCandidate>,
  ) =>
    JSON.stringify({
      recommendation: value.recommendation,
      eligible: value.eligible,
      selectedOpportunityId: value.selectedOpportunityId,
      rankedOpportunities: value.rankedOpportunities,
      roundTripCostBps: value.roundTripCostBps,
    });

  if (
    project(baseline) !== project(withFuture) ||
    withFuture.futureTelemetry.futureEvidenceCount !== 2
  ) throw new Error(JSON.stringify(withFuture));
});

Deno.test("rejects unverifiable and conflicting provenance without duplicate opportunities", () => {
  const base = input();
  const evidence = {
    ...base,
    reliabilitySnapshots: [
      ...base.reliabilitySnapshots,
      {
        ...reliability("o1", "g1", 0.1),
        provenance: {
          ...reliability("o1", "g1").provenance,
          lineageId: "contradictory-lineage",
        },
      },
      {
        ...reliability("o2", "alias", 1),
        provenance: {
          ...reliability("o2", "g2").provenance,
          rawIndependentGroup: "g2",
          sourceObservationId: "observation-g2",
        },
      },
    ],
  };
  const result = compileCostControlAlphaCandidate(evidence, { decisionAt });
  const permuted = compileCostControlAlphaCandidate({
    ...evidence,
    reliabilitySnapshots: [...evidence.reliabilitySnapshots].reverse(),
  }, { decisionAt });
  if (
    result.recommendation !== "CONTAMINATED_EVIDENCE" ||
    result.rankedOpportunities.length !== 0 ||
    result.matureIndependentGroupCount !== 2 ||
    !result.reasons.includes("conflicting reliability snapshots") ||
    JSON.stringify(result) !== JSON.stringify(permuted)
  ) throw new Error(JSON.stringify(result));
});

Deno.test("caller independence cannot replace exact source lineage binding", () => {
  const value = input();
  const result = compileCostControlAlphaCandidate({
    ...value,
    reliabilitySnapshots: [{
      ...value.reliabilitySnapshots[0],
      provenance: {
        ...value.reliabilitySnapshots[0].provenance,
        sourceObservationId: "unregistered-observation",
      },
    }, value.reliabilitySnapshots[1]],
  }, { decisionAt });
  if (
    result.recommendation !== "INSUFFICIENT_LAGGED_EVIDENCE" ||
    result.invalidEvidenceCount !== 1 ||
    result.matureIndependentGroupCount !== 1
  ) throw new Error(JSON.stringify(result));
});

Deno.test("canonical micro aliases cannot inflate independent group maturity", () => {
  const value = input();
  const aliases = ["micro_velocity", "micro_volume"];
  const aliased = {
    ...value,
    sourceObservations: value.sourceObservations.map((source, index) => ({
      ...source,
      observationId: `observation-${aliases[index]}`,
      independentGroup: aliases[index],
      sensorFamily: `family-${aliases[index]}`,
    })),
    reliabilitySnapshots: value.reliabilitySnapshots.map((row, index) => ({
      ...row,
      groupId: aliases[index],
      provenance: {
        ...row.provenance,
        sourceObservationId: `observation-${aliases[index]}`,
        rawIndependentGroup: aliases[index],
        sensorFamily: `family-${aliases[index]}`,
      },
    })),
  };
  const result = compileCostControlAlphaCandidate(aliased, { decisionAt });
  if (
    result.matureIndependentGroupCount !== 1 ||
    result.recommendation !== "INSUFFICIENT_LAGGED_EVIDENCE"
  ) throw new Error(JSON.stringify(result));
});

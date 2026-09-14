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
      reliability: .8,
      snapshotAt: "2026-09-13T12:58:00Z",
      mature: true,
    },
    {
      opportunityId: "o2",
      groupId: "g2",
      reliability: .8,
      snapshotAt: "2026-09-13T12:58:00Z",
      mature: true,
    },
  ],
  cost: {
    asOf: "2026-09-13T12:59:00Z",
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
  }, options);
  const project = (
    value: ReturnType<typeof compileCostControlAlphaCandidate>,
  ) =>
    JSON.stringify({
      recommendation: value.recommendation,
      selectedOpportunityId: value.selectedOpportunityId,
      rankedOpportunities: value.rankedOpportunities,
      roundTripCostBps: value.roundTripCostBps,
    });
  if (
    project(baseline) !== project(replay) ||
    replay.futureTelemetry.futureEvidenceCount !== 1 ||
    baseline.recommendation !== "ALLOW_EDGE"
  ) {
    throw new Error("replay projection changed");
  }
});

import { compileExpectedEdgeAlphaCandidate } from "../../../supabase/functions/_shared/evolution_candidates/expected-edge-alpha-expected-edge.ts";

const decisionAt = "2026-09-13T13:00:00Z";
const source = (id: string, extra: Record<string, unknown> = {}) => ({
  observationId: id,
  providerId: `p-${id}`,
  sensorFamily: "stress",
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
  horizon: "300s",
  cadenceSeconds: 300,
  provenance: "stress-reliability",
  snapshotAt: "2026-09-13T12:58:00Z",
  expectedMoveBps: 100,
  reliability: 1,
  uncertaintyBps: 0,
  mature: true,
  ...extra,
});
const validCost = (extra: Record<string, unknown> = {}) => ({
  asOf: "2026-09-13T12:59:00Z",
  sourceId: "stress-book",
  cadenceSeconds: 300,
  spreadBps: 1,
  feeBps: 1,
  slippageBps: 1,
  fillability: 1,
  ...extra,
});
const validInput = (extra: Record<string, unknown> = {}) => ({
  sourceObservations: [source("o0"), source("o1")],
  reliabilitySnapshots: [reliability("o0", "g0"), reliability("o1", "g1")],
  cost: validCost(),
  ...extra,
});

Deno.test("stress input is bounded, deterministic, and shadow-only", () => {
  const sourceObservations = Array.from(
    { length: 3_000 },
    (_, index) => source(`o${index}`),
  );
  const reliabilitySnapshots = [
    reliability("o0", "g0"),
    reliability("o1", "g1"),
  ];
  const value = validInput({ sourceObservations, reliabilitySnapshots });
  const result = compileExpectedEdgeAlphaCandidate(value, {
    decisionAt,
    maxInputRows: 100,
    maxContributions: 10,
  });
  const reversed = compileExpectedEdgeAlphaCandidate({
    ...value,
    sourceObservations: [...sourceObservations].reverse(),
    reliabilitySnapshots: [...reliabilitySnapshots].reverse(),
  }, { decisionAt, maxInputRows: 100, maxContributions: 10 });
  if (
    JSON.stringify(result) !== JSON.stringify(reversed) ||
    !result.truncated || result.recommendation === "COST_UNAVAILABLE" ||
    result.eligible || result.contributions.length > 10 ||
    result.shadow_only !== true || result.live_execution !== false ||
    result.canonical_mutation !== false || result.promotionReady !== false
  ) throw new Error(JSON.stringify(result));
});

Deno.test("stress matrix fails closed for malformed, extreme, and future evidence", () => {
  const base = validInput();
  const cases = [
    { ...base, sourceObservations: [null] },
    {
      ...base,
      reliabilitySnapshots: [{
        ...reliability("o0", "g0"),
        uncertaintyBps: Number.MAX_VALUE,
      }],
    },
    { ...base, cost: validCost({ fillability: 0 }) },
    { ...base, cost: validCost({ spreadBps: Number.NaN }) },
    {
      ...base,
      sourceObservations: [source("o0", {
        observedAt: "2026-09-14T00:00:00Z",
      })],
    },
    { ...base, reliabilitySnapshots: [] },
    {
      ...base,
      sourceObservations: [source("o0", {
        horizon: "24h",
        cadenceSeconds: 3600,
        evaluationEndAt: "2026-09-14T12:55:00Z",
      })],
    },
    { ...base, eventAt: "2026-09-14T00:00:00Z", eventCadenceSeconds: 300 },
    { ...base, cost: validCost({ asOf: "2026-09-14T00:00:00Z" }) },
    {
      ...base,
      sourceObservations: [source("o0"), source("o0", { providerId: "other" })],
    },
  ];
  for (const value of cases) {
    const result = compileExpectedEdgeAlphaCandidate(value, { decisionAt });
    if (
      result.eligible || result.shadow_only !== true ||
      result.live_execution !== false || result.canonical_mutation !== false ||
      result.promotionReady !== false
    ) throw new Error(JSON.stringify(result));
  }
});

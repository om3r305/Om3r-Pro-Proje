import { compileCapabilityGapMarketL2 } from "../../../supabase/functions/_shared/evolution_candidates/capability-gap-market-l2.ts";

const base = {
  venue: "binance",
  symbol: "BTCUSDT",
  collectorSessionId: "replay-session",
  connectionGeneration: 1,
  syncGeneration: 1,
  sourceLineage: { rawHash: "immutable-1" },
};
const snapshot = {
  ...base,
  kind: "snapshot",
  arrivalSeq: 1,
  sourceEventId: "s1",
  observedAt: "2026-09-13T12:59:00Z",
  receivedAt: "2026-09-13T12:59:00Z",
  lastUpdateId: "10",
  bids: [{ price: "99", size: "2" }],
  asks: [{ price: "101", size: "2" }],
};
const diff = {
  ...base,
  kind: "diff",
  arrivalSeq: 2,
  sourceEventId: "d1",
  observedAt: "2026-09-13T12:59:30Z",
  receivedAt: "2026-09-13T12:59:30Z",
  firstUpdateId: "11",
  finalUpdateId: "11",
  bidMutations: [],
  askMutations: [],
};
const opts = {
  decisionAt: "2026-09-13T13:00:00Z",
  side: "BUY" as const,
  requestedNotionalUsd: 100,
  feeBps: 5,
};

Deno.test("immutable replay excludes appended future rows from the decision projection", () => {
  const baseline = compileCapabilityGapMarketL2(
    { evidence: [snapshot, diff] },
    opts,
  );
  const future = {
    ...diff,
    arrivalSeq: 3,
    sourceEventId: "future",
    observedAt: "2026-09-14T00:00:00Z",
    receivedAt: "2026-09-14T00:00:00Z",
  };
  const replay = compileCapabilityGapMarketL2({
    evidence: [future, diff, snapshot],
  }, opts);
  const project = (value: ReturnType<typeof compileCapabilityGapMarketL2>) =>
    JSON.stringify({
      classification: value.classification,
      book: value.book,
      cost: value.cost,
      provenance: value.provenance,
      blockers: value.blockers,
    });
  if (
    project(baseline) !== project(replay) ||
    replay.futureTelemetry.futureEvidenceCount !== 1
  ) {
    throw new Error(JSON.stringify({ baseline, replay }));
  }
});

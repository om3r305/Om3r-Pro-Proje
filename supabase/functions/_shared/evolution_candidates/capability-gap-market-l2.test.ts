import { compileCapabilityGapMarketL2 } from "./capability-gap-market-l2.ts";

const options = {
  decisionAt: "2026-09-13T13:00:00Z",
  side: "BUY" as const,
  requestedNotionalUsd: 100,
  feeBps: 5,
};
const snapshot = (extra: Record<string, unknown> = {}) => ({
  kind: "snapshot",
  venue: "binance",
  symbol: "BTCUSDT",
  collectorSessionId: "session-a",
  connectionGeneration: 1,
  syncGeneration: 1,
  arrivalSeq: 1,
  sourceEventId: "snapshot-1",
  observedAt: "2026-09-13T12:59:00Z",
  receivedAt: "2026-09-13T12:59:00Z",
  lastUpdateId: "10",
  bids: [{ price: "99", size: "2" }],
  asks: [{ price: "101", size: "2" }],
  sourceLineage: { rawHash: "hash-1", capture: "capture-a" },
  ...extra,
});
const diff = (extra: Record<string, unknown> = {}) => ({
  kind: "diff",
  venue: "binance",
  symbol: "BTCUSDT",
  collectorSessionId: "session-a",
  connectionGeneration: 1,
  syncGeneration: 1,
  arrivalSeq: 2,
  sourceEventId: "diff-1",
  observedAt: "2026-09-13T12:59:30Z",
  receivedAt: "2026-09-13T12:59:30Z",
  firstUpdateId: "11",
  finalUpdateId: "11",
  bidMutations: [],
  askMutations: [{ price: "101", size: "3" }],
  sourceLineage: { rawHash: "hash-2", capture: "capture-a" },
  ...extra,
});
const compile = (evidence: unknown[]) =>
  compileCapabilityGapMarketL2({ evidence }, options);

Deno.test("compiles synchronized point-in-time depth and walks the book", () => {
  const result = compile([snapshot(), diff()]);
  if (
    result.classification !== "HEALTHY" || !result.healthy ||
    result.cost?.fillable !== true || result.cost.levelsConsumed !== 1 ||
    result.book.bestAsk !== "101" || result.shadow_only !== true ||
    result.live_execution !== false || result.promotionReady !== false
  ) {
    throw new Error(JSON.stringify(result));
  }
});

Deno.test("rejects missing lineage, gaps, crossed books, stale books, and malformed levels", () => {
  const cases = [
    [snapshot({ sourceLineage: undefined }), "invalid"],
    [snapshot({ bids: [{ price: "bad", size: "1" }] }), "invalid"],
    [snapshot(), "gap", {
      ...diff(),
      firstUpdateId: "13",
      finalUpdateId: "13",
    }],
    [snapshot({ bids: [{ price: "102", size: "1" }] }), "crossed", {
      ...diff(),
      asks: undefined,
      askMutations: [],
    }],
    [snapshot({ receivedAt: "2026-09-13T10:00:00Z" }), "stale"],
  ] as Array<[Record<string, unknown>, string, Record<string, unknown>?]>;
  for (const [base, expected, extra] of cases) {
    const result = expected === "gap"
      ? compile([base, extra!])
      : expected === "crossed"
      ? compile([base, extra!])
      : compile([base]);
    if (expected === "invalid" && result.invalidEvidenceCount !== 1) {
      throw new Error(JSON.stringify(result));
    }
    if (
      expected !== "invalid" &&
      !result.blockers.some((blocker) => blocker.includes(expected))
    ) {
      throw new Error(`${expected}: ${JSON.stringify(result)}`);
    }
  }
});

Deno.test("future evidence is telemetry only and permutations are deterministic", () => {
  const future = {
    ...diff(),
    arrivalSeq: 3,
    sourceEventId: "future",
    receivedAt: "2026-09-14T00:00:00Z",
    observedAt: "2026-09-14T00:00:00Z",
  };
  const a = compile([snapshot(), diff(), future]);
  const b = compile([future, diff(), snapshot()]);
  if (
    JSON.stringify(a) !== JSON.stringify(b) ||
    a.futureTelemetry.futureEvidenceCount !== 1
  ) throw new Error(JSON.stringify({ a, b }));
});

Deno.test("insufficient visible depth never produces a healthy report", () => {
  const result = compileCapabilityGapMarketL2({
    evidence: [
      snapshot({
        bids: [{ price: "99", size: "0.1" }],
        asks: [{ price: "101", size: "0.1" }],
      }),
    ],
  }, { ...options, requestedNotionalUsd: 1000 });
  if (
    result.healthy || result.classification === "HEALTHY" ||
    !result.blockers.includes("insufficient visible depth")
  ) throw new Error(JSON.stringify(result));
});

Deno.test("normalizes equivalent decimal level keys and rejects conflicting duplicates", () => {
  const normalized = compile([
    snapshot({
      bids: [{ price: "099.00", size: "1" }, { price: "99.0", size: "1.0" }],
      asks: [{ price: "101.00", size: "2" }],
    }),
  ]);
  if (
    normalized.classification !== "HEALTHY" ||
    normalized.book.visibleBidLevels !== 1
  ) throw new Error(JSON.stringify(normalized));
  const conflicting = compile([
    snapshot({
      bids: [{ price: "99", size: "1" }, { price: "99.0", size: "2" }],
    }),
  ]);
  if (conflicting.invalidEvidenceCount !== 1 || conflicting.healthy) {
    throw new Error(JSON.stringify(conflicting));
  }
});

Deno.test("invalid cost bounds fail closed without invoking arithmetic", () => {
  for (
    const override of [
      { requestedNotionalUsd: 0 },
      { requestedNotionalUsd: Infinity },
      { feeBps: -1 },
      { side: "HOLD" as unknown as "BUY" },
    ]
  ) {
    const result = compileCapabilityGapMarketL2(
      { evidence: [snapshot()] },
      { ...options, ...override },
    );
    if (result.healthy || !result.blockers.includes("invalid cost bounds")) {
      throw new Error(JSON.stringify(result));
    }
  }
});

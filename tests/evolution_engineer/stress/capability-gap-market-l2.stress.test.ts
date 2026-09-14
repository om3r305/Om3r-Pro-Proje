import { compileCapabilityGapMarketL2 } from "../../../supabase/functions/_shared/evolution_candidates/capability-gap-market-l2.ts";

Deno.test("bounded market L2 compiler fails closed on oversized adversarial evidence", () => {
  const base = {
    kind: "snapshot",
    venue: "binance",
    symbol: "BTCUSDT",
    collectorSessionId: "stress",
    connectionGeneration: 1,
    syncGeneration: 1,
    arrivalSeq: 1,
    sourceEventId: "s",
    observedAt: "2026-09-13T12:59:00Z",
    receivedAt: "2026-09-13T12:59:00Z",
    lastUpdateId: "1",
    bids: [{ price: "99", size: "2" }],
    asks: [{ price: "101", size: "2" }],
    sourceLineage: { rawHash: "stress" },
  };
  const evidence = Array.from({ length: 10_000 }, (_, index) => ({
    ...base,
    sourceEventId: `row-${index}`,
    arrivalSeq: index + 1,
  }));
  const result = compileCapabilityGapMarketL2({ evidence }, {
    decisionAt: "2026-09-13T13:00:00Z",
    side: "BUY",
    requestedNotionalUsd: 100,
    feeBps: 5,
    maxInputRows: 10,
    maxLevels: 2,
  });
  if (
    !result.truncated || result.invalidEvidenceCount > 10 ||
    result.shadow_only !== true || result.live_execution !== false ||
    result.promotionReady !== false || result.book.visibleBidLevels > 2
  ) {
    throw new Error(JSON.stringify(result));
  }
});

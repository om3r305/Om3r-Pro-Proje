import {
  assessTreasuryRuntimeEvidence,
  missingTreasuryPositionEdgeAssets,
} from "./evolution_treasury_runtime.ts";

Deno.test("Treasury runtime evidence is actionable only with fresh edge, evaluation and market mark", () => {
  const result = assessTreasuryRuntimeEvidence({
    nowIso: "2026-09-11T14:00:00Z",
    edgeObservedAt: "2026-09-11T13:58:30Z",
    edgeEvaluatedAt: "2026-09-11T13:59:20Z",
    markObservedAt: "2026-09-11T13:59:50Z",
  });
  if (!result.actionable || result.reasons.length) throw new Error(JSON.stringify(result));
});

Deno.test("Treasury allows a near-three-minute evaluation created by the two-minute challenger cadence", () => {
  const result = assessTreasuryRuntimeEvidence({
    nowIso: "2026-09-11T14:00:00Z",
    edgeObservedAt: "2026-09-11T13:57:10Z",
    edgeEvaluatedAt: "2026-09-11T13:59:55Z",
    markObservedAt: "2026-09-11T13:59:55Z",
  });
  if (!result.actionable || result.reasons.length) throw new Error(JSON.stringify(result));
});

Deno.test("Treasury rejects an edge whose challenger evaluation exceeds the bounded cadence budget", () => {
  const result = assessTreasuryRuntimeEvidence({
    nowIso: "2026-09-11T14:00:00Z",
    edgeObservedAt: "2026-09-11T13:56:50Z",
    edgeEvaluatedAt: "2026-09-11T13:59:55Z",
    markObservedAt: "2026-09-11T13:59:55Z",
  });
  if (result.actionable || !result.reasons.some((reason) => reason.includes("too late"))) throw new Error(JSON.stringify(result));
});

Deno.test("Treasury rejects stale market marks instead of treating them as fresh entry evidence", () => {
  const result = assessTreasuryRuntimeEvidence({
    nowIso: "2026-09-11T14:00:00Z",
    edgeObservedAt: "2026-09-11T13:59:00Z",
    edgeEvaluatedAt: "2026-09-11T13:59:20Z",
    markObservedAt: "2026-09-11T13:50:00Z",
  });
  if (result.actionable || !result.reasons.some((reason) => reason.includes("market mark is stale"))) throw new Error(JSON.stringify(result));
});

Deno.test("Treasury rejects stale edge even when current market mark is fresh", () => {
  const result = assessTreasuryRuntimeEvidence({
    nowIso: "2026-09-11T14:00:00Z",
    edgeObservedAt: "2026-09-11T13:50:00Z",
    edgeEvaluatedAt: "2026-09-11T13:50:30Z",
    markObservedAt: "2026-09-11T13:59:55Z",
  });
  if (result.actionable || !result.reasons.some((reason) => reason.includes("edge is stale"))) throw new Error(JSON.stringify(result));
});

Deno.test("Treasury separately fetches open-position edges that fall outside the global edge window", () => {
  const missing = missingTreasuryPositionEdgeAssets(
    ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BTCUSDT"],
    ["BTCUSDT", "SOLUSDT", "XRPUSDT"],
  );
  if (missing.length !== 1 || missing[0] !== "ETHUSDT") throw new Error(JSON.stringify(missing));
});

import { compileCapabilityGapAlphaCompiler } from "./capability-gap-alpha-compiler.ts";

const row = (overrides: Record<string, unknown> = {}) => ({
  providerId: "alpha",
  rowId: "r1",
  completedAt: "2026-09-13T12:00:00Z",
  freshnessAt: "2026-09-13T12:00:00Z",
  health: "HEALTHY",
  status: "COMPLETED",
  failures: [],
  ...overrides,
});
const now = "2026-09-13T13:00:00Z";

Deno.test("non-array input fails closed", () => {
  const result = compileCapabilityGapAlphaCompiler(null, { observedAt: now });
  if (
    result.classification !== "BLOCKED" ||
    !result.blockers.includes("input must be an array")
  ) throw new Error(JSON.stringify(result));
});

Deno.test("strict UTC accepts zero through three fractions and rejects offsets", () => {
  const valid = [0, 1, 2, 3].map((n) =>
    row({ completedAt: `2026-09-13T12:00:00${n ? "." + "1".repeat(n) : ""}Z` })
  );
  for (const item of valid) {
    if (
      compileCapabilityGapAlphaCompiler([item], { observedAt: now })
        .invalidEvidenceCount
    ) throw new Error("valid UTC rejected");
  }
  const result = compileCapabilityGapAlphaCompiler([
    row({ completedAt: "2026-09-13T12:00:00+01:00" }),
  ], { observedAt: now });
  if (result.invalidEvidenceCount !== 1) throw new Error("offset was accepted");
});

Deno.test("future rows are excluded without consuming decision capacity", () => {
  const result = compileCapabilityGapAlphaCompiler([
    row(),
    row({ rowId: "future", completedAt: "2026-09-14T00:00:00Z" }),
  ], { observedAt: now, maxRows: 1 });
  if (
    result.processedDecisionRowCount !== 1 ||
    result.futureTelemetry.futureEvidenceCount !== 1
  ) throw new Error(JSON.stringify(result));
});

Deno.test("future freshness and future-only providers are isolated", () => {
  const result = compileCapabilityGapAlphaCompiler([
    row(),
    row({
      providerId: "futureonly",
      rowId: "future-freshness",
      freshnessAt: "2026-09-14T00:00:00Z",
    }),
  ], { observedAt: now, maxProviders: 1 });
  if (
    result.providers.some((provider) => provider.providerId === "futureonly") ||
    result.futureTelemetry.futureEvidenceCount !== 1 ||
    result.providers[0]?.freshnessAt !== "2026-09-13T12:00:00Z"
  ) throw new Error(JSON.stringify(result));
});

Deno.test("malformed freshness and unsupported status fail closed", () => {
  const result = compileCapabilityGapAlphaCompiler([
    row({ freshnessAt: "2026-09-13T12:00:00+01:00" }),
    row({ rowId: "unsupported", status: "UNKNOWN_STATUS" }),
  ], { observedAt: now });
  if (
    result.invalidEvidenceCount !== 2 ||
    result.providers[0]?.invalidEvidenceCount !== 2 ||
    !result.blockers.includes("invalid evidence present")
  ) throw new Error(JSON.stringify(result));
});

Deno.test("malformed nested failures are invalid evidence", () => {
  const result = compileCapabilityGapAlphaCompiler([
    row({ failures: [null, { id: "missing-message" }] }),
    row({ rowId: "primitive-failures", failures: "not-an-array" }),
  ], { observedAt: now });
  if (
    result.invalidEvidenceCount !== 2 ||
    result.providers[0]?.invalidEvidenceCount !== 2 ||
    result.classification !== "BLOCKED"
  ) throw new Error(JSON.stringify(result));
});

Deno.test("conflicts and malformed rows are found before maxRows truncation", () => {
  const result = compileCapabilityGapAlphaCompiler([
    row(),
    row({ rowId: "conflict", health: "DEGRADED" }),
    row({ rowId: "bad", completedAt: "not-a-date" }),
  ], { observedAt: now, maxRows: 1 });
  if (
    !result.blockers.includes(
      "ambiguous conflicting equal-timestamp evidence",
    ) || result.invalidEvidenceCount !== 1
  ) throw new Error(JSON.stringify(result));
});

Deno.test("lease skips and exact duplicate failures are accounted once", () => {
  const failure = { id: "timeout", message: "timeout" };
  const result = compileCapabilityGapAlphaCompiler([
    row({ status: "LEASE_SKIPPED", failures: [failure, failure] }),
  ], { observedAt: now });
  if (
    result.providers[0]?.leaseSkippedCount !== 1 ||
    result.providers[0]?.failedCount !== 1
  ) throw new Error(JSON.stringify(result));
});

Deno.test("failure messages are part of deterministic identities", () => {
  const result = compileCapabilityGapAlphaCompiler([
    row({
      failures: [
        { id: "same", message: "z-message" },
        { id: "same", message: "a-message" },
        { id: "same", message: "z-message" },
      ],
    }),
  ], { observedAt: now });
  if (
    result.providers[0]?.failedCount !== 2 ||
    JSON.stringify(result.providers[0]?.recentFailures) !==
      JSON.stringify(["a-message", "z-message"])
  ) throw new Error(JSON.stringify(result));
});

Deno.test("non-finite limits use bounded defaults", () => {
  const result = compileCapabilityGapAlphaCompiler(
    [row(), row({ rowId: "second" })],
    { observedAt: now, maxRows: Number.NaN, maxProviders: Infinity },
  );
  if (result.processedDecisionRowCount !== 2 || result.providers.length !== 1) {
    throw new Error(JSON.stringify(result));
  }
});

Deno.test("overflow remains bounded and separate from invalid evidence", () => {
  const result = compileCapabilityGapAlphaCompiler(
    Array.from(
      { length: 20 },
      (_, i) => row({ rowId: `r${i}`, providerId: `p${i}` }),
    ),
    { observedAt: now, maxRows: 2, maxProviders: 2, maxInputRows: 50 },
  );
  if (
    result.processedDecisionRowCount > 2 || result.providers.length > 2 ||
    !result.rowsExceeded || result.invalidEvidenceCount !== 0 ||
    !result.blockers.includes(
      "missing prospective multi-window shadow A/B evidence",
    ) ||
    result.shadow_only !== true || result.live_execution !== false ||
    result.promotionReady !== false
  ) throw new Error(JSON.stringify(result));
});

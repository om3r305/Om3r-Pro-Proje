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
    !result.rowsExceeded || result.invalidEvidenceCount !== 0
  ) throw new Error(JSON.stringify(result));
});

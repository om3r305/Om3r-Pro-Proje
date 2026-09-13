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
    result.providers[0]?.freshnessAt !== "2026-09-13T12:00:00.000Z"
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

Deno.test("missing failures remain empty but explicit null fails closed", () => {
  const withoutFailures: Record<string, unknown> = row({
    rowId: "missing-failures",
  });
  delete withoutFailures.failures;
  const missing = compileCapabilityGapAlphaCompiler([withoutFailures], {
    observedAt: now,
  });
  const explicitNull = compileCapabilityGapAlphaCompiler([
    row({ rowId: "null-failures", failures: null }),
  ], { observedAt: now });
  if (
    missing.invalidEvidenceCount !== 0 ||
    missing.providers[0]?.failedCount !== 0 ||
    explicitNull.invalidEvidenceCount !== 1 ||
    explicitNull.providers[0]?.invalidEvidenceCount !== 1 ||
    explicitNull.classification !== "BLOCKED" ||
    !explicitNull.blockers.includes("invalid evidence present")
  ) throw new Error(JSON.stringify({ missing, explicitNull }));
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

Deno.test("canonical instants detect equivalent-spelling conflicts deterministically", () => {
  const equivalent = [
    row({ completedAt: "2026-09-13T12:00:00Z" }),
    row({ completedAt: "2026-09-13T12:00:00.0Z" }),
  ];
  const collapsed = compileCapabilityGapAlphaCompiler(equivalent, {
    observedAt: now,
  });
  if (
    collapsed.invalidEvidenceCount !== 0 ||
    collapsed.processedDecisionRowCount !== 1
  ) throw new Error(JSON.stringify(collapsed));

  const conflicting = [
    row({
      completedAt: "2026-09-13T12:00:00Z",
      freshnessAt: "2026-09-13T11:00:00Z",
    }),
    row({
      completedAt: "2026-09-13T12:00:00.0Z",
      freshnessAt: "2026-09-13T12:00:00Z",
    }),
  ];
  for (const input of [conflicting, [...conflicting].reverse()]) {
    const result = compileCapabilityGapAlphaCompiler(input, {
      observedAt: now,
    });
    if (
      !result.blockers.includes(
        "ambiguous conflicting equal-timestamp evidence",
      )
    ) throw new Error(JSON.stringify(result));
  }
});

Deno.test("same-instant conflict diagnostics remain input-order invariant", () => {
  const rows = [
    row({
      rowId: "first",
      completedAt: "2026-09-13T12:00:00Z",
      freshnessAt: "2026-09-13T11:00:00Z",
      health: "HEALTHY",
      failures: [{ id: "a", message: "alpha" }],
    }),
    row({
      rowId: "second",
      completedAt: "2026-09-13T12:00:00.0Z",
      freshnessAt: "2026-09-13T12:00:00Z",
      health: "DEGRADED",
      failures: [{ id: "b", message: "beta" }],
    }),
  ];
  const forward = compileCapabilityGapAlphaCompiler(rows, { observedAt: now });
  const backward = compileCapabilityGapAlphaCompiler([...rows].reverse(), {
    observedAt: now,
  });
  if (
    forward.blockers.join(",") !== backward.blockers.join(",") ||
    JSON.stringify(forward.providers) !== JSON.stringify(backward.providers)
  ) {
    throw new Error(`${JSON.stringify(forward)} | ${JSON.stringify(backward)}`);
  }
});

Deno.test("lease-skip aliases are equivalent in same-instant fingerprints", () => {
  const aliases = ["LEASE_SKIPPED", "SKIPPED_LEASE", "LEASE_UNAVAILABLE"];
  const result = compileCapabilityGapAlphaCompiler(
    aliases.map((status) => row({ rowId: "same-row", status })),
    { observedAt: now },
  );
  if (
    result.blockers.includes(
      "ambiguous conflicting equal-timestamp evidence",
    ) ||
    result.providers[0]?.leaseSkippedCount !== 1
  ) throw new Error(JSON.stringify(result));

  const conflict = compileCapabilityGapAlphaCompiler([
    row({ rowId: "same-row", status: "LEASE_SKIPPED" }),
    row({ rowId: "same-row", status: "COMPLETED" }),
  ], { observedAt: now });
  if (
    !conflict.blockers.includes(
      "ambiguous conflicting equal-timestamp evidence",
    )
  ) throw new Error(JSON.stringify(conflict));
});

Deno.test("future-only providers do not consume provider diagnostics or truncation", () => {
  const baseline = compileCapabilityGapAlphaCompiler([row()], {
    observedAt: now,
    maxProviders: 1,
  });
  const withFutureProvider = compileCapabilityGapAlphaCompiler([
    row(),
    row({
      providerId: "futureonly",
      rowId: "future-provider",
      completedAt: "2026-09-14T00:00:00Z",
      freshnessAt: "2026-09-14T00:00:00Z",
    }),
  ], { observedAt: now, maxProviders: 1 });
  if (
    JSON.stringify(baseline.providers) !==
      JSON.stringify(withFutureProvider.providers) ||
    baseline.providerDiagnosticsTruncated !==
      withFutureProvider.providerDiagnosticsTruncated
  ) throw new Error(JSON.stringify(withFutureProvider));
});

Deno.test("provider candidates include recognized providers before maxRows slicing", () => {
  const rows = [
    row({
      providerId: "alpha",
      rowId: "r1",
      completedAt: "2026-09-13T12:00:00Z",
    }),
    row({
      providerId: "alpha",
      rowId: "r2",
      completedAt: "2026-09-13T12:00:01Z",
    }),
    row({
      providerId: "beta",
      rowId: "r3",
      completedAt: "2026-09-13T12:00:02Z",
    }),
    row({
      providerId: "gamma",
      rowId: "r4",
      completedAt: "2026-09-13T12:00:03Z",
    }),
  ];
  const result = compileCapabilityGapAlphaCompiler(rows, {
    observedAt: now,
    maxRows: 2,
    maxProviders: 2,
  });
  if (
    result.processedDecisionRowCount !== 2 ||
    !result.providerDiagnosticsTruncated ||
    JSON.stringify(result.providers.map((provider) => provider.providerId)) !==
      JSON.stringify(["alpha", "beta"]) ||
    result.providers[0]?.classification !== "UNKNOWN"
  ) {
    throw new Error(JSON.stringify(result));
  }
});

Deno.test("non-finite limits use bounded defaults", () => {
  const result = compileCapabilityGapAlphaCompiler(
    [
      row(),
      row({
        rowId: "second",
        completedAt: "2026-09-13T12:00:01Z",
      }),
    ],
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

Deno.test("invalid limit values use the default bounded options", () => {
  const input = [
    row({
      providerId: "alpha",
      rowId: "a",
      completedAt: "2026-09-13T12:00:00Z",
    }),
    row({
      providerId: "beta",
      rowId: "b",
      completedAt: "2026-09-13T12:00:01Z",
    }),
    row({
      providerId: "gamma",
      rowId: "c",
      completedAt: "2026-09-13T12:00:02Z",
    }),
  ];
  const baseline = compileCapabilityGapAlphaCompiler(input, {
    observedAt: now,
  });
  for (const value of [NaN, Infinity, -Infinity, 0, -1, 1.5]) {
    const result = compileCapabilityGapAlphaCompiler(input, {
      observedAt: now,
      maxRows: value,
      maxProviders: value,
      maxInputRows: value,
    });
    const defaultResult = compileCapabilityGapAlphaCompiler(input, {
      observedAt: now,
    });
    if (
      result.processedDecisionRowCount !==
        defaultResult.processedDecisionRowCount ||
      result.providers.length !== baseline.providers.length ||
      result.providerDiagnosticsTruncated ||
      result.inputEnvelopeTruncated !== baseline.inputEnvelopeTruncated ||
      result.invalidEvidenceCount !== 0
    ) throw new Error(`invalid option changed defaults: ${value}`);
  }
});

Deno.test("future freshness is telemetry-only and malformed freshness is invalid", () => {
  const baseline = compileCapabilityGapAlphaCompiler([row()], {
    observedAt: now,
    maxProviders: 1,
  });
  const futureFreshness = compileCapabilityGapAlphaCompiler([
    row({
      rowId: "future-freshness",
      freshnessAt: "2026-09-14T00:00:00Z",
    }),
  ], { observedAt: now, maxProviders: 1 });
  if (
    futureFreshness.futureTelemetry.futureEvidenceCount !== 1 ||
    futureFreshness.providers.length !== 0 ||
    futureFreshness.invalidEvidenceCount !== 0 ||
    futureFreshness.processedDecisionRowCount !== 0 ||
    futureFreshness.providerDiagnosticsTruncated ||
    futureFreshness.decisionTruncated
  ) throw new Error(JSON.stringify(futureFreshness));
  const malformed = compileCapabilityGapAlphaCompiler([
    row({ freshnessAt: "2026-09-14T00:00:00+01:00" }),
  ], { observedAt: now });
  if (
    malformed.futureTelemetry.futureEvidenceCount !== 0 ||
    malformed.invalidEvidenceCount !== 1 ||
    malformed.providers[0]?.invalidEvidenceCount !== 1
  ) throw new Error(JSON.stringify({ baseline, malformed }));
});

Deno.test("provider diagnostics are independent from maxRows", () => {
  const rows = [
    row({ providerId: "alpha", rowId: "a" }),
    row({
      providerId: "beta",
      rowId: "b",
      completedAt: "2026-09-13T12:00:01Z",
    }),
    row({
      providerId: "gamma",
      rowId: "c",
      completedAt: "2026-09-13T12:00:02Z",
    }),
    row({
      providerId: "validonly",
      rowId: "bad",
      completedAt: "2026-09-13T12:00:03Z",
      failures: [{ id: "missing-message" }],
    }),
  ];
  const result = compileCapabilityGapAlphaCompiler(rows, {
    observedAt: now,
    maxRows: 1,
    maxProviders: 3,
  });
  if (
    JSON.stringify(result.providers.map((provider) => provider.providerId)) !==
      JSON.stringify(["alpha", "beta", "gamma"]) ||
    result.providers[0]?.classification !== "UNKNOWN" ||
    result.providerDiagnosticsTruncated !== true ||
    result.processedDecisionRowCount !== 1 ||
    result.invalidEvidenceCount !== 1
  ) throw new Error(JSON.stringify(result));
});

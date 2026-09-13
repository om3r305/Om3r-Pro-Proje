import { compileCapabilityGapAlphaCompiler } from "../../../supabase/functions/_shared/evolution_candidates/capability-gap-alpha-compiler.ts";

const make = (
  id: string,
  providerId = "alpha",
  extra: Record<string, unknown> = {},
) => ({
  providerId,
  rowId: id,
  completedAt: "2026-09-13T12:00:00Z",
  freshnessAt: "2026-09-13T12:00:00Z",
  health: "HEALTHY",
  status: "COMPLETED",
  failures: [],
  ...extra,
});

type StressFailure = {
  id: string;
  message: string;
};

type CountableAlphaRow = {
  providerId: string;
  rowId: string;
  completedAt: string;
  freshnessAt: string;
  failures: StressFailure[];
};

const completeDecisionProjection = (
  result: ReturnType<typeof compileCapabilityGapAlphaCompiler>,
) =>
  JSON.stringify({
    classification: result.classification,
    providers: result.providers,
    processedDecisionRowCount: result.processedDecisionRowCount,
    rowsExceeded: result.rowsExceeded,
    decisionTruncated: result.decisionTruncated,
    invalidEvidenceCount: result.invalidEvidenceCount,
    invalidProviderCount: result.invalidProviderCount,
    blockers: result.blockers,
    inputEnvelopeTruncated: result.inputEnvelopeTruncated,
    providerDiagnosticsTruncated: result.providerDiagnosticsTruncated,
    shadow_only: result.shadow_only,
    live_execution: result.live_execution,
    promotionReady: result.promotionReady,
  });

Deno.test("stress bounds oversized evidence and deduplicates named failures", () => {
  const failureA: StressFailure = { id: "failure-a", message: "a" };
  const failureB: StressFailure = { id: "failure-b", message: "b" };
  const alphaFailureRows: CountableAlphaRow[] = [
    {
      ...make("alpha-row-1", "alpha"),
      failures: [failureA, failureA],
    },
    {
      ...make("alpha-row-2", "alpha", {
        completedAt: "2026-09-13T12:00:01Z",
        freshnessAt: "2026-09-13T12:00:01Z",
      }),
      failures: [failureB],
    },
  ];
  const conflictRows = [
    make("conflict-healthy", "conflict", {
      completedAt: "2026-09-13T12:00:00Z",
      freshnessAt: "2026-09-13T12:00:00Z",
    }),
    make("conflict-degraded", "conflict", {
      completedAt: "2026-09-13T12:00:00.0Z",
      freshnessAt: "2026-09-13T12:00:00Z",
      health: "DEGRADED",
    }),
  ];
  const validOnlyInvalidRows = [
    make("malformed", "validonly", { failures: [{ id: "broken" }] }),
    make("unsupported", "validonly", { status: "UNSUPPORTED_STATUS" }),
    make("null-failures", "validonly", { failures: null }),
  ];
  const rows: unknown[] = [
    ...alphaFailureRows,
    ...conflictRows,
    ...validOnlyInvalidRows,
    make("future-only", "futureonly", {
      completedAt: "2026-09-14T00:00:00Z",
      freshnessAt: "2026-09-14T00:00:00Z",
    }),
    make("future-freshness", "futurefresh", {
      freshnessAt: "2026-09-14T00:00:00Z",
    }),
    make("bad-provider", "9bad"),
  ];
  for (let i = 0; i < 40; i++) {
    rows.push(make(`overflow-${i}`, `z-provider-${i}`));
  }
  const inspectedRows = rows.slice(0, 12);
  const overflowRows = rows.slice(12);
  const result = compileCapabilityGapAlphaCompiler(rows, {
    observedAt: "2026-09-13T13:00:00Z",
    maxRows: 2,
    maxProviders: 3,
    maxInputRows: 12,
  });
  const rawFailureOccurrences = alphaFailureRows.flatMap((row) => row.failures)
    .length;
  const uniqueFailureIdentities = new Set(
    alphaFailureRows.flatMap((row) =>
      row.failures.map((failure) =>
        JSON.stringify({
          providerId: row.providerId,
          rowId: row.rowId,
          id: failure.id,
          message: failure.message,
        })
      )
    ),
  ).size;
  const alpha = result.providers.find((provider) =>
    provider.providerId === "alpha"
  );
  const providerIds = result.providers.map((provider) => provider.providerId);
  const reversedResult = compileCapabilityGapAlphaCompiler([
    ...inspectedRows.slice().reverse(),
    ...overflowRows.slice().reverse(),
  ], {
    observedAt: "2026-09-13T13:00:00Z",
    maxRows: 2,
    maxProviders: 3,
    maxInputRows: 12,
  });
  if (
    uniqueFailureIdentities !== 2 ||
    rawFailureOccurrences <= uniqueFailureIdentities ||
    alpha?.failedCount !== 2 ||
    alpha?.recentFailures.length !== 2 ||
    JSON.stringify(alpha?.recentFailures) !== JSON.stringify(["a", "b"]) ||
    JSON.stringify(alpha?.recentFailures) !==
      JSON.stringify(
        reversedResult.providers.find((provider) =>
          provider.providerId === "alpha"
        )?.recentFailures,
      ) ||
    alpha?.failedCount !==
      reversedResult.providers.find((provider) =>
        provider.providerId === "alpha"
      )?.failedCount
  ) {
    throw new Error(`dedupe failed: ${JSON.stringify(result)}`);
  }
  if (
    JSON.stringify(providerIds) !==
      JSON.stringify(["alpha", "conflict", "validonly"]) ||
    result.providerDiagnosticsTruncated !== true ||
    result.providers.some((provider) => provider.providerId === "futureonly") ||
    result.providers.some((provider) =>
      provider.providerId.startsWith("z-provider-")
    ) ||
    !result.blockers.includes(
      "ambiguous conflicting equal-timestamp evidence",
    ) ||
    result.invalidProviderCount !== 1 ||
    result.invalidEvidenceCount !== 4 ||
    result.futureTelemetry.futureEvidenceCount !== 2 ||
    !result.inputEnvelopeTruncated ||
    !result.blockers.includes("input envelope truncated") ||
    !result.blockers.includes(
      "input envelope limit prevents complete evidence inspection",
    ) ||
    result.processedDecisionRowCount > 2 ||
    result.providers.length > 3 ||
    JSON.stringify(result.providers) !==
      JSON.stringify(reversedResult.providers) ||
    completeDecisionProjection(result) !==
      completeDecisionProjection(reversedResult)
  ) {
    throw new Error(
      `provider-limit or dedupe invariance failed: ${JSON.stringify(result)}`,
    );
  }
  if (
    result.providers.find((provider) => provider.providerId === "validonly")
        ?.invalidEvidenceCount !== 3 ||
    !result.blockers.includes(
      "missing prospective multi-window shadow A/B evidence",
    ) ||
    result.shadow_only !== true || result.live_execution !== false ||
    result.promotionReady !== false
  ) {
    throw new Error("stress blockers or shadow semantics failed");
  }
  if (
    result.rowsExceeded !== true ||
    result.decisionTruncated !== true ||
    result.inputEnvelopeTruncated !== true ||
    result.providerDiagnosticsTruncated !== true ||
    result.futureTelemetry.futureEvidenceCount !==
      reversedResult.futureTelemetry.futureEvidenceCount
  ) {
    throw new Error(
      `stress projection/category mismatch: ${
        completeDecisionProjection(result)
      }`,
    );
  }
});

Deno.test("stress malformed future evidence remains invalid and order invariant", () => {
  const malformedFuture = make("malformed-future", "future-valid", {
    completedAt: "2026-09-14T00:00:00Z",
    freshnessAt: "not-a-timestamp",
    failures: [{ id: "bad" }],
  });
  const ordinaryMalformed = make("malformed-now", "validonly", {
    freshnessAt: "2026-09-13T12:00:00+01:00",
  });
  const inputs = [
    [malformedFuture, ordinaryMalformed],
    [ordinaryMalformed, malformedFuture],
  ];
  const results = inputs.map((input) =>
    compileCapabilityGapAlphaCompiler(input, {
      observedAt: "2026-09-13T13:00:00Z",
    })
  );
  if (
    results.some((result) =>
      result.invalidEvidenceCount !== 2 ||
      result.futureTelemetry.futureEvidenceCount !== 0 ||
      result.providers.find((provider) => provider.providerId === "validonly")
          ?.invalidEvidenceCount !== 1 ||
      result.providers.find((provider) =>
          provider.providerId === "future-valid"
        )
          ?.invalidEvidenceCount !== 1
    )
  ) {
    throw new Error(
      `malformed future handling failed: ${JSON.stringify(results)}`,
    );
  }
});

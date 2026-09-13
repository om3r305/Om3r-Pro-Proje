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
  const rows: unknown[] = [
    ...alphaFailureRows,
    ...conflictRows,
    make("malformed", "validonly", { failures: [{ id: "broken" }] }),
    make("future-only", "futureonly", {
      completedAt: "2026-09-14T00:00:00Z",
      freshnessAt: "2026-09-14T00:00:00Z",
    }),
    make("bad-provider", "9bad"),
  ];
  for (let i = 0; i < 40; i++) {
    rows.push(make(`overflow-${i}`, `z-provider-${i}`));
  }
  const result = compileCapabilityGapAlphaCompiler(rows, {
    observedAt: "2026-09-13T13:00:00Z",
    maxRows: 2,
    maxProviders: 3,
    maxInputRows: 100,
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
  const reversedResult = compileCapabilityGapAlphaCompiler(
    [...rows].reverse(),
    {
      observedAt: "2026-09-13T13:00:00Z",
      maxRows: 2,
      maxProviders: 3,
      maxInputRows: 100,
    },
  );
  if (
    rawFailureOccurrences !== 3 ||
    uniqueFailureIdentities !== 2 ||
    rawFailureOccurrences <= uniqueFailureIdentities ||
    alpha?.failedCount !== 2 ||
    alpha?.recentFailures.length !== 2 ||
    JSON.stringify(alpha?.recentFailures) !== JSON.stringify(["a", "b"])
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
    result.invalidEvidenceCount !== 2 ||
    result.futureTelemetry.futureEvidenceCount !== 1 ||
    result.processedDecisionRowCount > 2 ||
    result.providers.length > 3 ||
    JSON.stringify(result.providers) !==
      JSON.stringify(reversedResult.providers)
  ) {
    throw new Error(
      `provider-limit or dedupe invariance failed: ${JSON.stringify(result)}`,
    );
  }
  if (
    result.providers.find((provider) => provider.providerId === "validonly")
        ?.invalidEvidenceCount !== 1 ||
    !result.blockers.includes(
      "missing prospective multi-window shadow A/B evidence",
    ) ||
    result.shadow_only !== true || result.live_execution !== false ||
    result.promotionReady !== false
  ) {
    throw new Error("stress blockers or shadow semantics failed");
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

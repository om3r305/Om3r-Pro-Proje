import { compileCapabilityGapAlphaCompiler } from "../../../supabase/functions/_shared/evolution_candidates/capability-gap-alpha-compiler.ts";

const base = {
  providerId: "alpha",
  rowId: "base",
  completedAt: "2026-09-13T12:00:00Z",
  freshnessAt: "2026-09-13T12:00:00Z",
  health: "HEALTHY",
  status: "COMPLETED",
  failures: [],
};
const future = (rowId: string, completedAt: string) => ({
  ...base,
  rowId,
  providerId: rowId === "future-only" ? "futureonly" : "alpha",
  completedAt,
  freshnessAt: completedAt,
});
const projection = (
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

const futureTelemetry = (
  result: ReturnType<typeof compileCapabilityGapAlphaCompiler>,
) => result.futureTelemetry;

Deno.test("immutable replay keeps every decision field isolated from future permutations", () => {
  const baseline = compileCapabilityGapAlphaCompiler([base], {
    observedAt: "2026-09-13T13:00:00Z",
    maxRows: 1,
  });
  const futures = [
    future("f1", "2026-09-14T00:00:00Z"),
    future("f2", "2026-09-15T00:00:00Z"),
    future("future-only", "2026-09-16T00:00:00Z"),
  ];
  for (
    const permutation of [[...futures], [futures[2], futures[0], futures[1]], [
      futures[1],
      futures[2],
      futures[0],
    ]]
  ) {
    const result = compileCapabilityGapAlphaCompiler([
      permutation[0],
      base,
      permutation[1],
      permutation[2],
    ], { observedAt: "2026-09-13T13:00:00Z", maxRows: 1 });
    if (projection(result) !== projection(baseline)) {
      throw new Error("future evidence changed decision projection");
    }
    if (result.futureTelemetry.futureEvidenceCount !== 3) {
      throw new Error("future telemetry missing");
    }
    if (
      JSON.stringify(futureTelemetry(result)) !== '{"futureEvidenceCount":3}'
    ) {
      throw new Error("future telemetry changed unexpectedly");
    }
    if (
      result.providers.some((provider) => provider.providerId === "futureonly")
    ) {
      throw new Error("future-only provider entered decision state");
    }
  }
});

Deno.test("hard envelope boundary is explicit and observable", () => {
  const inside = [
    base,
    future("inside-future", "2026-09-14T00:00:00Z"),
  ];
  const withinEnvelope = compileCapabilityGapAlphaCompiler(inside, {
    observedAt: "2026-09-13T13:00:00Z",
    maxRows: 1,
    maxInputRows: 2,
  });

  const beyondEnvelope = compileCapabilityGapAlphaCompiler([
    ...inside,
    {
      ...base,
      providerId: "beyond",
      rowId: "beyond-envelope",
      completedAt: "2026-09-13T12:00:02Z",
      freshnessAt: "2026-09-13T12:00:02Z",
      failures: [{ id: "beyond-failure", message: "observed only inside" }],
    },
  ], {
    observedAt: "2026-09-13T13:00:00Z",
    maxRows: 1,
    maxInputRows: 2,
  });
  const movedInside = compileCapabilityGapAlphaCompiler([
    base,
    {
      ...base,
      providerId: "beyond",
      rowId: "beyond-envelope",
      completedAt: "2026-09-13T12:00:02Z",
      freshnessAt: "2026-09-13T12:00:02Z",
      failures: [{ id: "beyond-failure", message: "observed only inside" }],
    },
  ], {
    observedAt: "2026-09-13T13:00:00Z",
    maxRows: 1,
    maxInputRows: 2,
  });
  if (
    withinEnvelope.inputEnvelopeTruncated ||
    !beyondEnvelope.inputEnvelopeTruncated ||
    !beyondEnvelope.blockers.includes("input envelope truncated") ||
    !beyondEnvelope.blockers.includes(
      "input envelope limit prevents complete evidence inspection",
    ) ||
    beyondEnvelope.providers.some((provider) =>
      provider.providerId === "beyond"
    ) ||
    movedInside.providers.every((provider) =>
      provider.providerId !== "beyond"
    ) ||
    movedInside.invalidEvidenceCount !== 0
  ) {
    throw new Error(
      JSON.stringify({ withinEnvelope, beyondEnvelope, movedInside }),
    );
  }
});

Deno.test("future freshness changes telemetry but no decision projection", () => {
  const baseline = compileCapabilityGapAlphaCompiler([base], {
    observedAt: "2026-09-13T13:00:00Z",
    maxRows: 1,
    maxProviders: 1,
    maxInputRows: 4,
  });
  const futureFreshness = compileCapabilityGapAlphaCompiler([
    base,
    {
      ...base,
      rowId: "future-freshness",
      freshnessAt: "2026-09-14T00:00:00Z",
    },
  ], {
    observedAt: "2026-09-13T13:00:00Z",
    maxRows: 1,
    maxProviders: 1,
    maxInputRows: 4,
  });
  if (projection(futureFreshness) !== projection(baseline)) {
    throw new Error("future freshness changed decision projection");
  }
  if (
    JSON.stringify(futureFreshness.futureTelemetry) !==
      '{"futureEvidenceCount":1}' ||
    futureFreshness.providers.length !== baseline.providers.length
  ) throw new Error(JSON.stringify(futureFreshness));
});

Deno.test("explicit null failures remain invalid under replay reversal", () => {
  const rows = [
    {
      ...base,
      rowId: "null-failures",
      failures: null,
    },
    {
      ...base,
      rowId: "valid",
      completedAt: "2026-09-13T12:00:01Z",
    },
  ];
  const forward = compileCapabilityGapAlphaCompiler(rows, {
    observedAt: "2026-09-13T13:00:00Z",
  });
  const reverse = compileCapabilityGapAlphaCompiler([...rows].reverse(), {
    observedAt: "2026-09-13T13:00:00Z",
  });
  if (
    projection(forward) !== projection(reverse) ||
    forward.invalidEvidenceCount !== 1 ||
    forward.providers.find((provider) => provider.providerId === "alpha")
        ?.invalidEvidenceCount !== 1 ||
    forward.futureTelemetry.futureEvidenceCount !== 0 ||
    reverse.futureTelemetry.futureEvidenceCount !== 0
  ) throw new Error(JSON.stringify({ forward, reverse }));
});

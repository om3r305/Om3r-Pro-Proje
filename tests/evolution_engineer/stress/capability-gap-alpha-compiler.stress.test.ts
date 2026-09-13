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

Deno.test("stress bounds oversized evidence and deduplicates named failures", () => {
  const first = { id: "failure-a", message: "a" },
    second = { id: "failure-b", message: "b" };
  const rows: unknown[] = [
    make("a", "alpha", { failures: [first, first] }),
    make("b", "alpha", { failures: [second] }),
    make("conflict", "alpha", { health: "DEGRADED" }),
    make("malformed", "validonly", { failures: [{ id: "broken" }] }),
    make("future", "alpha", { completedAt: "2026-09-14T00:00:00Z" }),
    make("bad-provider", "9bad"),
  ];
  for (let i = 0; i < 40; i++) {
    rows.push(make(`overflow-${i}`, `provider-${i}`));
  }
  const result = compileCapabilityGapAlphaCompiler(rows, {
    observedAt: "2026-09-13T13:00:00Z",
    maxRows: 2,
    maxProviders: 3,
    maxInputRows: 100,
  });
  const rawFailureOccurrences = 3;
  const expectedUniqueFailureIdentities = 2;
  const alpha = result.providers.find((provider) =>
    provider.providerId === "alpha"
  );
  if (
    rawFailureOccurrences <= expectedUniqueFailureIdentities ||
    alpha?.failedCount !== expectedUniqueFailureIdentities ||
    alpha?.recentFailures.length !== expectedUniqueFailureIdentities
  ) {
    throw new Error(`dedupe failed: ${JSON.stringify(result)}`);
  }
  if (
    !result.blockers.includes(
      "ambiguous conflicting equal-timestamp evidence",
    ) || result.invalidProviderCount < 1
  ) throw new Error("stress blockers missing");
  if (
    result.providers.find((provider) => provider.providerId === "validonly")
      ?.invalidEvidenceCount !== 1
  ) throw new Error("invalid-only provider missing");
  if (
    result.processedDecisionRowCount > 2 || result.providers.length > 3 ||
    result.invalidEvidenceCount !== 2 ||
    result.futureTelemetry.futureEvidenceCount !== 1 ||
    !result.blockers.includes(
      "missing prospective multi-window shadow A/B evidence",
    ) || result.shadow_only !== true || result.live_execution !== false ||
    result.promotionReady !== false
  ) throw new Error("hard bound or future isolation failed");
});

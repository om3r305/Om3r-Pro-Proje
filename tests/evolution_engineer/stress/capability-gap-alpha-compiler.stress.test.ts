import { compileCapabilityGapAlpha } from "../../../supabase/functions/_shared/evolution_candidates/capability-gap-alpha-compiler.ts";

Deno.test("adversarial evidence remains bounded and diagnostic-only", () => {
  const evidence = Array.from({ length: 500 }, (_, index) => ({
    evidenceId: `e-${index}`,
    capabilityId: "market.sensor",
    observedAt: "2026-09-13T00:00:00Z",
    leaseStatus: (index % 3 === 0
      ? "LEASE_SKIPPED"
      : index % 3 === 1
      ? "SKIPPED_LEASE"
      : "LEASE_UNAVAILABLE") as const,
    outcome: index % 2 ? "healthy" : "missing",
    ...(index === 0 ? { failures: null as never } : {}),
  }));
  const result = compileCapabilityGapAlpha({
    decisionAt: "2026-09-13T00:00:00Z",
    evidence,
  });
  if (
    result.diagnostics.length > 64 || result.acceptedEvidenceIds.length > 64
  ) throw new Error("unbounded output");
  if (!result.shadow_only || result.live_execution || result.promotionReady) {
    throw new Error("unsafe output");
  }
  const again = compileCapabilityGapAlpha({
    decisionAt: "2026-09-13T00:00:00Z",
    evidence: [...evidence].reverse(),
  });
  if (JSON.stringify(result) !== JSON.stringify(again)) {
    throw new Error("permutation instability");
  }
});

import { compileCapabilityGapAlpha } from "../../../supabase/functions/_shared/evolution_candidates/capability-gap-alpha-compiler.ts";

Deno.test("historical capability-gap replay is deterministic and point-in-time safe", () => {
  const evidence = [
    {
      evidenceId: "r-2",
      capabilityId: "market.sensor",
      observedAt: "2026-09-12T23:59:00Z",
      leaseStatus: "LEASE_UNAVAILABLE" as const,
      outcome: "missing",
    },
    {
      evidenceId: "r-1",
      capabilityId: "market.sensor",
      observedAt: "2026-09-12T23:58:00Z",
      leaseStatus: "LEASE_HELD" as const,
      outcome: "healthy",
    },
    {
      evidenceId: "telemetry",
      capabilityId: "market.sensor",
      observedAt: "2026-09-13T00:01:00Z",
      leaseStatus: "LEASE_HELD" as const,
      outcome: "healthy",
    },
  ];
  const result = compileCapabilityGapAlpha({
    decisionAt: "2026-09-13T00:00:00Z",
    evidence,
  });
  if (result.acceptedEvidenceIds.join(",") !== "r-1,r-2") {
    throw new Error("replay boundary changed");
  }
  if (result.futureEvidenceIds.join(",") !== "telemetry") {
    throw new Error("future telemetry missing");
  }
  const reordered = compileCapabilityGapAlpha({
    decisionAt: "2026-09-13T00:00:00Z",
    evidence: [...evidence].reverse(),
  });
  if (JSON.stringify(result) !== JSON.stringify(reordered)) {
    throw new Error("replay is not deterministic");
  }
});

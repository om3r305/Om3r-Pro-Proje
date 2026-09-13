import { assert, assertEquals } from "jsr:@std/assert@^1.0.0";
import { compileCapabilityGapAlpha } from "../../../supabase/functions/_shared/evolution_candidates/capability-gap-alpha-compiler.ts";

Deno.test("adversarial volume stays bounded, deterministic, and alias-safe", () => {
  const snapshots = Array.from({ length: 1200 }, (_, index) => ({
    capabilityId: `cap.${index % 24}`,
    collectorId: `collector.${index % 8}`,
    observedAt: "2026-09-13T10:00:00.000Z",
    health: index % 5 === 0 ? "MISSING" as const : "DEGRADED" as const,
    leaseOutcome: (index % 3 === 0
      ? "LEASE_SKIPPED"
      : index % 3 === 1
      ? "SKIPPED_LEASE"
      : "LEASE_UNAVAILABLE") as const,
    details: "stress",
  }));
  const input = { decisionAt: "2026-09-13T12:00:00.000Z", snapshots };
  const first = compileCapabilityGapAlpha(input);
  const second = compileCapabilityGapAlpha({
    ...input,
    snapshots: [...snapshots].reverse(),
  });
  assertEquals(first, second);
  assert(first.gaps.length <= 24 * 8);
  assert(first.diagnostics.length <= 128);
  assert(first.gaps.every((gap) => gap.leaseSkipped));
  assertEquals(first.shadowOnly, true);
  assertEquals(first.liveExecution, false);
  assertEquals(first.promotionReady, false);
});

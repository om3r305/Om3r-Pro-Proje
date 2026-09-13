import { assertEquals } from "jsr:@std/assert@^1.0.0";
import {
  type CapabilityGapCompilerInput,
  compileCapabilityGapAlpha,
} from "../../../supabase/functions/_shared/evolution_candidates/capability-gap-alpha-compiler.ts";

const fixture: CapabilityGapCompilerInput = {
  decisionAt: "2026-09-12T00:00:00.000Z",
  snapshots: [
    {
      capabilityId: "news",
      collectorId: "collector-b",
      observedAt: "2026-09-11T23:59:00.000Z",
      health: "STALE",
      leaseOutcome: "ACQUIRED",
    },
    {
      capabilityId: "alpha",
      collectorId: "collector-a",
      observedAt: "2026-09-11T23:59:00.000Z",
      health: "DEGRADED",
      leaseOutcome: "LEASE_UNAVAILABLE",
    },
    {
      capabilityId: "alpha",
      collectorId: "collector-a",
      observedAt: "2026-09-11T23:59:00.000Z",
      health: "DEGRADED",
      leaseOutcome: "LEASE_SKIPPED",
    },
  ],
  failures: [{
    failureId: "f1",
    capabilityId: "news",
    occurredAt: "2026-09-11T23:58:00.000Z",
    code: "TIMEOUT",
  }],
};

Deno.test("immutable replay is invariant to input order and lease spelling", () => {
  const reordered: CapabilityGapCompilerInput = {
    ...fixture,
    snapshots: [...fixture.snapshots].reverse(),
    failures: [...fixture.failures!].reverse(),
  };
  assertEquals(
    compileCapabilityGapAlpha(fixture),
    compileCapabilityGapAlpha(reordered),
  );
});

Deno.test("future replay records cannot contaminate the decision-time result", () => {
  const baseline = compileCapabilityGapAlpha(fixture);
  const contaminated = compileCapabilityGapAlpha({
    ...fixture,
    snapshots: [...fixture.snapshots, {
      capabilityId: "news",
      collectorId: "collector-b",
      observedAt: "2026-09-12T00:00:01.000Z",
      health: "MISSING",
      leaseOutcome: "FAILED",
    }],
  });
  assertEquals(contaminated.gaps, baseline.gaps);
});

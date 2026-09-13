import { assert, assertEquals } from "jsr:@std/assert@^1.0.0";
import {
  type CapabilityGapCompilerInput,
  compileCapabilityGapAlpha,
  MAX_DIAGNOSTICS,
} from "./capability-gap-alpha-compiler.ts";

const decisionAt = "2026-09-13T12:00:00.000Z";
const base = (
  leaseOutcome:
    | "ACQUIRED"
    | "LEASE_SKIPPED"
    | "SKIPPED_LEASE"
    | "LEASE_UNAVAILABLE",
) => ({
  capabilityId: "cap.alpha",
  collectorId: "collector.a",
  observedAt: "2026-09-13T11:59:00.000Z",
  health: "DEGRADED" as const,
  leaseOutcome,
  details: "contention",
});

Deno.test("lease aliases are one canonical fingerprint and do not amplify gaps", () => {
  const aliased = compileCapabilityGapAlpha({
    decisionAt,
    snapshots: [
      base("LEASE_SKIPPED"),
      base("SKIPPED_LEASE"),
      base("LEASE_UNAVAILABLE"),
    ],
  });
  assertEquals(aliased.gaps.length, 1);
  assertEquals(aliased.gaps[0].leaseSkipped, true);
  assertEquals(aliased.gaps[0].evidenceFingerprints.length, 1);
  assertEquals(aliased.gaps[0].reasons, [
    "health:DEGRADED",
    "lease:LEASE_SKIPPED",
  ]);
});

Deno.test("omitted failures are empty evidence but null failures are malformed", () => {
  const omitted = compileCapabilityGapAlpha({
    decisionAt,
    snapshots: [base("ACQUIRED")],
  });
  const explicitNull = compileCapabilityGapAlpha({
    decisionAt,
    snapshots: [base("ACQUIRED")],
    failures: null as unknown as never[],
  });
  assert(!omitted.diagnostics.some((row) => row.code === "MALFORMED_INPUT"));
  assert(
    explicitNull.diagnostics.some((row) =>
      row.message.includes("failures:null")
    ),
  );
});

Deno.test("future evidence is telemetry only and cannot change historical gaps", () => {
  const historical: CapabilityGapCompilerInput = {
    decisionAt,
    snapshots: [base("ACQUIRED")],
  };
  const withFuture: CapabilityGapCompilerInput = {
    ...historical,
    snapshots: [...historical.snapshots, {
      ...base("ACQUIRED"),
      health: "MISSING",
      observedAt: "2026-09-13T12:01:00.000Z",
    }],
    failures: [{
      failureId: "future",
      capabilityId: "cap.alpha",
      occurredAt: "2026-09-13T12:01:00.000Z",
      code: "X",
    }],
  };
  const first = compileCapabilityGapAlpha(historical);
  const second = compileCapabilityGapAlpha(withFuture);
  assertEquals(second.gaps, first.gaps);
  assertEquals(second.telemetry, {
    futureSnapshotCount: 1,
    futureCollectorCount: 0,
    futureFailureCount: 1,
  });
});

Deno.test("malformed metrics and conflicting same-instant evidence fail closed", () => {
  const result = compileCapabilityGapAlpha({
    decisionAt,
    snapshots: [
      base("ACQUIRED"),
      { ...base("ACQUIRED"), health: "HEALTHY" },
      { ...base("ACQUIRED"), score: Number.NaN },
    ],
  });
  assert(result.diagnostics.some((row) => row.code === "MALFORMED_INPUT"));
  assert(result.diagnostics.some((row) => row.code === "CONFLICT"));
  assertEquals(result.gaps[0].severity, "CRITICAL");
});

Deno.test("diagnostics and gaps remain bounded and execution stays shadow-only", () => {
  const snapshots = Array.from({ length: 600 }, (_, index) => ({
    ...base("ACQUIRED"),
    capabilityId: `cap.${index}`,
    observedAt: "not-a-time",
  }));
  const result = compileCapabilityGapAlpha({ decisionAt, snapshots });
  assert(result.diagnostics.length <= MAX_DIAGNOSTICS);
  assertEquals(result.shadowOnly, true);
  assertEquals(result.liveExecution, false);
  assertEquals(result.promotionReady, false);
});

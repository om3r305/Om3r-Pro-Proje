import {
  canonicalLeaseFamily,
  type CapabilityGapEvidence,
  compileCapabilityGapAlpha,
} from "./capability-gap-alpha-compiler.ts";

const at = "2026-09-13T12:00:00.000Z";
const row = (
  patch: Partial<CapabilityGapEvidence> = {},
): CapabilityGapEvidence => ({
  evidenceId: "e-1",
  capabilityId: "portfolio.treasury",
  observedAt: at,
  leaseStatus: "LEASE_SKIPPED",
  outcome: "missing",
  ...patch,
});

Deno.test("lease aliases share one identity and conflict family", () => {
  if (canonicalLeaseFamily("SKIPPED_LEASE") !== "LEASE_UNAVAILABLE") {
    throw new Error("alias");
  }
  const result = compileCapabilityGapAlpha({
    decisionAt: at,
    evidence: [
      row(),
      row({ evidenceId: "e-2", leaseStatus: "LEASE_UNAVAILABLE" }),
    ],
  });
  if (result.conflictKeys.length) throw new Error("aliases became a conflict");
  if (result.acceptedEvidenceIds.length !== 1) {
    throw new Error("duplicate was not deduplicated");
  }
});

Deno.test("explicit null failures fails closed while omission is compatible", () => {
  const result = compileCapabilityGapAlpha({
    decisionAt: at,
    evidence: [
      row({ evidenceId: "null", failures: null as never }),
      row({ evidenceId: "ok" }),
    ],
  });
  if (
    result.status !== "FAILED" ||
    !result.quarantinedEvidenceIds.includes("null")
  ) throw new Error("null accepted");
  if (result.acceptedEvidenceIds.length) {
    throw new Error("failed input produced accepted evidence");
  }
  const omitted = compileCapabilityGapAlpha({
    decisionAt: at,
    evidence: [row()],
  });
  if (
    omitted.status !== "COMPILED" ||
    !omitted.acceptedEvidenceIds.includes("e-1")
  ) {
    throw new Error("omitted failures rejected");
  }
});

Deno.test("future evidence is telemetry only", () => {
  const result = compileCapabilityGapAlpha({
    decisionAt: at,
    evidence: [
      row({ evidenceId: "future", observedAt: "2026-09-14T00:00:00.000Z" }),
    ],
  });
  if (
    !result.futureEvidenceIds.includes("future") ||
    result.acceptedEvidenceIds.length
  ) {
    throw new Error("future evidence influenced decision");
  }
});

Deno.test("conflicts are diagnostic and output is input-order invariant", () => {
  const evidence = [
    row({ evidenceId: "b", outcome: "healthy" }),
    row({ evidenceId: "a", outcome: "missing", leaseStatus: "SKIPPED_LEASE" }),
  ];
  const one = compileCapabilityGapAlpha({ decisionAt: at, evidence });
  const two = compileCapabilityGapAlpha({
    decisionAt: at,
    evidence: [...evidence].reverse(),
  });
  if (JSON.stringify(one) !== JSON.stringify(two)) {
    throw new Error("order changed output");
  }
  if (one.status !== "COMPILED" || one.conflictKeys.length !== 1) {
    throw new Error("conflict not diagnostic");
  }
  if (!one.shadow_only || one.live_execution || one.promotionReady) {
    throw new Error("execution boundary");
  }
});

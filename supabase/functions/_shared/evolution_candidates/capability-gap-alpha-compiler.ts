import {
  EVOLUTION_EVIDENCE_CLASS,
  EVOLUTION_LIVE_EXECUTION,
  EVOLUTION_SHADOW_ONLY,
} from "../evolution_contract.ts";

export const CAPABILITY_GAP_ALPHA_COMPILER_VERSION = "1.0.0";
export const MAX_SNAPSHOTS = 512;
export const MAX_FAILURES_PER_SOURCE = 32;
export const MAX_FAILURES_TOTAL = 256;
export const MAX_DIAGNOSTICS = 128;

export type LeaseOutcome =
  | "ACQUIRED"
  | "LEASE_SKIPPED"
  | "SKIPPED_LEASE"
  | "LEASE_UNAVAILABLE"
  | "FAILED";
export type GapSeverity = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";

export interface CapabilityGapSnapshot {
  capabilityId: string;
  observedAt: string;
  health: "HEALTHY" | "DEGRADED" | "STALE" | "MISSING" | "DISABLED";
  collectorId: string;
  leaseOutcome: LeaseOutcome;
  score?: number;
  details?: string;
}

export interface CapabilityFailure {
  failureId: string;
  capabilityId: string;
  occurredAt: string;
  code: string;
  message?: string;
  metric?: number;
}

export interface CollectorDiagnostic {
  collectorId: string;
  observedAt: string;
  leaseOutcome: LeaseOutcome;
  fingerprint?: string;
  detail?: string;
}

export interface CapabilityGapCompilerInput {
  decisionAt: string;
  snapshots: CapabilityGapSnapshot[];
  collectors?: CollectorDiagnostic[];
  failures?: CapabilityFailure[];
}

export interface CapabilityGap {
  capabilityId: string;
  severity: GapSeverity;
  reasons: string[];
  evidenceFingerprints: string[];
  leaseSkipped: boolean;
  failureCount: number;
}

export interface CompilerDiagnostic {
  code:
    | "MALFORMED_INPUT"
    | "FUTURE_EVIDENCE_TELEMETRY_ONLY"
    | "TRUNCATED_INPUT"
    | "CONFLICT";
  message: string;
  identity?: string;
}

export interface CapabilityGapCompilerResult {
  compilerVersion: string;
  decisionAt: string;
  evidenceClass: typeof EVOLUTION_EVIDENCE_CLASS;
  shadowOnly: typeof EVOLUTION_SHADOW_ONLY;
  liveExecution: typeof EVOLUTION_LIVE_EXECUTION;
  promotionReady: false;
  gaps: CapabilityGap[];
  diagnostics: CompilerDiagnostic[];
  telemetry: {
    futureSnapshotCount: number;
    futureCollectorCount: number;
    futureFailureCount: number;
  };
}

const LEASE_SKIP = "LEASE_SKIPPED";
const VALID_HEALTH = new Set([
  "HEALTHY",
  "DEGRADED",
  "STALE",
  "MISSING",
  "DISABLED",
]);
const VALID_LEASE = new Set([
  "ACQUIRED",
  "LEASE_SKIPPED",
  "SKIPPED_LEASE",
  "LEASE_UNAVAILABLE",
  "FAILED",
]);

function text(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function iso(value: unknown): number | null {
  const valueText = text(value);
  if (!valueText) return null;
  const parsed = Date.parse(valueText);
  return Number.isFinite(parsed) ? parsed : null;
}

function boundedDiagnostics(rows: CompilerDiagnostic[]): CompilerDiagnostic[] {
  return rows.slice(0, MAX_DIAGNOSTICS);
}

function normalizedLease(value: unknown): LeaseOutcome | null {
  const candidate = text(value).toUpperCase();
  if (!VALID_LEASE.has(candidate)) return null;
  return candidate === "SKIPPED_LEASE" || candidate === "LEASE_UNAVAILABLE"
    ? LEASE_SKIP
    : candidate as LeaseOutcome;
}

function keyPart(value: string): string {
  return value.replaceAll("|", "%7C");
}

function snapshotFingerprint(
  snapshot: CapabilityGapSnapshot,
  lease: LeaseOutcome,
): string {
  return [
    keyPart(snapshot.capabilityId),
    keyPart(snapshot.collectorId),
    snapshot.observedAt,
    snapshot.health,
    lease,
    Number.isFinite(snapshot.score) ? String(snapshot.score) : "",
    keyPart(text(snapshot.details)),
  ].join("|");
}

function identity(snapshot: CapabilityGapSnapshot): string {
  return `${snapshot.capabilityId}|${snapshot.collectorId}|${snapshot.observedAt}`;
}

function severityFor(
  health: CapabilityGapSnapshot["health"],
  leaseSkipped: boolean,
  failures: number,
): GapSeverity {
  if (health === "MISSING" || health === "DISABLED") return "CRITICAL";
  if (health === "STALE" || failures > 0) return "HIGH";
  if (health === "DEGRADED" || leaseSkipped) return "MEDIUM";
  return "LOW";
}

function validSnapshot(value: unknown): value is CapabilityGapSnapshot {
  if (!value || typeof value !== "object") return false;
  const row = value as Partial<CapabilityGapSnapshot>;
  return Boolean(
    text(row.capabilityId) && text(row.collectorId) &&
      iso(row.observedAt) !== null &&
      VALID_HEALTH.has(row.health ?? "") &&
      normalizedLease(row.leaseOutcome) !== null &&
      (row.score === undefined ||
        (typeof row.score === "number" && Number.isFinite(row.score))),
  );
}

function validCollector(value: unknown): value is CollectorDiagnostic {
  if (!value || typeof value !== "object") return false;
  const row = value as Partial<CollectorDiagnostic>;
  return Boolean(
    text(row.collectorId) && iso(row.observedAt) !== null &&
      normalizedLease(row.leaseOutcome) !== null,
  );
}

function validFailure(value: unknown): value is CapabilityFailure {
  if (!value || typeof value !== "object") return false;
  const row = value as Partial<CapabilityFailure>;
  return Boolean(
    text(row.failureId) && text(row.capabilityId) &&
      iso(row.occurredAt) !== null && text(row.code) &&
      (row.metric === undefined ||
        (typeof row.metric === "number" && Number.isFinite(row.metric))),
  );
}

export function compileCapabilityGapAlpha(
  input: CapabilityGapCompilerInput,
): CapabilityGapCompilerResult {
  const diagnostics: CompilerDiagnostic[] = [];
  const decisionMs = iso(input?.decisionAt);
  const decisionAt = text(input?.decisionAt);
  const future = {
    futureSnapshotCount: 0,
    futureCollectorCount: 0,
    futureFailureCount: 0,
  };
  if (decisionMs === null) {
    diagnostics.push({
      code: "MALFORMED_INPUT",
      message: "decisionAt must be an ISO timestamp",
    });
  }
  if (!Array.isArray(input?.snapshots)) {
    diagnostics.push({
      code: "MALFORMED_INPUT",
      message: "snapshots must be an array",
    });
  }
  if (input?.failures === null) {
    diagnostics.push({
      code: "MALFORMED_INPUT",
      message: "failures:null is malformed; omit failures for empty evidence",
    });
  } else if (input?.failures !== undefined && !Array.isArray(input.failures)) {
    diagnostics.push({
      code: "MALFORMED_INPUT",
      message: "failures must be an array when provided",
    });
  }

  const snapshots = Array.isArray(input?.snapshots)
    ? input.snapshots.slice(0, MAX_SNAPSHOTS)
    : [];
  if (
    Array.isArray(input?.snapshots) && input.snapshots.length > MAX_SNAPSHOTS
  ) {
    diagnostics.push({
      code: "TRUNCATED_INPUT",
      message: `snapshots limited to ${MAX_SNAPSHOTS}`,
    });
  }
  const usableSnapshots: CapabilityGapSnapshot[] = [];
  for (const row of snapshots) {
    if (!validSnapshot(row)) {
      diagnostics.push({
        code: "MALFORMED_INPUT",
        message: "invalid capability snapshot",
      });
      continue;
    }
    if (decisionMs !== null && iso(row.observedAt)! > decisionMs) {
      future.futureSnapshotCount++;
      diagnostics.push({
        code: "FUTURE_EVIDENCE_TELEMETRY_ONLY",
        message: "future snapshot excluded from decision evidence",
        identity: identity(row),
      });
      continue;
    }
    usableSnapshots.push({
      ...row,
      leaseOutcome: normalizedLease(row.leaseOutcome)!,
    });
  }

  const collectors = Array.isArray(input?.collectors)
    ? input.collectors.slice(0, MAX_SNAPSHOTS)
    : [];
  if (
    Array.isArray(input?.collectors) && input.collectors.length > MAX_SNAPSHOTS
  ) {
    diagnostics.push({
      code: "TRUNCATED_INPUT",
      message: `collectors limited to ${MAX_SNAPSHOTS}`,
    });
  }
  for (const row of collectors) {
    if (!validCollector(row)) {
      diagnostics.push({
        code: "MALFORMED_INPUT",
        message: "invalid collector diagnostic",
      });
    } else if (decisionMs !== null && iso(row.observedAt)! > decisionMs) {
      future.futureCollectorCount++;
      diagnostics.push({
        code: "FUTURE_EVIDENCE_TELEMETRY_ONLY",
        message: "future collector diagnostic excluded from decision evidence",
        identity: row.collectorId,
      });
    }
  }

  const failuresByCapability = new Map<string, CapabilityFailure[]>();
  const failures = Array.isArray(input?.failures)
    ? input.failures.slice(0, MAX_FAILURES_TOTAL)
    : [];
  if (
    Array.isArray(input?.failures) && input.failures.length > MAX_FAILURES_TOTAL
  ) {
    diagnostics.push({
      code: "TRUNCATED_INPUT",
      message: `failures limited to ${MAX_FAILURES_TOTAL}`,
    });
  }
  for (const row of failures) {
    if (!validFailure(row)) {
      diagnostics.push({
        code: "MALFORMED_INPUT",
        message: "invalid failure evidence",
      });
      continue;
    }
    if (decisionMs !== null && iso(row.occurredAt)! > decisionMs) {
      future.futureFailureCount++;
      diagnostics.push({
        code: "FUTURE_EVIDENCE_TELEMETRY_ONLY",
        message: "future failure excluded from decision evidence",
        identity: row.failureId,
      });
      continue;
    }
    const existing = failuresByCapability.get(row.capabilityId) ?? [];
    if (existing.length < MAX_FAILURES_PER_SOURCE) existing.push(row);
    failuresByCapability.set(row.capabilityId, existing);
  }

  const grouped = new Map<string, CapabilityGapSnapshot[]>();
  for (
    const row of usableSnapshots.sort((a, b) =>
      snapshotFingerprint(a, a.leaseOutcome).localeCompare(
        snapshotFingerprint(b, b.leaseOutcome),
      )
    )
  ) {
    const rows = grouped.get(identity(row)) ?? [];
    if (
      !rows.some((candidate) =>
        snapshotFingerprint(candidate, candidate.leaseOutcome) ===
          snapshotFingerprint(row, row.leaseOutcome)
      )
    ) rows.push(row);
    grouped.set(identity(row), rows);
  }
  const gaps: CapabilityGap[] = [];
  for (const rows of grouped.values()) {
    const first = rows[0];
    const fingerprints = rows.map((row) =>
      snapshotFingerprint(row, row.leaseOutcome)
    ).sort();
    const aliases = rows.filter((row) => row.leaseOutcome === LEASE_SKIP);
    const conflict = rows.some((row) =>
      row.health !== first.health || row.leaseOutcome !== first.leaseOutcome
    );
    if (conflict) {
      diagnostics.push({
        code: "CONFLICT",
        message:
          "same-instant evidence has conflicting normalized observations",
        identity: identity(first),
      });
    }
    const failureCount = failuresByCapability.get(first.capabilityId)?.length ??
      0;
    const reasons = new Set<string>();
    if (first.health !== "HEALTHY") reasons.add(`health:${first.health}`);
    if (aliases.length) reasons.add("lease:LEASE_SKIPPED");
    if (failureCount) reasons.add("failures:present");
    if (
      first.health === "HEALTHY" && !aliases.length && !failureCount &&
      !conflict
    ) continue;
    gaps.push({
      capabilityId: first.capabilityId,
      severity: severityFor(
        conflict ? "MISSING" : first.health,
        aliases.length > 0,
        failureCount,
      ),
      reasons: [...reasons].sort(),
      evidenceFingerprints: fingerprints,
      leaseSkipped: aliases.length > 0,
      failureCount,
    });
  }
  for (const [capabilityId, failuresForCapability] of failuresByCapability) {
    if (groupedHasCapability(grouped, capabilityId)) continue;
    gaps.push({
      capabilityId,
      severity: "HIGH",
      reasons: ["failures:present"],
      evidenceFingerprints: [],
      leaseSkipped: false,
      failureCount: failuresForCapability.length,
    });
  }
  gaps.sort((a, b) =>
    `${a.capabilityId}|${a.severity}`.localeCompare(
      `${b.capabilityId}|${b.severity}`,
    )
  );
  diagnostics.sort((a, b) =>
    `${a.code}|${a.identity ?? ""}|${a.message}`.localeCompare(
      `${b.code}|${b.identity ?? ""}|${b.message}`,
    )
  );
  return {
    compilerVersion: CAPABILITY_GAP_ALPHA_COMPILER_VERSION,
    decisionAt,
    evidenceClass: EVOLUTION_EVIDENCE_CLASS,
    shadowOnly: EVOLUTION_SHADOW_ONLY,
    liveExecution: EVOLUTION_LIVE_EXECUTION,
    promotionReady: false,
    gaps,
    diagnostics: boundedDiagnostics(diagnostics),
    telemetry: future,
  };
}

function groupedHasCapability(
  grouped: Map<string, CapabilityGapSnapshot[]>,
  capabilityId: string,
): boolean {
  for (const rows of grouped.values()) {
    if (rows[0]?.capabilityId === capabilityId) return true;
  }
  return false;
}

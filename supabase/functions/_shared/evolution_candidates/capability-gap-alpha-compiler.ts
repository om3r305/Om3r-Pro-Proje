// Pure, diagnostic-only compiler for capability-gap evidence.

export const CAPABILITY_GAP_ALPHA_COMPILER_VERSION =
  "brian.capability-gap-alpha.v1";
const MAX_ITEMS = 64;
const MAX_TEXT = 240;

export type LeaseStatus =
  | "LEASE_SKIPPED"
  | "SKIPPED_LEASE"
  | "LEASE_UNAVAILABLE"
  | "LEASE_HELD"
  | "LEASE_EXPIRED"
  | "NO_LEASE";

export interface CapabilityGapFailure {
  code: string;
  message?: string;
}

export interface CapabilityGapEvidence {
  evidenceId: string;
  capabilityId: string;
  observedAt: string;
  leaseStatus: LeaseStatus;
  sourceId?: string;
  outcome?: string;
  failures?: CapabilityGapFailure[];
}

export interface CapabilityGapCompileInput {
  decisionAt: string;
  evidence: CapabilityGapEvidence[];
}

export interface CapabilityGapDiagnostic {
  code:
    | "INVALID_INPUT"
    | "INVALID_RECORD"
    | "FUTURE_EVIDENCE"
    | "DUPLICATE_EVIDENCE"
    | "CONFLICTING_EVIDENCE";
  message: string;
  evidenceIds: string[];
}

export interface CapabilityGapCompileResult {
  compilerVersion: string;
  status: "COMPILED" | "FAILED";
  decisionAt: string;
  diagnostics: CapabilityGapDiagnostic[];
  acceptedEvidenceIds: string[];
  quarantinedEvidenceIds: string[];
  futureEvidenceIds: string[];
  conflictKeys: string[];
  telemetry: {
    futureEvidenceCount: number;
    invalidEvidenceCount: number;
  };
  shadow_only: true;
  live_execution: false;
  promotionReady: false;
}

export function canonicalLeaseFamily(status: string): string {
  if (
    status === "LEASE_SKIPPED" || status === "SKIPPED_LEASE" ||
    status === "LEASE_UNAVAILABLE"
  ) return "LEASE_UNAVAILABLE";
  return status;
}

export const normalizeLeaseStatus = canonicalLeaseFamily;

function bounded(value: string, limit = MAX_TEXT): string {
  return value.length <= limit ? value : `${value.slice(0, limit - 3)}...`;
}

function validText(value: unknown): value is string {
  return typeof value === "string" && value.length > 0 &&
    value.length <= MAX_TEXT;
}

function validLease(value: unknown): value is LeaseStatus {
  return typeof value === "string" && [
    "LEASE_SKIPPED",
    "SKIPPED_LEASE",
    "LEASE_UNAVAILABLE",
    "LEASE_HELD",
    "LEASE_EXPIRED",
    "NO_LEASE",
  ].includes(value);
}

function stableFailureValue(failures: CapabilityGapFailure[]): string {
  return failures.map((failure) => `${failure.code}:${failure.message ?? ""}`)
    .sort().join("|");
}

function recordKey(row: CapabilityGapEvidence): string {
  return [
    row.capabilityId,
    new Date(row.observedAt).toISOString(),
    canonicalLeaseFamily(row.leaseStatus),
  ].join("|");
}

function outcomeKey(row: CapabilityGapEvidence): string {
  return `${row.outcome ?? ""}|${stableFailureValue(row.failures ?? [])}`;
}

function diagnostic(
  code: CapabilityGapDiagnostic["code"],
  message: string,
  ids: string[],
): CapabilityGapDiagnostic {
  return {
    code,
    message: bounded(message),
    evidenceIds: ids.map((id) => bounded(id)).sort().slice(0, MAX_ITEMS),
  };
}

function baseResult(decisionAt: string): CapabilityGapCompileResult {
  return {
    compilerVersion: CAPABILITY_GAP_ALPHA_COMPILER_VERSION,
    status: "COMPILED",
    decisionAt,
    diagnostics: [],
    acceptedEvidenceIds: [],
    quarantinedEvidenceIds: [],
    futureEvidenceIds: [],
    conflictKeys: [],
    telemetry: { futureEvidenceCount: 0, invalidEvidenceCount: 0 },
    shadow_only: true,
    live_execution: false,
    promotionReady: false,
  };
}

/**
 * Compiles only historical, structurally valid evidence. The result intentionally has no
 * action, recommendation, policy, or execution field.
 */
export function compileCapabilityGapAlpha(
  input: CapabilityGapCompileInput,
): CapabilityGapCompileResult {
  const decisionAt = typeof input?.decisionAt === "string"
    ? input.decisionAt
    : "";
  const result = baseResult(decisionAt);
  const decisionMs = Date.parse(decisionAt);
  if (!Number.isFinite(decisionMs) || !Array.isArray(input?.evidence)) {
    result.status = "FAILED";
    result.diagnostics.push(diagnostic(
      "INVALID_INPUT",
      "decisionAt must be a valid timestamp and evidence must be an array",
      [],
    ));
    return result;
  }

  const sorted = [...input.evidence].sort((a, b) =>
    JSON.stringify(a).localeCompare(JSON.stringify(b))
  );
  const valid: CapabilityGapEvidence[] = [];
  for (const row of sorted) {
    let validRow = !!row && typeof row === "object" &&
      validText(row.evidenceId) && validText(row.capabilityId) &&
      typeof row.observedAt === "string" &&
      Number.isFinite(Date.parse(row.observedAt)) &&
      validLease(row.leaseStatus);
    const hasFailures = !!row &&
      Object.prototype.hasOwnProperty.call(row, "failures");
    if (validRow && hasFailures && row.failures === null) validRow = false;
    if (validRow && hasFailures && !Array.isArray(row.failures)) {
      validRow = false;
    }
    if (validRow && Array.isArray(row.failures)) {
      validRow = row.failures.length <= MAX_ITEMS &&
        row.failures.every((failure) =>
          !!failure && validText(failure.code) &&
          (failure.message === undefined || validText(failure.message))
        );
    }
    if (validRow && row.sourceId !== undefined && !validText(row.sourceId)) {
      validRow = false;
    }
    if (validRow && row.outcome !== undefined && !validText(row.outcome)) {
      validRow = false;
    }
    if (!validRow) {
      result.status = "FAILED";
      result.telemetry.invalidEvidenceCount++;
      result.quarantinedEvidenceIds.push(
        typeof row?.evidenceId === "string"
          ? bounded(row.evidenceId)
          : "<invalid>",
      );
      result.diagnostics.push(diagnostic(
        "INVALID_RECORD",
        "evidence is malformed; failures:null is not compatible with omitted failures",
        [typeof row?.evidenceId === "string" ? row.evidenceId : "<invalid>"],
      ));
      continue;
    }
    valid.push(row);
  }

  const historical = valid.filter((row) =>
    Date.parse(row.observedAt) <= decisionMs
  );
  for (const row of valid) {
    if (Date.parse(row.observedAt) > decisionMs) {
      result.futureEvidenceIds.push(row.evidenceId);
      result.telemetry.futureEvidenceCount++;
      result.diagnostics.push(diagnostic(
        "FUTURE_EVIDENCE",
        "future evidence retained as telemetry and excluded from the historical decision",
        [row.evidenceId],
      ));
    }
  }

  const byKey = new Map<string, CapabilityGapEvidence[]>();
  for (const row of historical) {
    const key = recordKey(row);
    const group = byKey.get(key) ?? [];
    group.push(row);
    byKey.set(key, group);
  }
  for (
    const [key, group] of [...byKey.entries()].sort(([a], [b]) =>
      a.localeCompare(b)
    )
  ) {
    const unique = new Map<string, CapabilityGapEvidence>();
    for (const row of group) unique.set(outcomeKey(row), row);
    const ids = group.map((row) => row.evidenceId).sort();
    const first = [...unique.values()].sort((a, b) =>
      a.evidenceId.localeCompare(b.evidenceId)
    )[0];
    if (unique.size > 1) {
      result.conflictKeys.push(key);
      result.diagnostics.push(diagnostic(
        "CONFLICTING_EVIDENCE",
        `same-instant evidence has ${unique.size} distinct outcomes`,
        ids,
      ));
    } else if (group.length > 1) {
      result.diagnostics.push(
        diagnostic(
          "DUPLICATE_EVIDENCE",
          "equivalent evidence was deduplicated",
          ids,
        ),
      );
    }
    if (first) {
      result.acceptedEvidenceIds.push(first.evidenceId);
    }
  }
  result.acceptedEvidenceIds.sort();
  result.quarantinedEvidenceIds.sort();
  result.futureEvidenceIds.sort();
  result.conflictKeys.sort();
  result.diagnostics.sort((a, b) =>
    `${a.code}|${a.evidenceIds.join(",")}`.localeCompare(
      `${b.code}|${b.evidenceIds.join(",")}`,
    )
  );
  result.diagnostics = result.diagnostics.slice(0, MAX_ITEMS);
  result.acceptedEvidenceIds = result.acceptedEvidenceIds.slice(0, MAX_ITEMS);
  result.quarantinedEvidenceIds = result.quarantinedEvidenceIds.slice(
    0,
    MAX_ITEMS,
  );
  result.futureEvidenceIds = result.futureEvidenceIds.slice(0, MAX_ITEMS);
  result.conflictKeys = result.conflictKeys.slice(0, MAX_ITEMS);
  if (result.status === "FAILED") {
    result.acceptedEvidenceIds = [];
    result.conflictKeys = [];
  }
  return result;
}

export const compileCapabilityGap = compileCapabilityGapAlpha;

export type ProviderHealth = "HEALTHY" | "DEGRADED" | "UNHEALTHY" | "UNKNOWN";
export type ReportClassification =
  | "HEALTHY"
  | "BLOCKED"
  | "INSUFFICIENT_EVIDENCE";

export interface CompilerOptions {
  observedAt: number | string;
  maxRows?: number;
  maxProviders?: number;
  maxInputRows?: number;
}

export interface ProviderDiagnostic {
  providerId: string;
  classification: ProviderHealth;
  health: ProviderHealth;
  freshnessAt: string | null;
  stale: boolean;
  failedCount: number;
  leaseSkippedCount: number;
  recentFailures: string[];
  invalidEvidenceCount: number;
}

export interface CompilerReport {
  classification: ReportClassification;
  providers: ProviderDiagnostic[];
  processedDecisionRowCount: number;
  rowsExceeded: boolean;
  decisionTruncated: boolean;
  invalidEvidenceCount: number;
  invalidProviderCount: number;
  blockers: string[];
  futureTelemetry: { futureEvidenceCount: number };
  inputEnvelopeTruncated: boolean;
  providerDiagnosticsTruncated: boolean;
  shadow_only: true;
  live_execution: false;
  promotionReady: false;
}

interface ValidRow {
  providerId: string;
  rowId: string;
  completedAt: string;
  completionMs: number;
  freshnessAt: string | null;
  freshnessMs: number | null;
  health: ProviderHealth;
  status: CollectorStatus;
  failures: Array<{ id: string; message: string }>;
}

type CollectorStatus =
  | "COMPLETED"
  | "LEASE_SKIPPED"
  | "SKIPPED_LEASE"
  | "LEASE_UNAVAILABLE";

const DEFAULT_MAX_ROWS = 100;
const DEFAULT_MAX_PROVIDERS = 20;
const DEFAULT_MAX_INPUT_ROWS = 10_000;
const ISO_UTC =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?Z$/;
const LEASE_STATUSES = new Set([
  "LEASE_SKIPPED",
  "SKIPPED_LEASE",
  "LEASE_UNAVAILABLE",
]);
const SUPPORTED_STATUSES = new Set<string>([
  "COMPLETED",
  "LEASE_SKIPPED",
  "SKIPPED_LEASE",
  "LEASE_UNAVAILABLE",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function health(value: unknown): ProviderHealth | null {
  if (
    value === "HEALTHY" || value === "DEGRADED" || value === "UNHEALTHY" ||
    value === "UNKNOWN"
  ) return value;
  return null;
}

function positiveInt(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) &&
      Number.isInteger(value) && value > 0
    ? value
    : fallback;
}

function strictUtc(value: unknown): { text: string; ms: number } | null {
  if (typeof value !== "string") return null;
  const match = ISO_UTC.exec(value);
  if (!match) return null;
  const ms = Date.parse(value);
  if (!Number.isFinite(ms)) return null;
  const fraction = (match[7] ?? "").padEnd(3, "0");
  const normalized = `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${
    match[5]
  }:${match[6]}.${fraction}Z`;
  if (new Date(ms).toISOString() !== normalized) return null;
  return { text: value, ms };
}

function providerId(value: unknown): string | null {
  return typeof value === "string" && /^[a-z][a-z0-9_-]{1,63}$/i.test(value)
    ? value
    : null;
}

function observedMs(value: number | string): number | null {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  const parsed = strictUtc(value);
  return parsed?.ms ?? null;
}

function collectorStatus(value: unknown): CollectorStatus | null {
  if (typeof value !== "string" || !SUPPORTED_STATUSES.has(value)) {
    return null;
  }
  switch (value) {
    case "COMPLETED":
    case "LEASE_SKIPPED":
    case "SKIPPED_LEASE":
    case "LEASE_UNAVAILABLE":
      return value;
    default:
      return null;
  }
}

function stableRow(
  value: unknown,
  observation: number,
): {
  row: ValidRow | null;
  providerId: string | null;
  future: boolean;
  invalid: boolean;
} {
  if (!isRecord(value)) {
    return { row: null, providerId: null, future: false, invalid: true };
  }
  const raw = value;
  const parsedProvider = providerId(raw.providerId);
  const completion = strictUtc(raw.completedAt);
  const freshness = raw.freshnessAt == null ? null : strictUtc(raw.freshnessAt);
  const rowId = typeof raw.rowId === "string" && raw.rowId.trim()
    ? raw.rowId.trim()
    : null;
  const rowHealth = health(raw.health);
  const status = collectorStatus(raw.status);
  const rawFailures = raw.failures == null ? [] : raw.failures;
  const failures: Array<{ id: string; message: string }> = [];
  let nestedInvalid = false;
  if (!Array.isArray(rawFailures)) nestedInvalid = true;
  else {
    for (const failure of rawFailures) {
      if (!isRecord(failure)) {
        nestedInvalid = true;
        continue;
      }
      const item = failure;
      const id = typeof item.id === "string" && item.id.trim()
        ? item.id.trim()
        : null;
      const message = typeof item.message === "string" && item.message.trim()
        ? item.message.trim()
        : null;
      if (!id || !message) nestedInvalid = true;
      else failures.push({ id, message });
    }
  }
  const future = (completion != null && completion.ms > observation) ||
    (freshness != null && freshness.ms > observation);
  if (
    !parsedProvider || !completion || !rowId || !rowHealth || !status ||
    (raw.freshnessAt != null && !freshness) || nestedInvalid
  ) {
    return { row: null, providerId: parsedProvider, future, invalid: true };
  }
  return {
    providerId: parsedProvider,
    future,
    invalid: false,
    row: {
      providerId: parsedProvider,
      rowId,
      completedAt: completion.text,
      completionMs: completion.ms,
      freshnessAt: freshness?.text ?? null,
      freshnessMs: freshness?.ms ?? null,
      health: rowHealth,
      status,
      failures,
    },
  };
}

function baseReport(): CompilerReport {
  return {
    classification: "BLOCKED",
    providers: [],
    processedDecisionRowCount: 0,
    rowsExceeded: false,
    decisionTruncated: false,
    invalidEvidenceCount: 0,
    invalidProviderCount: 0,
    blockers: [
      "missing prospective multi-window shadow A/B evidence",
    ],
    futureTelemetry: { futureEvidenceCount: 0 },
    inputEnvelopeTruncated: false,
    providerDiagnosticsTruncated: false,
    shadow_only: true,
    live_execution: false,
    promotionReady: false,
  };
}

export function compileCapabilityGapAlphaCompiler(
  input: unknown,
  options: CompilerOptions,
): CompilerReport {
  const report = baseReport();
  const maxRows = Math.min(
    DEFAULT_MAX_INPUT_ROWS,
    positiveInt(options?.maxRows, DEFAULT_MAX_ROWS),
  );
  const maxProviders = Math.min(
    1_000,
    positiveInt(options?.maxProviders, DEFAULT_MAX_PROVIDERS),
  );
  const maxInputRows = Math.min(
    DEFAULT_MAX_INPUT_ROWS,
    Math.max(
      maxRows,
      positiveInt(options?.maxInputRows, DEFAULT_MAX_INPUT_ROWS),
    ),
  );
  const observed = observedMs(options?.observedAt);
  if (observed == null) {
    report.blockers.push("invalid observation timestamp");
    report.blockers.push("invalid input evidence");
    return report;
  }
  if (!Array.isArray(input)) {
    report.blockers.push("input must be an array");
    report.blockers.push("invalid input evidence");
    return report;
  }
  const envelope = input.slice(0, maxInputRows);
  report.inputEnvelopeTruncated = input.length > envelope.length;
  if (report.inputEnvelopeTruncated) {
    report.blockers.push("input envelope truncated");
  }
  const recognized = new Set<string>();
  const invalidByProvider = new Map<string, number>();
  const valid: ValidRow[] = [];
  let invalid = 0;
  let invalidProviders = 0;
  let future = 0;
  for (const value of envelope) {
    const parsed = stableRow(value, observed);
    if (parsed.future && !parsed.invalid) {
      future++;
      continue;
    }
    if (parsed.invalid) {
      invalid++;
      if (parsed.providerId) {
        recognized.add(parsed.providerId);
        invalidByProvider.set(
          parsed.providerId,
          (invalidByProvider.get(parsed.providerId) ?? 0) + 1,
        );
      }
    } else if (parsed.row) {
      recognized.add(parsed.row.providerId);
      valid.push(parsed.row);
    } else invalid++;
  }
  report.futureTelemetry.futureEvidenceCount = future;
  report.invalidEvidenceCount = invalid;
  for (const value of envelope) {
    if (isRecord(value) && providerId(value.providerId) == null) {
      invalidProviders++;
    }
  }
  report.invalidProviderCount = invalidProviders;

  const conflicts = new Set<string>();
  const seen = new Map<string, string>();
  for (const row of valid) {
    const key = `${row.providerId}|${row.completedAt}`;
    const identity = JSON.stringify({
      rowId: row.rowId,
      health: row.health,
      status: row.status,
      failures: [...row.failures].sort((a, b) =>
        a.id.localeCompare(b.id) || a.message.localeCompare(b.message)
      ),
    });
    const prior = seen.get(key);
    if (prior && prior !== identity) conflicts.add(key);
    else seen.set(key, identity);
  }
  if (conflicts.size) {
    report.blockers.push("ambiguous conflicting equal-timestamp evidence");
  }
  if (invalid) report.blockers.push("invalid evidence present");
  if (report.inputEnvelopeTruncated) {
    report.blockers.push(
      "input envelope limit prevents complete evidence inspection",
    );
  }
  valid.sort((a, b) =>
    b.completionMs - a.completionMs ||
    a.providerId.localeCompare(b.providerId) || a.rowId.localeCompare(b.rowId)
  );
  const selected = valid.slice(0, maxRows);
  report.processedDecisionRowCount = selected.length;
  report.rowsExceeded = valid.length > maxRows;
  report.decisionTruncated = report.rowsExceeded;

  const providerIds = new Set<string>(selected.map((row) => row.providerId));
  for (const id of [...recognized].sort()) {
    if (invalidByProvider.has(id) && providerIds.size < maxProviders) {
      providerIds.add(id);
    }
  }
  const boundedProviders = [...providerIds].sort().slice(0, maxProviders);
  report.providerDiagnosticsTruncated = providerIds.size > maxProviders ||
    [...recognized].length > maxProviders;
  report.providers = boundedProviders.map((id): ProviderDiagnostic => {
    const rows = selected.filter((row) => row.providerId === id);
    const failures = new Map<string, string>();
    let leaseSkippedCount = 0;
    for (const row of rows) {
      if (LEASE_STATUSES.has(row.status)) leaseSkippedCount++;
      for (const failure of row.failures) {
        failures.set(
          JSON.stringify({
            providerId: row.providerId,
            rowId: row.rowId,
            id: failure.id,
            message: failure.message,
          }),
          failure.message,
        );
      }
    }
    const newest = rows[0];
    const stale = newest?.freshnessMs != null
      ? observed - newest.freshnessMs > 86_400_000
      : true;
    const health = newest?.health ?? "UNKNOWN";
    const providerInvalid = invalidByProvider.get(id) ?? 0;
    return {
      providerId: id,
      classification: providerInvalid || !rows.length ? "UNKNOWN" : health,
      health,
      freshnessAt: newest?.freshnessAt ?? null,
      stale,
      failedCount: failures.size,
      leaseSkippedCount,
      recentFailures: [...failures.entries()].sort(([a], [b]) =>
        a.localeCompare(b)
      ).map(([, message]) => message),
      invalidEvidenceCount: providerInvalid,
    };
  });
  if (report.providers.length === 0) {
    report.blockers.push("no valid provider evidence");
  }
  report.classification = report.blockers.length
    ? "BLOCKED"
    : report.providers.length
    ? "HEALTHY"
    : "INSUFFICIENT_EVIDENCE";
  return report;
}

export const compileProviderHealthDiagnostics =
  compileCapabilityGapAlphaCompiler;

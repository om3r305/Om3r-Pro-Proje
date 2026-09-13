export type ExpectedEdgeRecommendation =
  | "ALLOW_EDGE"
  | "DOWNGRADE_TO_WAIT"
  | "COST_UNAVAILABLE"
  | "INSUFFICIENT_LAGGED_EVIDENCE"
  | "CONTAMINATED_EVIDENCE";

export interface ExpectedEdgeOptions {
  decisionAt: string | number;
  minimumNetMarginBps?: number;
  minimumMatureGroups?: number;
  maxInputRows?: number;
  maxContributions?: number;
}

export interface ExpectedEdgeContribution {
  observationId: string;
  groupId: string;
  expectedMoveBps: number;
  reliability: number;
  contributionBps: number;
}

export interface ExpectedEdgeReport {
  recommendation: ExpectedEdgeRecommendation;
  eligible: boolean;
  expectedGrossMoveBps: number;
  estimatedRoundTripCostBps: number | null;
  uncertaintyPenaltyBps: number;
  eventDecayPenaltyBps: number;
  expectedNetEdgeBps: number | null;
  matureIndependentGroupCount: number;
  contributions: ExpectedEdgeContribution[];
  reasons: string[];
  invalidEvidenceCount: number;
  truncated: boolean;
  futureTelemetry: { futureEvidenceCount: number };
  provenance: {
    decisionAt: string | null;
    supportingObservationIds: string[];
    costAsOf: string | null;
  };
  shadow_only: true;
  live_execution: false;
  canonical_mutation: false;
  promotionReady: false;
}

const MAX_ROWS = 2_000;
const MAX_CONTRIBUTIONS = 200;
const MAX_BPS = 1_000_000;
const ISO_UTC =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?Z$/;

type Instant = { text: string; ms: number };
type Observation = {
  observationId: string;
  providerId: string;
  sensorFamily: string;
  horizon: string;
  direction: string;
  observedAt: Instant;
};
type Reliability = Observation & {
  groupId: string;
  snapshotAt: Instant;
  expectedMoveBps: number;
  reliability: number;
  uncertaintyBps: number;
  mature: boolean;
};

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function instant(value: unknown): Instant | null {
  if (typeof value !== "string") return null;
  const match = ISO_UTC.exec(value);
  if (!match) return null;
  const ms = Date.parse(value);
  if (!Number.isFinite(ms)) return null;
  const fraction = (match[7] ?? "").padEnd(3, "0");
  const text = `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${
    match[6]
  }.${fraction}Z`;
  return new Date(ms).toISOString() === text ? { text, ms } : null;
}

function boundedNumber(value: unknown, max = MAX_BPS): number | null {
  return typeof value === "number" && Number.isFinite(value) &&
      Math.abs(value) <= max
    ? value
    : null;
}

function positiveInt(
  value: unknown,
  fallback: number,
  maximum: number,
): number {
  return typeof value === "number" && Number.isInteger(value) &&
      Number.isFinite(value) && value > 0
    ? Math.min(value, maximum)
    : fallback;
}

function identifier(value: unknown): string | null {
  return typeof value === "string" && /^[a-z][a-z0-9_.:-]{0,127}$/i.test(value)
    ? value
    : null;
}

function text(value: unknown): string | null {
  return typeof value === "string" && /^[a-zA-Z0-9_.:/-]{1,128}$/.test(value)
    ? value
    : null;
}

function baseReport(decisionAt: string | null): ExpectedEdgeReport {
  return {
    recommendation: "INSUFFICIENT_LAGGED_EVIDENCE",
    eligible: false,
    expectedGrossMoveBps: 0,
    estimatedRoundTripCostBps: null,
    uncertaintyPenaltyBps: 0,
    eventDecayPenaltyBps: 0,
    expectedNetEdgeBps: null,
    matureIndependentGroupCount: 0,
    contributions: [],
    reasons: [],
    invalidEvidenceCount: 0,
    truncated: false,
    futureTelemetry: { futureEvidenceCount: 0 },
    provenance: { decisionAt, supportingObservationIds: [], costAsOf: null },
    shadow_only: true,
    live_execution: false,
    canonical_mutation: false,
    promotionReady: false,
  };
}

function fingerprint(value: Observation | Reliability): string {
  return JSON.stringify(value);
}

export function compileExpectedEdgeAlphaCandidate(
  input: unknown,
  options: ExpectedEdgeOptions,
): ExpectedEdgeReport {
  const decision = instant(options?.decisionAt);
  const report = baseReport(decision?.text ?? null);
  if (!decision) {
    report.reasons.push("invalid decision timestamp");
    report.recommendation = "CONTAMINATED_EVIDENCE";
    return report;
  }
  if (!record(input)) {
    report.reasons.push("input must be an evidence envelope");
    report.recommendation = "CONTAMINATED_EVIDENCE";
    return report;
  }
  const source = Array.isArray(input.sourceObservations)
    ? input.sourceObservations
    : [];
  const reliabilityRows = Array.isArray(input.reliabilitySnapshots)
    ? input.reliabilitySnapshots
    : [];
  const maxRows = positiveInt(options.maxInputRows, MAX_ROWS, MAX_ROWS);
  const maxContributions = positiveInt(
    options.maxContributions,
    MAX_CONTRIBUTIONS,
    MAX_CONTRIBUTIONS,
  );
  const future = (row: Record<string, unknown>): boolean =>
    [row.observedAt, row.snapshotAt, row.asOf].some((value) => {
      const parsed = instant(value);
      return parsed != null && parsed.ms > decision.ms;
    });
  const observations = new Map<string, Observation>();
  const observationFingerprints = new Map<string, Set<string>>();
  const validSource = source.slice(0, maxRows);
  report.truncated = source.length > validSource.length ||
    reliabilityRows.length > maxRows;
  for (const value of validSource) {
    if (!record(value)) {
      report.invalidEvidenceCount++;
      continue;
    }
    if (future(value)) {
      report.futureTelemetry.futureEvidenceCount++;
      continue;
    }
    const observationId = identifier(value.observationId);
    const providerId = identifier(value.providerId);
    const sensorFamily = text(value.sensorFamily);
    const horizon = text(value.horizon);
    const direction = text(value.direction);
    const observedAt = instant(value.observedAt);
    if (
      !observationId || !providerId || !sensorFamily || !horizon ||
      !direction || !observedAt || observedAt.ms > decision.ms
    ) {
      report.invalidEvidenceCount++;
      continue;
    }
    const row = {
      observationId,
      providerId,
      sensorFamily,
      horizon,
      direction,
      observedAt,
    };
    const set = observationFingerprints.get(observationId) ?? new Set<string>();
    set.add(fingerprint(row));
    observationFingerprints.set(observationId, set);
    const previous = observations.get(observationId);
    if (
      !previous || fingerprint(row).localeCompare(fingerprint(previous)) < 0
    ) {
      observations.set(observationId, row);
    }
  }
  const supportingIds = [...observations.keys()].sort();
  report.provenance.supportingObservationIds = supportingIds;
  if ([...observationFingerprints.values()].some((set) => set.size > 1)) {
    report.reasons.push("conflicting source observations");
  }

  const reliability: Reliability[] = [];
  const seenReliability = new Map<string, Set<string>>();
  for (const value of reliabilityRows.slice(0, maxRows)) {
    if (!record(value)) {
      report.invalidEvidenceCount++;
      continue;
    }
    if (future(value)) {
      report.futureTelemetry.futureEvidenceCount++;
      continue;
    }
    const observationId = identifier(value.observationId);
    const base = observationId ? observations.get(observationId) : undefined;
    const groupId = identifier(value.groupId);
    const snapshotAt = instant(value.snapshotAt);
    const expectedMoveBps = boundedNumber(value.expectedMoveBps);
    const reliabilityValue = boundedNumber(value.reliability, 1);
    const uncertaintyBps = boundedNumber(value.uncertaintyBps);
    const mature = value.mature === true;
    if (
      !base || !groupId || !snapshotAt || snapshotAt.ms > decision.ms ||
      expectedMoveBps == null || reliabilityValue == null ||
      reliabilityValue < 0 || reliabilityValue > 1 || uncertaintyBps == null ||
      uncertaintyBps < 0 || !mature
    ) {
      report.invalidEvidenceCount++;
      continue;
    }
    if (
      value.providerId !== undefined && value.providerId !== base.providerId ||
      value.sensorFamily !== undefined &&
        value.sensorFamily !== base.sensorFamily ||
      value.horizon !== undefined && value.horizon !== base.horizon ||
      value.direction !== undefined && value.direction !== base.direction
    ) {
      report.invalidEvidenceCount++;
      continue;
    }
    const row: Reliability = {
      ...base,
      groupId,
      snapshotAt,
      expectedMoveBps,
      reliability: reliabilityValue,
      uncertaintyBps,
      mature,
    };
    const key = `${observationId}|${groupId}|${snapshotAt.ms}`;
    const set = seenReliability.get(key) ?? new Set<string>();
    set.add(fingerprint(row));
    seenReliability.set(key, set);
    reliability.push(row);
  }
  if ([...seenReliability.values()].some((set) => set.size > 1)) {
    report.reasons.push("conflicting reliability snapshots");
  }
  const unique = new Map<string, Reliability>();
  for (const row of reliability) {
    const key = `${row.observationId}|${row.groupId}|${row.snapshotAt.ms}`;
    const previous = unique.get(key);
    if (
      !previous || fingerprint(row).localeCompare(fingerprint(previous)) < 0
    ) {
      unique.set(key, row);
    }
  }
  const rows = [...unique.values()].sort((a, b) =>
    a.groupId.localeCompare(b.groupId) ||
    a.observationId.localeCompare(b.observationId) ||
    a.snapshotAt.ms - b.snapshotAt.ms
  );
  const groups = new Set(rows.map((row) => row.groupId));
  report.matureIndependentGroupCount = groups.size;
  const minimumGroups = positiveInt(options.minimumMatureGroups, 2, 100);
  for (const row of rows.slice(0, maxContributions)) {
    const contributionBps = row.expectedMoveBps * row.reliability;
    report.contributions.push({
      observationId: row.observationId,
      groupId: row.groupId,
      expectedMoveBps: row.expectedMoveBps,
      reliability: row.reliability,
      contributionBps,
    });
  }
  report.expectedGrossMoveBps = report.contributions.reduce(
    (sum, row) => sum + row.contributionBps,
    0,
  );
  report.uncertaintyPenaltyBps = rows.slice(0, maxContributions).reduce(
    (sum, row) => sum + row.uncertaintyBps,
    0,
  );
  const cost = record(input.cost) ? input.cost : null;
  const costAsOf = cost ? instant(cost.asOf) : null;
  if (costAsOf && costAsOf.ms <= decision.ms) {
    report.provenance.costAsOf = costAsOf.text;
  } else if (costAsOf && costAsOf.ms > decision.ms) {
    report.futureTelemetry.futureEvidenceCount++;
  }
  const costParts = cost &&
    ["spreadBps", "feeBps", "slippageBps"].map((key) =>
      boundedNumber(cost[key])
    );
  const fillability = cost ? boundedNumber(cost.fillability, 1) : null;
  if (
    !cost || !costAsOf || costAsOf.ms > decision.ms ||
    !costParts || costParts.some((part) => part == null || part < 0) ||
    fillability == null || fillability <= 0 || fillability > 1 ||
    costParts.every((part) => part === 0)
  ) {
    report.reasons.push("decision-time fillability-aware cost unavailable");
    report.recommendation = "COST_UNAVAILABLE";
    return report;
  }
  report.estimatedRoundTripCostBps =
    (costParts as number[]).reduce((sum, part) => sum + part, 0) /
    fillability;
  const freshness = rows.length
    ? Math.max(...rows.map((row) => decision.ms - row.snapshotAt.ms))
    : 0;
  const eventAt = instant(input.eventAt);
  const eventDecay = eventAt && eventAt.ms <= decision.ms
    ? Math.max(0, (decision.ms - eventAt.ms) / 3_600_000)
    : 0;
  report.eventDecayPenaltyBps = Math.min(
    MAX_BPS,
    eventDecay + Math.max(0, freshness / 3_600_000),
  );
  report.expectedNetEdgeBps = report.expectedGrossMoveBps -
    report.estimatedRoundTripCostBps - report.uncertaintyPenaltyBps -
    report.eventDecayPenaltyBps;
  if (
    report.reasons.length || report.invalidEvidenceCount ||
    report.matureIndependentGroupCount < minimumGroups
  ) {
    if (report.matureIndependentGroupCount < minimumGroups) {
      report.reasons.push("insufficient mature independent groups");
    }
    report.recommendation = report.reasons.some((reason) =>
        reason.includes("conflicting") || reason.includes("source")
      )
      ? "CONTAMINATED_EVIDENCE"
      : "INSUFFICIENT_LAGGED_EVIDENCE";
    return report;
  }
  const margin = boundedNumber(options.minimumNetMarginBps, MAX_BPS) ?? 0;
  report.eligible = report.expectedNetEdgeBps > margin;
  report.recommendation = report.eligible ? "ALLOW_EDGE" : "DOWNGRADE_TO_WAIT";
  if (!report.eligible) report.reasons.push("net edge does not clear margin");
  return report;
}

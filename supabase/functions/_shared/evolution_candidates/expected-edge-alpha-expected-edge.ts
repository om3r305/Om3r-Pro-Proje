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
const MAX_ESTIMATED_COST_BPS = MAX_BPS * 3;
const MAX_DATE_MS = 8.64e15;
const MAX_CADENCE_SECONDS = 7 * 86_400;
const ISO_UTC =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?Z$/;
type Instant = { text: string; ms: number };
type Horizon = { text: string; seconds: number };
type Observation = {
  observationId: string;
  providerId: string;
  sensorFamily: string;
  horizon: Horizon;
  direction: string;
  observedAt: Instant;
  cadenceSeconds: number;
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
  if (typeof value === "number") {
    if (
      !Number.isFinite(value) || !Number.isInteger(value) ||
      Math.abs(value) > MAX_DATE_MS
    ) return null;
    const date = new Date(value);
    return { text: date.toISOString(), ms: value };
  }
  if (typeof value !== "string") return null;
  const match = ISO_UTC.exec(value);
  if (!match) return null;
  const ms = Date.parse(value);
  const fraction = (match[7] ?? "").padEnd(3, "0");
  const text = `${match[1]}-${match[2]}-${match[3]}T${match[4]}:${match[5]}:${
    match[6]
  }.${fraction}Z`;
  return Number.isFinite(ms) && new Date(ms).toISOString() === text
    ? { text, ms }
    : null;
}
function horizon(value: unknown): Horizon | null {
  if (typeof value !== "string") return null;
  const match = /^(\d+)(s|m|h|d)$/.exec(value);
  if (!match) return null;
  const seconds = Number(match[1]) *
    ({ s: 1, m: 60, h: 3600, d: 86400 } as Record<string, number>)[match[2]];
  return seconds > 0 && seconds <= 86_400 ? { text: value, seconds } : null;
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
function cadence(value: unknown): number | null {
  const result = boundedNumber(value, MAX_CADENCE_SECONDS);
  return result != null && result > 0 ? result : null;
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
function fingerprint(value: unknown): string {
  return JSON.stringify(value) ?? "";
}
function futureEvidence(
  row: Record<string, unknown>,
  decision: Instant,
): boolean {
  return Object.values(row).some((value) => {
    const parsed = instant(value);
    return parsed != null && parsed.ms > decision.ms;
  });
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
  report.truncated = source.length > maxRows ||
    reliabilityRows.length > maxRows;
  const addInvalid = () => {
    report.invalidEvidenceCount++;
  };
  const observations = new Map<string, Observation>();
  const conflicts = new Set<string>();
  const boundedSource = [...source].sort((a, b) =>
    fingerprint(a).localeCompare(fingerprint(b))
  ).slice(0, maxRows);
  for (const value of boundedSource) {
    if (!record(value)) {
      addInvalid();
      continue;
    }
    if (futureEvidence(value, decision)) {
      report.futureTelemetry.futureEvidenceCount++;
      continue;
    }
    const id = identifier(value.observationId);
    const providerId = identifier(value.providerId);
    const sensorFamily = text(value.sensorFamily);
    const parsedHorizon = horizon(value.horizon);
    const direction = text(value.direction);
    const observedAt = instant(value.observedAt);
    const cadenceSeconds = cadence(value.cadenceSeconds);
    const start = instant(value.evaluationStartAt);
    const end = instant(value.evaluationEndAt);
    const aligned = parsedHorizon && start && end &&
      start.ms === observedAt?.ms &&
      end.ms - start.ms === parsedHorizon.seconds * 1000;
    if (
      !id || !providerId || !sensorFamily || !parsedHorizon || !direction ||
      !observedAt || cadenceSeconds == null || !aligned ||
      end!.ms > decision.ms || decision.ms - observedAt.ms >
        Math.min(
          7 * 86_400_000,
          Math.max(parsedHorizon.seconds, cadenceSeconds * 2) * 1000 + 300_000,
        )
    ) {
      addInvalid();
      continue;
    }
    const row = {
      observationId: id,
      providerId,
      sensorFamily,
      horizon: parsedHorizon,
      direction,
      observedAt,
      cadenceSeconds,
    };
    const prior = observations.get(id);
    if (prior && fingerprint(prior) !== fingerprint(row)) conflicts.add(id);
    if (!prior || fingerprint(row).localeCompare(fingerprint(prior)) < 0) {
      observations.set(id, row);
    }
  }
  report.provenance.supportingObservationIds = [...observations.keys()].sort();
  if (conflicts.size) report.reasons.push("conflicting source observations");
  const validReliability: Reliability[] = [];
  const reliabilityKeys = new Map<string, Set<string>>();
  const boundedReliability = [...reliabilityRows].sort((a, b) =>
    fingerprint(a).localeCompare(fingerprint(b))
  ).slice(0, maxRows);
  for (const value of boundedReliability) {
    if (!record(value)) {
      addInvalid();
      continue;
    }
    if (futureEvidence(value, decision)) {
      report.futureTelemetry.futureEvidenceCount++;
      continue;
    }
    const id = identifier(value.observationId);
    const base = id ? observations.get(id) : undefined;
    const groupId = identifier(value.groupId);
    const snapshotAt = instant(value.snapshotAt);
    const expectedMoveBps = boundedNumber(value.expectedMoveBps);
    const reliabilityValue = boundedNumber(value.reliability, 1);
    const uncertaintyBps = boundedNumber(value.uncertaintyBps);
    const provenance = identifier(value.provenance);
    const snapshotCadence = cadence(value.cadenceSeconds);
    if (
      !base || !groupId || !snapshotAt || expectedMoveBps == null ||
      reliabilityValue == null || reliabilityValue < 0 ||
      reliabilityValue > 1 ||
      uncertaintyBps == null || uncertaintyBps < 0 || value.mature !== true ||
      !provenance || snapshotCadence == null ||
      snapshotCadence !== base.cadenceSeconds ||
      decision.ms - snapshotAt.ms >
        Math.min(
          7 * 86_400_000,
          Math.max(base.horizon.seconds, snapshotCadence * 2) * 1000 + 300_000,
        ) ||
      value.horizon !== base.horizon.text
    ) {
      addInvalid();
      continue;
    }
    const row = {
      ...base,
      groupId,
      snapshotAt,
      expectedMoveBps,
      reliability: reliabilityValue,
      uncertaintyBps,
      mature: true,
    };
    const key = `${id}|${groupId}|${snapshotAt.ms}`;
    const set = reliabilityKeys.get(key) ?? new Set<string>();
    set.add(fingerprint(row));
    reliabilityKeys.set(key, set);
    validReliability.push(row);
  }
  if ([...reliabilityKeys.values()].some((set) => set.size > 1)) {
    report.reasons.push("conflicting reliability snapshots");
  }
  const unique = new Map<string, Reliability>();
  for (const row of validReliability) {
    const key = `${row.observationId}|${row.groupId}|${row.snapshotAt.ms}`;
    const prior = unique.get(key);
    if (!prior || fingerprint(row).localeCompare(fingerprint(prior)) < 0) {
      unique.set(key, row);
    }
  }
  const rows = [...unique.values()].sort((a, b) =>
    a.groupId.localeCompare(b.groupId) ||
    a.observationId.localeCompare(b.observationId) ||
    a.snapshotAt.ms - b.snapshotAt.ms
  );
  report.matureIndependentGroupCount =
    new Set(rows.map((row) => row.groupId)).size;
  const minimumGroups = positiveInt(options.minimumMatureGroups, 2, 100);
  if (report.truncated || rows.length > maxContributions) {
    report.reasons.push("incomplete bounded evidence");
  }
  const selected = rows.slice(0, maxContributions);
  for (const row of selected) {
    report.contributions.push({
      observationId: row.observationId,
      groupId: row.groupId,
      expectedMoveBps: row.expectedMoveBps,
      reliability: row.reliability,
      contributionBps: row.expectedMoveBps * row.reliability,
    });
  }
  report.expectedGrossMoveBps = selected.reduce(
    (sum, row) => sum + row.expectedMoveBps * row.reliability,
    0,
  );
  report.uncertaintyPenaltyBps = selected.reduce(
    (sum, row) => sum + row.uncertaintyBps,
    0,
  );
  const cost = record(input.cost) ? input.cost : null;
  const costAsOf = cost ? instant(cost.asOf) : null;
  if (costAsOf && costAsOf.ms <= decision.ms) {
    report.provenance.costAsOf = costAsOf.text;
  } else if (costAsOf) report.futureTelemetry.futureEvidenceCount++;
  const costCadence = cost ? cadence(cost.cadenceSeconds) : null;
  const costSource = cost ? identifier(cost.sourceId) : null;
  const costParts = cost
    ? ["spreadBps", "feeBps", "slippageBps"].map((key) =>
      boundedNumber(cost[key])
    )
    : null;
  const fillability = cost ? boundedNumber(cost.fillability, 1) : null;
  let validatedCostParts: number[] | null = null;
  if (costParts !== null) {
    const numericParts: number[] = [];
    let valid = true;
    for (const part of costParts) {
      if (part == null || !Number.isFinite(part) || part < 0) {
        valid = false;
        break;
      }
      numericParts.push(part);
    }
    if (valid) validatedCostParts = numericParts;
  }
  const validatedFillability = fillability != null &&
      Number.isFinite(fillability) && fillability > 0 && fillability <= 1
    ? fillability
    : null;
  if (
    !cost || !costAsOf || costAsOf.ms > decision.ms || !costSource ||
    costCadence == null || validatedCostParts === null ||
    validatedCostParts.length !== 3 || validatedFillability === null ||
    validatedCostParts.every((part) => part === 0) ||
    decision.ms - costAsOf.ms >
      Math.min(7 * 86_400_000, costCadence * 2 * 1000 + 300_000)
  ) {
    report.reasons.push("decision-time fillability-aware cost unavailable");
    report.recommendation = "COST_UNAVAILABLE";
    return report;
  }
  const totalCostBps = validatedCostParts.reduce((sum, part) => sum + part, 0);
  if (totalCostBps <= 0) {
    report.reasons.push("decision-time fillability-aware cost unavailable");
    report.recommendation = "COST_UNAVAILABLE";
    return report;
  }
  const estimatedRoundTripCostBps = totalCostBps / validatedFillability;
  if (
    validatedFillability < totalCostBps / MAX_ESTIMATED_COST_BPS ||
    !Number.isFinite(estimatedRoundTripCostBps) ||
    estimatedRoundTripCostBps > MAX_ESTIMATED_COST_BPS
  ) {
    report.reasons.push("decision-time fillability-aware cost unavailable");
    report.recommendation = "COST_UNAVAILABLE";
    return report;
  }
  report.estimatedRoundTripCostBps = estimatedRoundTripCostBps;
  const eventAt = instant(input.eventAt);
  const eventCadence = cadence(input.eventCadenceSeconds);
  if (eventAt && eventAt.ms > decision.ms) {
    report.futureTelemetry.futureEvidenceCount++;
  }
  if (
    input.eventAt !== undefined &&
    (!eventAt || eventAt.ms > decision.ms || eventCadence == null ||
      decision.ms - eventAt.ms >
        Math.min(7 * 86_400_000, eventCadence * 2 * 1000 + 300_000))
  ) {
    addInvalid();
    report.reasons.push("event evidence unavailable");
  }
  const freshness = rows.length
    ? Math.max(...rows.map((row) => decision.ms - row.snapshotAt.ms))
    : 0;
  report.eventDecayPenaltyBps = Math.min(
    MAX_BPS,
    (eventAt ? decision.ms - eventAt.ms : 0) / 3_600_000 +
      freshness /
        Math.max(3_600_000, (rows[0]?.horizon.seconds ?? 3600) * 1000),
  );
  report.expectedNetEdgeBps = report.expectedGrossMoveBps -
    report.estimatedRoundTripCostBps -
    report.uncertaintyPenaltyBps - report.eventDecayPenaltyBps;
  if (!Number.isFinite(report.expectedNetEdgeBps)) {
    report.reasons.push("decision-time edge arithmetic unavailable");
    report.recommendation = "COST_UNAVAILABLE";
    return report;
  }
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

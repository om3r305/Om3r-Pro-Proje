import { canonicalIndependentGroup } from "../alpha_decision.ts";

export type CostControlRecommendation =
  | "ALLOW_EDGE"
  | "DOWNGRADE_TO_WAIT"
  | "COST_UNAVAILABLE"
  | "INSUFFICIENT_LAGGED_EVIDENCE"
  | "CONTAMINATED_EVIDENCE";

export interface CostControlOptions {
  decisionAt: string | number;
  minimumNetEdgeBps?: number;
  minimumEdgeToCostMargin?: number;
  minimumMatureGroups?: number;
  maxInputRows?: number;
  maxRankedOpportunities?: number;
}

export interface CostControlRankedOpportunity {
  opportunityId: string;
  grossEdgeBps: number;
  reliability: number;
  weightedGrossEdgeBps: number;
  netEdgeBps: number;
  edgeToCostMargin: number;
  costBps: number;
  observationAt: string;
  reliabilityAsOf: string;
  reliabilityGroups: string[];
  reliabilitySourceObservationIds: string[];
}

export interface CostControlReport {
  recommendation: CostControlRecommendation;
  eligible: boolean;
  rankedOpportunities: CostControlRankedOpportunity[];
  selectedOpportunityId: string | null;
  roundTripCostBps: number | null;
  costComponentsBps: { spread: number; fee: number; depth: number };
  costConvention: "ROUND_TRIP_COMPONENTS_BPS";
  matureIndependentGroupCount: number;
  invalidEvidenceCount: number;
  truncated: boolean;
  reasons: string[];
  futureTelemetry: { futureEvidenceCount: number };
  provenance: { decisionAt: string | null; costAsOf: string | null };
  shadow_only: true;
  live_execution: false;
  canonical_mutation: false;
  promotionReady: false;
}

const MAX_ROWS = 2_000;
const MAX_RANKED = 200;
const MAX_BPS = 1_000_000;
const MAX_COST = 3_000_000;
const MAX_DATE_MS = 8.64e15;
const ISO_UTC =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?Z$/;
type Instant = { text: string; ms: number };

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function parseInstant(value: unknown): Instant | null {
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

function identifier(value: unknown): string | null {
  return typeof value === "string" &&
      /^[a-z][a-z0-9_.:-]{0,127}$/i.test(value)
    ? value
    : null;
}

function finite(value: unknown, maximum = MAX_BPS): number | null {
  return typeof value === "number" && Number.isFinite(value) &&
      Math.abs(value) <= maximum
    ? value
    : null;
}

function positiveLimit(value: unknown, fallback: number, maximum: number) {
  return typeof value === "number" && Number.isInteger(value) &&
      Number.isFinite(value) && value > 0
    ? Math.min(value, maximum)
    : fallback;
}

function fingerprint(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(fingerprint).join(",")}]`;
  if (record(value)) {
    return `{${
      Object.keys(value).sort().map((key) =>
        `${JSON.stringify(key)}:${fingerprint(value[key])}`
      ).join(",")
    }}`;
  }
  return JSON.stringify(value) ?? "";
}

function futureRow(row: Record<string, unknown>, decision: Instant): boolean {
  return Object.values(row).some((value) => {
    const parsed = parseInstant(value);
    return parsed !== null && parsed.ms > decision.ms;
  });
}

function report(decisionAt: string | null): CostControlReport {
  return {
    recommendation: "INSUFFICIENT_LAGGED_EVIDENCE",
    eligible: false,
    rankedOpportunities: [],
    selectedOpportunityId: null,
    roundTripCostBps: null,
    costComponentsBps: { spread: 0, fee: 0, depth: 0 },
    costConvention: "ROUND_TRIP_COMPONENTS_BPS",
    matureIndependentGroupCount: 0,
    invalidEvidenceCount: 0,
    truncated: false,
    reasons: [],
    futureTelemetry: { futureEvidenceCount: 0 },
    provenance: { decisionAt, costAsOf: null },
    shadow_only: true,
    live_execution: false,
    canonical_mutation: false,
    promotionReady: false,
  };
}

type ParsedSourceObservation = {
  observationId: string;
  opportunityId: string;
  providerId: string;
  sourceId: string;
  lineageId: string;
  independentGroup: string;
  sensorFamily: string;
  sensorHorizon: string;
  direction: -1 | 1;
  observedAt: Instant;
};

function direction(value: unknown): -1 | 1 | null {
  return value === -1 || value === 1 ? value : null;
}

/**
 * Compiles a bounded, decision-time-only cost-control projection.
 * This function has no execution or persistence side effects.
 */
export function compileCostControlAlphaCandidate(
  input: unknown,
  options: CostControlOptions,
): CostControlReport {
  const decision = parseInstant(options?.decisionAt);
  const result = report(decision?.text ?? null);
  if (!decision) {
    result.reasons.push("invalid decision timestamp");
    result.recommendation = "CONTAMINATED_EVIDENCE";
    return result;
  }
  if (!record(input)) {
    result.reasons.push("input must be an evidence envelope");
    result.recommendation = "CONTAMINATED_EVIDENCE";
    return result;
  }

  const opportunities = Array.isArray(input.opportunities)
    ? input.opportunities
    : [];
  const sourceObservations = Array.isArray(input.sourceObservations)
    ? input.sourceObservations
    : [];
  const reliabilitySnapshots = Array.isArray(input.reliabilitySnapshots)
    ? input.reliabilitySnapshots
    : [];
  const maxRows = positiveLimit(options.maxInputRows, MAX_ROWS, MAX_ROWS);
  const maxRanked = positiveLimit(
    options.maxRankedOpportunities,
    MAX_RANKED,
    MAX_RANKED,
  );
  const invalid = () => result.invalidEvidenceCount++;
  const pointInTimeOpportunities = opportunities.filter((value) => {
    if (record(value) && futureRow(value, decision)) {
      result.futureTelemetry.futureEvidenceCount++;
      return false;
    }
    return true;
  });
  const pointInTimeReliability = reliabilitySnapshots.filter((value) => {
    if (record(value) && futureRow(value, decision)) {
      result.futureTelemetry.futureEvidenceCount++;
      return false;
    }
    return true;
  });
  const pointInTimeSources = sourceObservations.filter((value) => {
    if (record(value) && futureRow(value, decision)) {
      result.futureTelemetry.futureEvidenceCount++;
      return false;
    }
    return true;
  });
  result.truncated = pointInTimeOpportunities.length > maxRows ||
    pointInTimeSources.length > maxRows ||
    pointInTimeReliability.length > maxRows;
  const selectedOpportunities = pointInTimeOpportunities
    .sort((a, b) => fingerprint(a).localeCompare(fingerprint(b)))
    .slice(0, maxRows);
  const selectedReliability = pointInTimeReliability
    .sort((a, b) => fingerprint(a).localeCompare(fingerprint(b)))
    .slice(0, maxRows);

  const parsedOpportunities = new Map<string, {
    id: string;
    gross: number;
    observedAt: Instant;
  }>();
  const opportunityConflicts = new Set<string>();
  for (const value of selectedOpportunities) {
    if (!record(value)) {
      invalid();
      continue;
    }
    const id = identifier(value.opportunityId);
    const observedAt = parseInstant(value.observedAt);
    const gross = finite(value.grossEdgeBps);
    if (
      !id || !observedAt || observedAt.ms > decision.ms || gross == null ||
      gross <= 0 || decision.ms - observedAt.ms > 7 * 86_400_000
    ) {
      invalid();
      continue;
    }
    const parsed = { id, gross, observedAt };
    const prior = parsedOpportunities.get(id);
    if (prior && fingerprint(prior) !== fingerprint(parsed)) {
      opportunityConflicts.add(id);
    }
    if (!prior || fingerprint(parsed).localeCompare(fingerprint(prior)) < 0) {
      parsedOpportunities.set(id, parsed);
    }
  }
  if (opportunityConflicts.size) {
    result.reasons.push("conflicting opportunities");
  }

  const parsedSources = new Map<string, ParsedSourceObservation>();
  const sourceConflicts = new Set<string>();
  const selectedSources = pointInTimeSources
    .sort((a, b) => fingerprint(a).localeCompare(fingerprint(b)))
    .slice(0, maxRows);
  for (const value of selectedSources) {
    if (!record(value)) {
      invalid();
      continue;
    }
    const observationId = identifier(value.observationId);
    const opportunityId = identifier(value.opportunityId);
    const providerId = identifier(value.providerId);
    const sourceId = identifier(value.sourceId);
    const lineageId = identifier(value.lineageId);
    const independentGroup = identifier(value.independentGroup);
    const sensorFamily = identifier(value.sensorFamily);
    const sensorHorizon = identifier(value.sensorHorizon);
    const observedAt = parseInstant(value.observedAt);
    const parsedDirection = direction(value.direction);
    const opportunity = opportunityId
      ? parsedOpportunities.get(opportunityId)
      : undefined;
    const parsed = observationId && opportunityId && providerId && sourceId &&
        lineageId &&
        independentGroup && sensorFamily && sensorHorizon && observedAt &&
        parsedDirection
      ? {
        observationId,
        opportunityId,
        providerId,
        sourceId,
        lineageId,
        independentGroup,
        sensorFamily,
        sensorHorizon,
        direction: parsedDirection,
        observedAt,
      }
      : null;
    if (
      !parsed || !opportunity ||
      parsed.observedAt.ms !== opportunity.observedAt.ms ||
      parsed.observedAt.ms > decision.ms ||
      decision.ms - parsed.observedAt.ms > 7 * 86_400_000
    ) {
      invalid();
      continue;
    }
    const prior = parsedSources.get(parsed.observationId);
    if (prior && fingerprint(prior) !== fingerprint(parsed)) {
      sourceConflicts.add(parsed.observationId);
    }
    if (!prior || fingerprint(parsed).localeCompare(fingerprint(prior)) < 0) {
      parsedSources.set(parsed.observationId, parsed);
    }
  }
  if (sourceConflicts.size) {
    result.reasons.push("conflicting source observations");
  }

  const reliabilities = new Map<string, {
    opportunityId: string;
    groupId: string;
    canonicalGroupId: string;
    reliability: number;
    snapshotAt: Instant;
    sourceObservationId: string;
    sourceId: string;
    lineageId: string;
  }>();
  const reliabilityConflicts = new Set<string>();
  const lineageBindings = new Map<string, string>();
  for (const value of selectedReliability) {
    if (!record(value)) {
      invalid();
      continue;
    }
    const opportunityId = identifier(value.opportunityId);
    const groupId = identifier(value.groupId);
    const snapshotAt = parseInstant(value.snapshotAt);
    const reliability = finite(value.reliability, 1);
    const provenance = record(value.provenance) ? value.provenance : null;
    const sourceObservationId = provenance
      ? identifier(provenance.sourceObservationId)
      : null;
    const sourceId = provenance ? identifier(provenance.sourceId) : null;
    const lineageId = provenance ? identifier(provenance.lineageId) : null;
    const source = sourceObservationId
      ? parsedSources.get(sourceObservationId)
      : undefined;
    const independent = provenance?.independent === true;
    const rawGroup = provenance
      ? identifier(provenance.rawIndependentGroup)
      : null;
    const sensorFamily = provenance
      ? identifier(provenance.sensorFamily)
      : null;
    const sensorHorizon = provenance
      ? identifier(provenance.sensorHorizon)
      : null;
    const provenanceDirection = provenance
      ? direction(provenance.direction)
      : null;
    const snapshotWindowEnd = provenance
      ? parseInstant(provenance.snapshotWindowEnd)
      : null;
    const snapshotGeneratedAt = provenance
      ? parseInstant(provenance.snapshotGeneratedAt)
      : null;
    const key = opportunityId && groupId && snapshotAt
      ? `${opportunityId}|${
        canonicalIndependentGroup(groupId)
      }|${snapshotAt.ms}`
      : null;
    const lineageKey = sourceId && lineageId
      ? `${sourceId}|${lineageId}`
      : null;
    const lineageBinding = lineageKey && opportunityId && groupId
      ? `${opportunityId}|${groupId}`
      : null;
    const priorLineageBinding = lineageKey
      ? lineageBindings.get(lineageKey)
      : undefined;
    if (
      lineageKey && lineageBinding &&
      priorLineageBinding && priorLineageBinding !== lineageBinding
    ) {
      reliabilityConflicts.add(lineageKey);
    } else if (lineageKey && lineageBinding) {
      lineageBindings.set(lineageKey, lineageBinding);
    }
    if (
      !opportunityId || !parsedOpportunities.has(opportunityId) ||
      !groupId || !snapshotAt || snapshotAt.ms > decision.ms ||
      reliability == null || reliability <= 0 || reliability > 1 ||
      value.mature !== true || !source ||
      source.opportunityId !== opportunityId ||
      source.independentGroup !== groupId ||
      source.independentGroup !== rawGroup ||
      source.sourceId !== sourceId ||
      source.lineageId !== lineageId ||
      source.sensorFamily !== sensorFamily ||
      source.sensorHorizon !== sensorHorizon ||
      source.direction !== provenanceDirection ||
      !sourceObservationId || !sourceId || !lineageId || !independent ||
      !snapshotWindowEnd || !snapshotGeneratedAt ||
      (lineageKey !== null && priorLineageBinding !== undefined &&
        priorLineageBinding !== lineageBinding) ||
      snapshotWindowEnd.ms !== snapshotAt.ms ||
      snapshotGeneratedAt.ms !== snapshotAt.ms ||
      decision.ms - snapshotAt.ms > 7 * 86_400_000
    ) {
      invalid();
      continue;
    }
    const parsed = {
      opportunityId,
      groupId,
      canonicalGroupId: canonicalIndependentGroup(groupId),
      reliability,
      snapshotAt,
      sourceObservationId,
      sourceId,
      lineageId,
    };
    const prior = reliabilities.get(key!);
    if (prior && fingerprint(prior) !== fingerprint(parsed)) {
      reliabilityConflicts.add(key!);
    }
    if (!prior || fingerprint(parsed).localeCompare(fingerprint(prior)) < 0) {
      reliabilities.set(key!, parsed);
    }
  }
  if (reliabilityConflicts.size) {
    result.reasons.push("conflicting reliability snapshots");
  }
  result.matureIndependentGroupCount = new Set(
    [...reliabilities.values()].map((value) => value.canonicalGroupId),
  ).size;

  const cost = record(input.cost) ? input.cost : null;
  const costAsOf = cost ? parseInstant(cost.asOf) : null;
  if (costAsOf && costAsOf.ms <= decision.ms) {
    result.provenance.costAsOf = costAsOf.text;
  } else if (costAsOf) {
    result.futureTelemetry.futureEvidenceCount++;
  }
  const spread = cost ? finite(cost.spreadBps) : null;
  const fee = cost ? finite(cost.feeBps) : null;
  const depth = cost ? finite(cost.depthCostBps ?? cost.slippageBps) : null;
  const fillability = cost ? finite(cost.fillability, 1) : null;
  const cadence = cost ? finite(cost.cadenceSeconds, 7 * 86_400) : null;
  const sourceId = cost ? identifier(cost.sourceId) : null;
  const validCost = cost !== null && cost.costConvention ===
      "ONE_WAY_COMPONENTS_BPS" &&
    costAsOf !== null &&
    costAsOf.ms <= decision.ms && spread !== null && spread >= 0 &&
    fee !== null && fee >= 0 && depth !== null && depth >= 0 &&
    fillability !== null && fillability > 0 && fillability <= 1 &&
    cadence !== null && cadence > 0 && sourceId !== null &&
    spread + fee + depth > 0 &&
    decision.ms - costAsOf.ms <= Math.min(
        7 * 86_400_000,
        cadence * 2 * 1_000 + 300_000,
      );
  const rawCost = validCost ? (spread! + fee! + depth!) : null;
  const roundTrip = rawCost === null ? null : rawCost * 2 / fillability!;
  if (
    !validCost || roundTrip === null || !Number.isFinite(roundTrip) ||
    roundTrip <= 0 || roundTrip > MAX_COST
  ) {
    result.reasons.push("decision-time round-trip cost unavailable");
    result.recommendation = "COST_UNAVAILABLE";
    return result;
  }
  result.roundTripCostBps = roundTrip;
  const costScale = 2 / fillability!;
  result.costComponentsBps = {
    spread: spread! * costScale,
    fee: fee! * costScale,
    depth: depth! * costScale,
  };

  const verifiedReliability = [...reliabilities.values()];
  const candidates = [
    ...new Set(verifiedReliability.map((value) => value.opportunityId)),
  ]
    .map((opportunityId) => {
      const opportunity = parsedOpportunities.get(opportunityId)!;
      const evidence = verifiedReliability.filter((value) =>
        value.opportunityId === opportunityId
      );
      const reliability = evidence.reduce((sum, value) =>
        sum + value.reliability, 0) / evidence.length;
      const weightedGross = opportunity.gross * reliability;
      const net = weightedGross - roundTrip!;
      return {
        opportunityId: opportunity.id,
        grossEdgeBps: opportunity.gross,
        reliability,
        weightedGrossEdgeBps: weightedGross,
        netEdgeBps: net,
        edgeToCostMargin: net / roundTrip!,
        costBps: roundTrip!,
        observationAt: opportunity.observedAt.text,
        reliabilityAsOf: [...evidence].sort((a, b) =>
          b.snapshotAt.ms - a.snapshotAt.ms ||
          a.sourceObservationId.localeCompare(b.sourceObservationId)
        )[0].snapshotAt.text,
        reliabilityGroups: [
          ...new Set(evidence.map((value) =>
            value.canonicalGroupId
          )),
        ].sort(),
        reliabilitySourceObservationIds: evidence.map((value) =>
          value.sourceObservationId
        ).sort(),
      };
    })
    .sort((a, b) =>
      b.netEdgeBps - a.netEdgeBps ||
      a.opportunityId.localeCompare(b.opportunityId)
    );
  if (result.truncated) result.reasons.push("incomplete bounded evidence");
  const minimumGroups = positiveLimit(options.minimumMatureGroups, 2, 100);
  if (
    result.reasons.length || result.invalidEvidenceCount ||
    result.matureIndependentGroupCount < minimumGroups
  ) {
    if (result.matureIndependentGroupCount < minimumGroups) {
      result.reasons.push("insufficient mature independent groups");
    }
    result.recommendation = result.reasons.some((reason) =>
        reason.includes("conflicting")
      )
      ? "CONTAMINATED_EVIDENCE"
      : "INSUFFICIENT_LAGGED_EVIDENCE";
    return result;
  }
  result.rankedOpportunities = candidates.slice(0, maxRanked);
  const top = candidates[0];
  if (!top) {
    result.reasons.push("no opportunity has mature lagged reliability");
    result.recommendation = "INSUFFICIENT_LAGGED_EVIDENCE";
    return result;
  }
  result.selectedOpportunityId = top.opportunityId;
  const minimumNet = finite(options.minimumNetEdgeBps) ?? 0;
  const minimumMargin = finite(options.minimumEdgeToCostMargin) ?? 0;
  result.eligible = top.netEdgeBps > minimumNet &&
    top.edgeToCostMargin > minimumMargin;
  result.recommendation = result.eligible ? "ALLOW_EDGE" : "DOWNGRADE_TO_WAIT";
  if (!result.eligible) {
    result.reasons.push("edge-to-cost margin is insufficient");
  }
  return result;
}

export const compileCostControlAlphaExpectedEdgeCandidate =
  compileCostControlAlphaCandidate;
import { canonicalIndependentGroup } from "../alpha_decision.ts";

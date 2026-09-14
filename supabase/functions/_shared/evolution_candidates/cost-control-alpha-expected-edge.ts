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
}

export interface CostControlReport {
  recommendation: CostControlRecommendation;
  eligible: boolean;
  rankedOpportunities: CostControlRankedOpportunity[];
  selectedOpportunityId: string | null;
  roundTripCostBps: number | null;
  costComponentsBps: { spread: number; fee: number; depth: number };
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
  const reliabilitySnapshots = Array.isArray(input.reliabilitySnapshots)
    ? input.reliabilitySnapshots
    : [];
  const maxRows = positiveLimit(options.maxInputRows, MAX_ROWS, MAX_ROWS);
  const maxRanked = positiveLimit(
    options.maxRankedOpportunities,
    MAX_RANKED,
    MAX_RANKED,
  );
  result.truncated = opportunities.length > maxRows ||
    reliabilitySnapshots.length > maxRows;
  const invalid = () => result.invalidEvidenceCount++;
  const selectedOpportunities = [...opportunities]
    .sort((a, b) => fingerprint(a).localeCompare(fingerprint(b)))
    .slice(0, maxRows);
  const selectedReliability = [...reliabilitySnapshots]
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
    if (futureRow(value, decision)) {
      result.futureTelemetry.futureEvidenceCount++;
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

  const reliabilities = new Map<string, {
    opportunityId: string;
    groupId: string;
    reliability: number;
    snapshotAt: Instant;
  }>();
  const reliabilityConflicts = new Set<string>();
  for (const value of selectedReliability) {
    if (!record(value)) {
      invalid();
      continue;
    }
    if (futureRow(value, decision)) {
      result.futureTelemetry.futureEvidenceCount++;
      continue;
    }
    const opportunityId = identifier(value.opportunityId);
    const groupId = identifier(value.groupId);
    const snapshotAt = parseInstant(value.snapshotAt);
    const reliability = finite(value.reliability, 1);
    const key = opportunityId && groupId && snapshotAt
      ? `${opportunityId}|${groupId}|${snapshotAt.ms}`
      : null;
    if (
      !opportunityId || !parsedOpportunities.has(opportunityId) ||
      !groupId || !snapshotAt || snapshotAt.ms > decision.ms ||
      reliability == null || reliability <= 0 || reliability > 1 ||
      value.mature !== true || decision.ms - snapshotAt.ms > 7 * 86_400_000
    ) {
      invalid();
      continue;
    }
    const parsed = { opportunityId, groupId, reliability, snapshotAt };
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
    [...reliabilities.values()].map((value) => value.groupId),
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
  const validCost = cost !== null && costAsOf !== null &&
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
  result.costComponentsBps = { spread: spread!, fee: fee!, depth: depth! };

  const candidates = [...reliabilities.values()]
    .map((value) => {
      const opportunity = parsedOpportunities.get(value.opportunityId)!;
      const weightedGross = opportunity.gross * value.reliability;
      const net = weightedGross - roundTrip!;
      return {
        opportunityId: opportunity.id,
        grossEdgeBps: opportunity.gross,
        reliability: value.reliability,
        weightedGrossEdgeBps: weightedGross,
        netEdgeBps: net,
        edgeToCostMargin: net / roundTrip!,
        costBps: roundTrip!,
        observationAt: opportunity.observedAt.text,
        reliabilityAsOf: value.snapshotAt.text,
        groupId: value.groupId,
      };
    })
    .sort((a, b) =>
      b.netEdgeBps - a.netEdgeBps ||
      a.opportunityId.localeCompare(b.opportunityId) ||
      a.groupId.localeCompare(b.groupId)
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
  result.rankedOpportunities = candidates.slice(0, maxRanked).map(
    ({ groupId: _groupId, ...candidate }) => candidate,
  );
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

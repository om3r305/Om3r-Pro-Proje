export const EVOLUTION_ALPHA_INTELLIGENCE_VERSION = "brian.alpha-intelligence-challenger.v1";

export interface LaggedReliabilityEvidence {
  group: string;
  sampleCount: number;
  bayesianHitRate: number;
  avgSignedBps: number;
  avgCostAdjustedSignedBps: number;
  outcomeHorizonSeconds: number;
  snapshotWindowEnd: string;
  snapshotGeneratedAt: string;
}

export interface EvidenceFreshness {
  group: string;
  observedAt: string;
  horizon: string;
}

export interface ExpectedEdgeInput {
  decisionObservedAt: string;
  direction: -1 | 1;
  evidenceScore: number;
  roundTripCostBps: number | null;
  reliability: LaggedReliabilityEvidence[];
  freshness: EvidenceFreshness[];
  minimumNetMarginBps?: number;
}

export interface GroupEdgeContribution {
  group: string;
  samples: number;
  measuredReliability: number;
  boundedReliability: number;
  maturity: number;
  historicalGrossSignedBps: number;
  historicalAfterCostSignedBps: number;
  contributionWeight: number;
  contributionGrossBps: number;
  outcomeHorizonSeconds: number;
  snapshotWindowEnd: string;
  snapshotGeneratedAt: string;
}

export interface ExpectedEdgeDecomposition {
  version: typeof EVOLUTION_ALPHA_INTELLIGENCE_VERSION;
  expectedGrossMoveBps: number | null;
  estimatedRoundTripCostBps: number | null;
  uncertaintyPenaltyBps: number | null;
  eventDecayPenaltyBps: number | null;
  expectedNetEdgeBps: number | null;
  minimumNetMarginBps: number;
  eligible: boolean;
  recommendation: "ALLOW_EDGE" | "DOWNGRADE_TO_WAIT" | "COST_UNAVAILABLE" | "INSUFFICIENT_LAGGED_EVIDENCE" | "CONTAMINATED_EVIDENCE";
  matureGroupCount: number;
  groupContributions: GroupEdgeContribution[];
  reliabilityWeights: Record<string, number>;
  pitClear: boolean;
  reasons: string[];
}

const PRIOR_RELIABILITY = 0.5;
const SHRINK_SAMPLES = 200;
const MAX_RELIABILITY_DEVIATION = 0.15;
const MIN_MATURE_SAMPLES = 100;
const MIN_MATURE_GROUPS = 2;
const GROSS_CAP_BPS = 75;
const UNCERTAINTY_FLOOR_BPS = 1.5;
const IMMATURITY_PENALTY_BPS = 8;

function finite(value: unknown): number | null {
  if (value == null || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, value));
}

function parseTime(value: string): number | null {
  const t = Date.parse(value);
  return Number.isFinite(t) ? t : null;
}

export function freshnessBudgetSeconds(horizon: string): number | null {
  if (horizon === "MICRO_1_5M") return 5 * 60;
  if (horizon === "FAST_5_30M") return 30 * 60;
  if (horizon === "EVENT_DRIVEN") return 60 * 60;
  if (horizon === "DAILY") return 36 * 60 * 60;
  return null;
}

export function boundedProspectiveReliability(sampleCount: number, measuredReliability: number): number {
  if (!Number.isFinite(sampleCount) || sampleCount < 0) throw new Error("sampleCount must be non-negative");
  if (!Number.isFinite(measuredReliability) || measuredReliability < 0 || measuredReliability > 1) {
    throw new Error("measuredReliability must be in [0,1]");
  }
  const maturity = sampleCount / (sampleCount + SHRINK_SAMPLES);
  const raw = PRIOR_RELIABILITY + (measuredReliability - PRIOR_RELIABILITY) * maturity;
  return clamp(raw, PRIOR_RELIABILITY - MAX_RELIABILITY_DEVIATION, PRIOR_RELIABILITY + MAX_RELIABILITY_DEVIATION);
}

function isPitClear(row: LaggedReliabilityEvidence, decisionMs: number): boolean {
  const windowMs = parseTime(row.snapshotWindowEnd);
  const generatedMs = parseTime(row.snapshotGeneratedAt);
  return windowMs != null && generatedMs != null && windowMs <= decisionMs && generatedMs <= decisionMs;
}

function freshnessDecayFraction(rows: EvidenceFreshness[], decisionMs: number): { fraction: number; contaminated: boolean; reasons: string[] } {
  if (!rows.length) return { fraction: 0.25, contaminated: false, reasons: ["source-observation freshness unavailable; conservative decay prior applied"] };
  const fractions: number[] = [];
  const reasons: string[] = [];
  let contaminated = false;
  for (const row of rows) {
    const observedMs = parseTime(row.observedAt);
    const budget = freshnessBudgetSeconds(row.horizon);
    if (observedMs == null || budget == null) {
      fractions.push(1);
      reasons.push(`unknown freshness for ${row.group}`);
      continue;
    }
    if (observedMs > decisionMs + 5_000) {
      contaminated = true;
      reasons.push(`future observation detected for ${row.group}`);
      continue;
    }
    const ageSeconds = Math.max(0, (decisionMs - observedMs) / 1000);
    fractions.push(clamp(ageSeconds / budget, 0, 1));
  }
  const fraction = fractions.length ? fractions.reduce((a, b) => a + b, 0) / fractions.length : 1;
  return { fraction, contaminated, reasons };
}

function weightedMean(values: Array<{ value: number; weight: number }>): number | null {
  const total = values.reduce((sum, row) => sum + row.weight, 0);
  return total > 0 ? values.reduce((sum, row) => sum + row.value * row.weight, 0) / total : null;
}

function weightedMeanAbsoluteDeviation(values: Array<{ value: number; weight: number }>, mean: number): number {
  const total = values.reduce((sum, row) => sum + row.weight, 0) || 1;
  return values.reduce((sum, row) => sum + Math.abs(row.value - mean) * row.weight, 0) / total;
}

export function estimateExpectedNetEdge(input: ExpectedEdgeInput): ExpectedEdgeDecomposition {
  const reasons: string[] = [];
  const decisionMs = parseTime(input.decisionObservedAt);
  const margin = Number.isFinite(input.minimumNetMarginBps) ? Math.max(0, Number(input.minimumNetMarginBps)) : 2;
  if (decisionMs == null) {
    return {
      version: EVOLUTION_ALPHA_INTELLIGENCE_VERSION,
      expectedGrossMoveBps: null,
      estimatedRoundTripCostBps: null,
      uncertaintyPenaltyBps: null,
      eventDecayPenaltyBps: null,
      expectedNetEdgeBps: null,
      minimumNetMarginBps: margin,
      eligible: false,
      recommendation: "CONTAMINATED_EVIDENCE",
      matureGroupCount: 0,
      groupContributions: [],
      reliabilityWeights: {},
      pitClear: false,
      reasons: ["decision timestamp invalid"],
    };
  }

  const cost = finite(input.roundTripCostBps);
  if (cost == null || cost < 0) {
    return {
      version: EVOLUTION_ALPHA_INTELLIGENCE_VERSION,
      expectedGrossMoveBps: null,
      estimatedRoundTripCostBps: null,
      uncertaintyPenaltyBps: null,
      eventDecayPenaltyBps: null,
      expectedNetEdgeBps: null,
      minimumNetMarginBps: margin,
      eligible: false,
      recommendation: "COST_UNAVAILABLE",
      matureGroupCount: 0,
      groupContributions: [],
      reliabilityWeights: {},
      pitClear: true,
      reasons: ["decision-time round-trip cost is unavailable"],
    };
  }

  const unique = new Map<string, LaggedReliabilityEvidence>();
  let pitClear = true;
  for (const row of input.reliability) {
    if (!row.group || !Number.isFinite(row.sampleCount) || row.sampleCount < 0 || !Number.isFinite(row.bayesianHitRate) || row.bayesianHitRate < 0 || row.bayesianHitRate > 1) continue;
    if (!Number.isFinite(row.avgSignedBps) || !Number.isFinite(row.avgCostAdjustedSignedBps)) continue;
    if (!isPitClear(row, decisionMs)) {
      pitClear = false;
      reasons.push(`post-decision reliability rejected:${row.group}`);
      continue;
    }
    const previous = unique.get(row.group);
    if (!previous || row.sampleCount > previous.sampleCount) unique.set(row.group, row);
  }

  const contributions: GroupEdgeContribution[] = [];
  for (const row of unique.values()) {
    const maturity = clamp(row.sampleCount / (row.sampleCount + SHRINK_SAMPLES), 0, 1);
    const bounded = boundedProspectiveReliability(row.sampleCount, row.bayesianHitRate);
    const reliabilityScale = clamp(bounded / PRIOR_RELIABILITY, 0.7, 1.3);
    const gross = clamp(row.avgSignedBps, -GROSS_CAP_BPS, GROSS_CAP_BPS);
    const weight = maturity * reliabilityScale;
    contributions.push({
      group: row.group,
      samples: row.sampleCount,
      measuredReliability: row.bayesianHitRate,
      boundedReliability: bounded,
      maturity,
      historicalGrossSignedBps: gross,
      historicalAfterCostSignedBps: clamp(row.avgCostAdjustedSignedBps, -GROSS_CAP_BPS, GROSS_CAP_BPS),
      contributionWeight: weight,
      contributionGrossBps: gross * weight,
      outcomeHorizonSeconds: row.outcomeHorizonSeconds,
      snapshotWindowEnd: row.snapshotWindowEnd,
      snapshotGeneratedAt: row.snapshotGeneratedAt,
    });
  }

  const mature = contributions.filter((row) => row.samples >= MIN_MATURE_SAMPLES);
  if (mature.length < MIN_MATURE_GROUPS) {
    reasons.push(`only ${mature.length} mature independent groups; ${MIN_MATURE_GROUPS} required`);
    return {
      version: EVOLUTION_ALPHA_INTELLIGENCE_VERSION,
      expectedGrossMoveBps: null,
      estimatedRoundTripCostBps: cost,
      uncertaintyPenaltyBps: null,
      eventDecayPenaltyBps: null,
      expectedNetEdgeBps: null,
      minimumNetMarginBps: margin,
      eligible: false,
      recommendation: pitClear ? "INSUFFICIENT_LAGGED_EVIDENCE" : "CONTAMINATED_EVIDENCE",
      matureGroupCount: mature.length,
      groupContributions: contributions,
      reliabilityWeights: Object.fromEntries(contributions.map((row) => [row.group, row.boundedReliability])),
      pitClear,
      reasons,
    };
  }

  const weighted = mature.map((row) => ({ value: row.historicalGrossSignedBps, weight: row.contributionWeight }));
  const gross = weightedMean(weighted)!;
  const dispersion = weightedMeanAbsoluteDeviation(weighted, gross);
  const avgMaturity = mature.reduce((sum, row) => sum + row.maturity, 0) / mature.length;
  const uncertainty = UNCERTAINTY_FLOOR_BPS + 0.5 * dispersion + (1 - avgMaturity) * IMMATURITY_PENALTY_BPS;
  const freshness = freshnessDecayFraction(input.freshness, decisionMs);
  if (freshness.contaminated) pitClear = false;
  reasons.push(...freshness.reasons);
  const decay = Math.max(0, gross) * 0.25 * freshness.fraction;
  const net = gross - cost - uncertainty - decay;
  const eligible = pitClear && net > margin;
  if (gross <= 0) reasons.push("lagged prospective evidence implies non-positive gross directional edge");
  if (net <= margin) reasons.push(`expected net edge ${net.toFixed(2)} bps does not clear ${margin.toFixed(2)} bps margin`);
  if (eligible) reasons.push(`expected net edge clears margin with ${mature.length} mature independent groups`);

  return {
    version: EVOLUTION_ALPHA_INTELLIGENCE_VERSION,
    expectedGrossMoveBps: gross,
    estimatedRoundTripCostBps: cost,
    uncertaintyPenaltyBps: uncertainty,
    eventDecayPenaltyBps: decay,
    expectedNetEdgeBps: net,
    minimumNetMarginBps: margin,
    eligible,
    recommendation: !pitClear ? "CONTAMINATED_EVIDENCE" : eligible ? "ALLOW_EDGE" : "DOWNGRADE_TO_WAIT",
    matureGroupCount: mature.length,
    groupContributions: contributions,
    reliabilityWeights: Object.fromEntries(contributions.map((row) => [row.group, row.boundedReliability])),
    pitClear,
    reasons,
  };
}

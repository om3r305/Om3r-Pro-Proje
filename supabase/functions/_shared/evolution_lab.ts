import type { ExperimentMetrics } from "./evolution_research.ts";

export const EVOLUTION_LAB_VERSION = "brian.evolution-lab.v1";

export interface ProspectiveOutcomePoint {
  decisionId: string;
  observedAt: string;
  directionAdjustedReturn: number;
  estimatedRoundTripCostBps: number;
  classification: string;
}

export interface ChallengerDecisionLabel {
  decisionId: string;
  challengerAction: string;
}

export interface GateMeasurement {
  control: ExperimentMetrics;
  challenger: ExperimentMetrics;
  lineage: {
    controlSamples: number;
    challengerSamples: number;
    allowedDecisionIds: string[];
    sourceDecisionIds: string[];
    stabilityWindows: number;
    allowedLabel: string;
  };
}

export type ActionGateMeasurement = GateMeasurement;

function finite(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function avg(values: number[]): number | null {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

function sixHourWindow(iso: string): string | null {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return null;
  const d = new Date(t);
  const bucket = Math.floor(d.getUTCHours() / 6) * 6;
  return `${d.toISOString().slice(0, 10)}T${String(bucket).padStart(2, "0")}`;
}

function maxDrawdownPctFromNetBps(netBps: number[]): number | null {
  if (!netBps.length) return null;
  let equity = 100;
  let peak = equity;
  let maxDrawdown = 0;
  for (const bps of netBps) {
    const boundedReturn = Math.max(-0.95, bps / 10_000);
    equity *= 1 + boundedReturn;
    peak = Math.max(peak, equity);
    if (peak > 0) maxDrawdown = Math.max(maxDrawdown, (peak - equity) / peak * 100);
  }
  return maxDrawdown;
}

function stability(points: ProspectiveOutcomePoint[]): { score: number | null; regimes: number } {
  const buckets = new Map<string, number[]>();
  for (const point of points) {
    const key = sixHourWindow(point.observedAt);
    if (!key) continue;
    const net = point.directionAdjustedReturn * 10_000 - point.estimatedRoundTripCostBps;
    if (!Number.isFinite(net)) continue;
    const values = buckets.get(key) ?? [];
    values.push(net);
    buckets.set(key, values);
  }
  const mature = [...buckets.values()].filter((values) => values.length >= 5);
  if (!mature.length) return { score: null, regimes: buckets.size };
  const positive = mature.filter((values) => Number(avg(values)) > 0).length;
  return { score: positive / mature.length, regimes: mature.length };
}

export function measureOutcomeSet(points: ProspectiveOutcomePoint[], complexityDelta: number): ExperimentMetrics {
  const valid = points.filter((point) => {
    return Boolean(point.decisionId) && Number.isFinite(Date.parse(point.observedAt)) &&
      finite(point.directionAdjustedReturn) != null && finite(point.estimatedRoundTripCostBps) != null &&
      point.estimatedRoundTripCostBps >= 0;
  });
  const gross = valid.map((point) => point.directionAdjustedReturn * 10_000);
  const costs = valid.map((point) => point.estimatedRoundTripCostBps);
  const net = valid.map((point) => point.directionAdjustedReturn * 10_000 - point.estimatedRoundTripCostBps);
  const favorable = valid.filter((point) => point.classification === "ACTION_FAVORABLE_AFTER_COST").length;
  const stable = stability(valid);
  const distinctDays = new Set(valid.map((point) => point.observedAt.slice(0, 10))).size;
  return {
    samples: valid.length,
    regimes: Math.max(stable.regimes, distinctDays),
    netEdgeBps: avg(net),
    grossEdgeBps: avg(gross),
    maxDrawdownPct: maxDrawdownPctFromNetBps(net),
    favorableAfterCostRate: valid.length ? favorable / valid.length : null,
    turnover: distinctDays ? valid.length / distinctDays : valid.length || null,
    costBps: avg(costs),
    leakageDetected: false,
    dataQualityOk: valid.length >= 30 && valid.length === points.length,
    stabilityScore: stable.score,
    complexityDelta,
  };
}

export function measureGateExperiment(
  outcomes: ProspectiveOutcomePoint[],
  labels: ChallengerDecisionLabel[],
  allowedLabel: string,
  complexityDelta = 1,
): GateMeasurement {
  const labelMap = new Map(labels.map((label) => [label.decisionId, label.challengerAction]));
  const control = outcomes.filter((point) => labelMap.has(point.decisionId));
  const allowed = control.filter((point) => labelMap.get(point.decisionId) === allowedLabel);
  const stabilityWindows = new Set(control.map((point) => sixHourWindow(point.observedAt)).filter(Boolean)).size;
  return {
    control: measureOutcomeSet(control, 0),
    challenger: measureOutcomeSet(allowed, complexityDelta),
    lineage: {
      controlSamples: control.length,
      challengerSamples: allowed.length,
      allowedDecisionIds: allowed.map((point) => point.decisionId),
      sourceDecisionIds: control.map((point) => point.decisionId),
      stabilityWindows,
      allowedLabel,
    },
  };
}

export function measureActionGateExperiment(
  outcomes: ProspectiveOutcomePoint[],
  labels: ChallengerDecisionLabel[],
): ActionGateMeasurement {
  return measureGateExperiment(outcomes, labels, "ALLOW_ACTION", 1);
}

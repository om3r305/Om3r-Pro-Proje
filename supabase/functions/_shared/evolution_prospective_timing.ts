export interface ProspectiveTimingInput {
  experimentStartedAt: string;
  decisionObservedAt: string;
  labelEvaluatedAt: string;
  outcomeResolvedAt: string;
  measuredAt: string;
  maxLabelLatencySeconds?: number | null;
}

export interface ProspectiveTimingAssessment {
  clean: boolean;
  reasons: string[];
  labelLatencySeconds: number | null;
}

const CLOCK_SKEW_SECONDS = 5;

function time(value: string): number | null {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function assessProspectiveTiming(input: ProspectiveTimingInput): ProspectiveTimingAssessment {
  const experiment = time(input.experimentStartedAt);
  const decision = time(input.decisionObservedAt);
  const evaluated = time(input.labelEvaluatedAt);
  const resolved = time(input.outcomeResolvedAt);
  const measured = time(input.measuredAt);
  const reasons: string[] = [];
  if (experiment == null) reasons.push("experiment start timestamp invalid");
  if (decision == null) reasons.push("decision timestamp invalid");
  if (evaluated == null) reasons.push("challenger evaluation timestamp invalid");
  if (resolved == null) reasons.push("outcome resolution timestamp invalid");
  if (measured == null) reasons.push("measurement timestamp invalid");

  const labelLatencySeconds = decision == null || evaluated == null ? null : (evaluated - decision) / 1000;
  if (experiment != null && decision != null && decision < experiment) reasons.push("decision predates experiment start");
  if (decision != null && evaluated != null && evaluated < decision - CLOCK_SKEW_SECONDS * 1000) reasons.push("challenger label predates decision");
  if (evaluated != null && resolved != null && evaluated > resolved) reasons.push("challenger label was produced after outcome resolution");
  if (resolved != null && measured != null && resolved > measured + CLOCK_SKEW_SECONDS * 1000) reasons.push("outcome resolves after measurement time");
  const maxLatency = input.maxLabelLatencySeconds;
  if (maxLatency != null && Number.isFinite(maxLatency) && maxLatency >= 0 && labelLatencySeconds != null && labelLatencySeconds > maxLatency) {
    reasons.push("challenger label exceeded prospective latency budget");
  }

  return { clean: reasons.length === 0, reasons, labelLatencySeconds };
}

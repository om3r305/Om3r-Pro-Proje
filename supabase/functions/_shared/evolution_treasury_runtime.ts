export const TREASURY_MARK_MAX_AGE_SECONDS = 3 * 60;
export const TREASURY_EDGE_MAX_AGE_SECONDS = 5 * 60;
export const TREASURY_EDGE_EVALUATION_MAX_LATENCY_SECONDS = 3 * 60;
const FUTURE_SKEW_SECONDS = 5;

export interface TreasuryRuntimeEvidenceInput {
  nowIso: string;
  edgeObservedAt: string;
  edgeEvaluatedAt: string;
  markObservedAt: string;
}

export interface TreasuryRuntimeEvidenceAssessment {
  actionable: boolean;
  reasons: string[];
  edgeAgeSeconds: number | null;
  evaluationLatencySeconds: number | null;
  markAgeSeconds: number | null;
}

function time(value: string): number | null {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function assessTreasuryRuntimeEvidence(
  input: TreasuryRuntimeEvidenceInput,
): TreasuryRuntimeEvidenceAssessment {
  const now = time(input.nowIso);
  const edge = time(input.edgeObservedAt);
  const evaluated = time(input.edgeEvaluatedAt);
  const mark = time(input.markObservedAt);
  const reasons: string[] = [];
  if (now == null) throw new Error("invalid Treasury runtime nowIso");

  const edgeAgeSeconds = edge == null ? null : (now - edge) / 1000;
  const evaluationLatencySeconds = edge == null || evaluated == null ? null : (evaluated - edge) / 1000;
  const markAgeSeconds = mark == null ? null : (now - mark) / 1000;

  if (edge == null) reasons.push("edge timestamp invalid");
  else if (edgeAgeSeconds! < -FUTURE_SKEW_SECONDS) reasons.push("edge timestamp is in the future");
  else if (edgeAgeSeconds! > TREASURY_EDGE_MAX_AGE_SECONDS) reasons.push("edge is stale for Treasury execution");

  if (evaluated == null) reasons.push("edge evaluation timestamp invalid");
  else if (edge != null && evaluationLatencySeconds! < 0) reasons.push("edge was evaluated before its decision timestamp");
  else if (edge != null && evaluationLatencySeconds! > TREASURY_EDGE_EVALUATION_MAX_LATENCY_SECONDS) reasons.push("edge evaluation arrived too late for Treasury execution");
  if (evaluated != null && evaluated > now + FUTURE_SKEW_SECONDS * 1000) reasons.push("edge evaluation timestamp is in the future");

  if (mark == null) reasons.push("market mark timestamp invalid");
  else if (markAgeSeconds! < -FUTURE_SKEW_SECONDS) reasons.push("market mark timestamp is in the future");
  else if (markAgeSeconds! > TREASURY_MARK_MAX_AGE_SECONDS) reasons.push("market mark is stale");

  return { actionable: reasons.length === 0, reasons, edgeAgeSeconds, evaluationLatencySeconds, markAgeSeconds };
}

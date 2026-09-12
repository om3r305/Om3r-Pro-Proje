export interface OceanPreflightProbe {
  label: string;
  observedAt: string | null;
  maxAgeSeconds: number;
  status?: string | null;
  requireSuccess?: boolean;
}

export interface OceanPreflightAssessment {
  ready: boolean;
  reasons: string[];
  agesSeconds: Record<string, number | null>;
}

function timestamp(value: string | null): number | null {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

export function assessOceanPreflightFreshness(
  nowIso: string,
  probes: OceanPreflightProbe[],
  futureSkewSeconds = 5,
): OceanPreflightAssessment {
  const nowMs = timestamp(nowIso);
  if (nowMs == null) throw new Error("invalid Ocean preflight nowIso");
  const reasons: string[] = [];
  const agesSeconds: Record<string, number | null> = {};

  for (const probe of probes) {
    const maxAge = Number(probe.maxAgeSeconds);
    if (!probe.label || !Number.isFinite(maxAge) || maxAge <= 0) {
      reasons.push(`${probe.label || "unnamed probe"} has invalid freshness policy`);
      continue;
    }
    if (probe.requireSuccess && probe.status !== "SUCCESS") {
      reasons.push(`${probe.label} latest status is ${probe.status ?? "MISSING"}`);
    }
    const observedMs = timestamp(probe.observedAt);
    if (observedMs == null) {
      agesSeconds[probe.label] = null;
      reasons.push(`${probe.label} has no valid timestamp`);
      continue;
    }
    const ageSeconds = (nowMs - observedMs) / 1000;
    agesSeconds[probe.label] = ageSeconds;
    if (ageSeconds < -futureSkewSeconds) {
      reasons.push(`${probe.label} timestamp is in the future`);
      continue;
    }
    if (ageSeconds > maxAge) {
      reasons.push(`${probe.label} is stale (${Math.round(ageSeconds)}s > ${Math.round(maxAge)}s)`);
    }
  }

  return { ready: reasons.length === 0, reasons, agesSeconds };
}

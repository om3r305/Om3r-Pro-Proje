// Freshness of a public market mark, decided from its CURRENT age, never from a latency value that
// was stored when the row was written (that value does not grow while the row goes stale).
// A mark is "live" only if it is at most LIVE_MAX_AGE_SECONDS old now; the UI uses the same rule.

export const LIVE_MAX_AGE_SECONDS = 120;

export function markFreshness(observedAt: unknown, nowMs: number, sessionOk = true): { age_seconds: number | null; live: boolean } {
  const t = Date.parse(String(observedAt ?? ""));
  if (!Number.isFinite(t)) return { age_seconds: null, live: false };
  const age = Math.max(0, (nowMs - t) / 1000);
  // A timestamp more than 30 s in the future is a clock/feed error, not a fresh price.
  if (t - nowMs > 30_000) return { age_seconds: null, live: false };
  return { age_seconds: Math.round(age * 10) / 10, live: sessionOk && age <= LIVE_MAX_AGE_SECONDS };
}

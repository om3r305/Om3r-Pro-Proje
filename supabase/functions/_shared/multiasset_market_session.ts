// Provider timestamps and session boundaries are authoritative; polling is not a price update.
export function assessMarketSession(providerTime: string, regularStart: unknown, regularEnd: unknown, nowMs = Date.now()) {
  const providerMs = Date.parse(providerTime);
  const start = typeof regularStart === "number" ? regularStart * 1000 : NaN;
  const end = typeof regularEnd === "number" ? regularEnd * 1000 : NaN;
  const age = (nowMs - providerMs) / 1000;
  const latency = Number.isFinite(age) ? Math.max(0, age) : null;
  if (!Number.isFinite(nowMs) || !Number.isFinite(providerMs) || age < -5) {
    return { sessionState: "UNKNOWN", eligible: false, latency, reason: "INVALID_PROVIDER_TIME" };
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || start >= end) {
    return { sessionState: "UNKNOWN", eligible: false, latency, reason: "SESSION_UNVERIFIED" };
  }
  if (nowMs < start || nowMs >= end) {
    return { sessionState: "CLOSED", eligible: false, latency, reason: "OUTSIDE_REGULAR_SESSION" };
  }
  if (age > 900 || providerMs < start || providerMs >= end) {
    return { sessionState: "STALE_OR_CLOSED", eligible: false, latency, reason: "STALE_PROVIDER_PRICE" };
  }
  return { sessionState: "REGULAR", eligible: true, latency, reason: "FRESH_REGULAR_SESSION" };
}

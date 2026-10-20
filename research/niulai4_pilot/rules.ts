// NIULAI4 observation pilot -- pre-registered rules (frozen).
//
// Source of truth: docs/reviews/BRIAN_NIULAI4_OBSERVATION_PLAN_20261020.md. Changing ANY value here
// requires a new RULES_VERSION; results from different versions must never be pooled. The pilot
// is LOCAL, READ-ONLY and SHADOW-ONLY: it never places orders, never uses a key, never writes to
// CORE, and does not touch brian_prediction_control.wallet_tracking_enabled.

export const RULES_VERSION = "niulai4-pilot.v1";

export const RULES = Object.freeze({
  wallet: "0x44832d0d2ec11187c1e77d786feb15f6a50254c6",
  displayName: "NIULAI4",
  asset: "BTC" as const,
  windowSeconds: 300, // 5-minute Up/Down only
  slugPrefix: "btc-updown-5m-",
  activeHoursUtc: { start: 12, end: 24 }, // [start, end)
  pilotDays: 7,
  activityPollMs: 2000, // 5 requests / 10 s against a 200 / 10 s limit
  spotCrossCheckMs: 5000,
  sampleLastSecondsOfWindow: 120,
  bookSampleLevels: 5,
  followSizeShares: 10, // >= orderMinSize (5)
  delaysSeconds: Object.freeze([0, 1, 5, 15, 30, 60]),
  priceToleranceAboveBestAsk: 0.02,
  feeRateFallback: null as number | null, // fee must come from the market's feeSchedule, never assumed
  bankrollUsdc: 100,
  primary: Object.freeze({ delaySeconds: 5, baseline: "B1_SPOT_DIRECTION" as const }),
  randomSeed: 20261020,
  diskCapBytes: 500 * 1024 * 1024,
  maxClockSkewMs: 50,
  wsMaxReconnectsPer10Min: 5,
  maxConsecutiveRateLimited: 5,
});

// Hosts the pilot may contact. Anything else is refused before a request is made.
export const PILOT_ALLOWED_HOSTS = Object.freeze([
  "gamma-api.polymarket.com",
  "data-api.polymarket.com",
  "clob.polymarket.com",
  "ws-subscriptions-clob.polymarket.com",
  "api.binance.com",
  "stream.binance.com",
  "data-stream.binance.vision",
  "api.exchange.coinbase.com",
]);

export function assertPilotHost(raw: string): URL {
  const url = new URL(raw);
  if (url.protocol !== "https:" && url.protocol !== "wss:") throw new Error(`pilot url must be https/wss: ${url.protocol}`);
  if (url.username || url.password) throw new Error("pilot url must not carry credentials");
  if (!PILOT_ALLOWED_HOSTS.includes(url.hostname)) throw new Error(`pilot host not allow-listed: ${url.hostname}`);
  return url;
}

// A stable fingerprint of the frozen rules, written into every output file header so a report can
// prove which rules produced it.
export async function rulesFingerprint(): Promise<string> {
  const text = JSON.stringify({ RULES_VERSION, RULES });
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

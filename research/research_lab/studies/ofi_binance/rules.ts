// OFI study (arXiv:1011.6402) — frozen collection rules. Pre-registration:
// research/research_lab/ledger/arxiv-1011.6402.json. Changing ANY value needs a new RULES_VERSION.
// Collection is LOCAL, READ-ONLY and SHADOW-ONLY: no orders, no keys, public data only.

export const RULES_VERSION = "ofi-binance.v1";

export const RULES = Object.freeze({
  symbols: Object.freeze(["BTCUSDT", "ETHUSDT"]),
  // Binance public market-data endpoint (key-less, reachable from Germany).
  wsBase: "wss://data-stream.binance.vision/stream",
  restBase: "https://data-api.binance.vision",
  bucketMs: 1000, // one aggregate per symbol per second
  collectionDays: 14,
  maxSilenceMs: 5000, // no message for longer than this -> the seconds are a gap, never filled
  diskCapBytes: 500 * 1024 * 1024,
  maxClockUncertaintyMs: 50,
  healthEveryMs: 60_000,
  maxReconnectsPer10Min: 10,
});

export const ALLOWED_HOSTS = Object.freeze(["data-stream.binance.vision", "data-api.binance.vision"]);

export function assertAllowed(raw: string): URL {
  const u = new URL(raw);
  if (u.protocol !== "https:" && u.protocol !== "wss:") throw new Error(`must be https/wss: ${u.protocol}`);
  if (u.username || u.password) throw new Error("no credentials in URLs");
  if (!ALLOWED_HOSTS.includes(u.hostname)) throw new Error(`host not allow-listed: ${u.hostname}`);
  return u;
}

export async function rulesFingerprint(): Promise<string> {
  const text = JSON.stringify({ RULES_VERSION, RULES });
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  return [...d].map((b) => b.toString(16).padStart(2, "0")).join("");
}

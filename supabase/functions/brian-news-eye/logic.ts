// Pure, side-effect-free parsing/scoring logic for brian-news-eye, split out of index.ts so it
// can be exercised directly by logic.test.ts without a live network/Supabase dependency.

export const POS = [
  "approval", "approved", "partnership", "launch", "record high", "inflow", "rally", "surge",
  "upgrade", "adoption", "buyback",
];
export const NEG = [
  "hack", "hacked", "exploit", "ban", "lawsuit", "outflow", "crash", "fraud", "liquidation",
  "breach", "attack", "downgrade",
];

export function clip(v: number): number {
  return Math.max(0, Math.min(1, v));
}

export function sign(v: number): number {
  return v > 0 ? 1 : v < 0 ? -1 : 0;
}

/** Parses GDELT's compact "YYYYMMDDTHHMMSSZ" seendate into an ISO-8601 string, or null if unparseable. */
export function parseSeen(v?: string): string | null {
  if (!v) return null;
  const m = v.match(/^(\d{4})(\d{2})(\d{2})T?(\d{2})(\d{2})(\d{2})Z?$/);
  if (!m) return null;
  const d = new Date(`${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}Z`);
  return Number.isFinite(d.getTime()) ? d.toISOString() : null;
}

/** Maps a headline to the asset/domain bucket it most likely concerns. First match wins. */
export function asset(title: string): string {
  const t = title.toLowerCase();
  if (/bitcoin|\bbtc\b/.test(t)) return "BTCUSDT";
  if (/ethereum|\beth\b/.test(t)) return "ETHUSDT";
  if (/solana|\bsol\b/.test(t)) return "SOLUSDT";
  if (/\bxrp\b|ripple/.test(t)) return "XRPUSDT";
  if (/binance coin|\bbnb\b/.test(t)) return "BNBUSDT";
  if (/gold/.test(t)) return "GOLD";
  if (/silver/.test(t)) return "SILVER";
  if (/\boil\b|brent|wti/.test(t)) return "OIL";
  if (/federal reserve|\bfed\b|ecb|interest rate|inflation/.test(t)) return "MACRO";
  return "CRYPTO_MARKET";
}

/** Bag-of-words directional pressure from a headline: +1 positive-only, -1 negative-only, 0 mixed/neutral. */
export function pressure(title: string): number {
  const t = title.toLowerCase();
  const p = POS.filter((w) => t.includes(w)).length;
  const n = NEG.filter((w) => t.includes(w)).length;
  if (p > 0 && n === 0) return 1;
  if (n > 0 && p === 0) return -1;
  return 0;
}

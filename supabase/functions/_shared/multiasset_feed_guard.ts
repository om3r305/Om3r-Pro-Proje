// Commodity feed guard for the multiasset opportunity engine (pure, no I/O).
//
// Found 2026-10-23: the primary feed (Yahoo continuous futures, e.g. BZ=F / NG=F) silently switched
// to the next contract days before expiry. One 10-minute step (+13.7% NATGAS, -4.3% BRENT) then
// looked like a strong "market reaction" for the next hour, and the engine emitted signals on it.
//
// Second source: TradingView's public scanner, front (1!) and next (2!) contract of each future.
//  * Which contract the primary price matches (FRONT / NEXT / UNMATCHED) is recorded per decision;
//    a change of the matched contract within the guard window is a FEED_CONTRACT_ROLL.
//  * A single step > maxStepLog between consecutive own observations is a PRICE_DISCONTINUITY
//    unless the second source independently shows a same-direction daily move of at least half the
//    step (a real spike). Without the second source a discontinuity fails closed.
// Only commodities are guarded; everything else passes unchanged.

export const FEED_GUARD_VERSION = "brian.multiasset-feed-guard.v1";
export const FEED_GUARD_RULES = Object.freeze({
  windowMinutes: 75, // return_1h straddles a step for one hour after it
  maxStepLog: 0.04,
  maxStepGapMinutes: 20,
  maxMatchLog: 0.03, // the second source is ~10 min delayed
  minContractSpreadLog: 0.02, // closer contracts (e.g. gold) cannot be told apart; their roll steps are small anyway
});

/** Second-source tickers per guarded asset (front month = `${root}1!`, next = `${root}2!`). */
export const FEED_GUARD_ROOTS: Readonly<Record<string, string>> = Object.freeze({
  "commodity:NATGAS": "NYMEX:NG",
  "commodity:BRENT": "ICEEUR:BRN",
  "commodity:WTI": "NYMEX:CL",
  "commodity:GOLD": "COMEX:GC",
  "commodity:SILVER": "COMEX:SI",
});

export type FeedContract = "FRONT" | "NEXT" | "UNMATCHED" | "UNKNOWN";
export interface SecondSourceQuote { front: number | null; next: number | null; frontDailyChangePct: number | null }
export interface OwnObservation { observedAt: string; price: number; feedContract: string | null }
export interface FeedGuardVerdict {
  guarded: boolean;
  feedContract: FeedContract;
  veto: "FEED_CONTRACT_ROLL" | "PRICE_DISCONTINUITY" | null;
  maxStepLog: number | null;
  reason: string;
}

const logRatio = (a: number, b: number) => Math.log(a / b);
const valid = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v) && v > 0;

export function matchContract(price: number, quote: SecondSourceQuote | null): FeedContract {
  if (!quote || !valid(price) || !valid(quote.front) || !valid(quote.next)) return "UNKNOWN";
  const spread = Math.abs(logRatio(quote.next, quote.front));
  if (spread < FEED_GUARD_RULES.minContractSpreadLog) return "UNKNOWN";
  const df = Math.abs(logRatio(price, quote.front));
  const dn = Math.abs(logRatio(price, quote.next));
  // Nearest contract, and clearly nearer than the other one (inside half the spread).
  const best = Math.min(df, dn);
  if (best > FEED_GUARD_RULES.maxMatchLog || best > spread / 2) return "UNMATCHED";
  return df <= dn ? "FRONT" : "NEXT";
}

export function feedGuardVerdict(input: {
  assetId: string;
  nowIso: string;
  price: number;
  history: OwnObservation[]; // own earlier observations of this asset (any order)
  quote: SecondSourceQuote | null; // null = second source unavailable
}): FeedGuardVerdict {
  if (!(input.assetId in FEED_GUARD_ROOTS)) {
    return { guarded: false, feedContract: "UNKNOWN", veto: null, maxStepLog: null, reason: "NOT_GUARDED" };
  }
  const now = Date.parse(input.nowIso);
  const since = now - FEED_GUARD_RULES.windowMinutes * 60_000;
  const feedContract = matchContract(input.price, input.quote);
  const series = input.history
    .map((h) => ({ ...h, t: Date.parse(h.observedAt) }))
    .filter((h) => Number.isFinite(h.t) && h.t >= since && h.t < now && valid(h.price))
    .sort((a, b) => a.t - b.t);
  const points = [...series, { observedAt: input.nowIso, price: input.price, feedContract, t: now }];

  let maxStep = 0;
  let signedStep = 0;
  for (let i = 1; i < points.length; i++) {
    if (points[i].t - points[i - 1].t > FEED_GUARD_RULES.maxStepGapMinutes * 60_000) continue;
    const step = logRatio(points[i].price, points[i - 1].price);
    if (Math.abs(step) > maxStep) {
      maxStep = Math.abs(step);
      signedStep = step;
    }
  }

  const known = (c: string | null) => c === "FRONT" || c === "NEXT";
  const contracts = new Set(points.map((p) => p.feedContract).filter(known));
  if (contracts.size > 1) {
    return { guarded: true, feedContract, veto: "FEED_CONTRACT_ROLL", maxStepLog: maxStep, reason: "primary feed changed contract inside the guard window" };
  }
  if (maxStep > FEED_GUARD_RULES.maxStepLog) {
    const daily = input.quote?.frontDailyChangePct;
    const confirmed = typeof daily === "number" && Number.isFinite(daily) &&
      Math.sign(daily) === Math.sign(signedStep) && Math.abs(daily) / 100 >= maxStep / 2;
    if (!confirmed) {
      return {
        guarded: true, feedContract, veto: "PRICE_DISCONTINUITY", maxStepLog: maxStep,
        reason: input.quote ? "single-step jump not confirmed by the second source" : "single-step jump and second source unavailable",
      };
    }
    return { guarded: true, feedContract, veto: null, maxStepLog: maxStep, reason: "jump confirmed by the second source" };
  }
  return { guarded: true, feedContract, veto: null, maxStepLog: maxStep, reason: "OK" };
}

/** Parse a TradingView scanner response into quotes per guarded asset. */
export function parseScanner(payload: unknown): Map<string, SecondSourceQuote> {
  const byTicker = new Map<string, { close: number | null; change: number | null }>();
  const rows = (payload as { data?: unknown[] } | null)?.data;
  for (const row of Array.isArray(rows) ? rows : []) {
    const r = row as { s?: unknown; d?: unknown[] };
    if (typeof r.s !== "string" || !Array.isArray(r.d)) continue;
    const close = Number(r.d[0]);
    const change = Number(r.d[1]);
    byTicker.set(r.s, { close: Number.isFinite(close) ? close : null, change: Number.isFinite(change) ? change : null });
  }
  const out = new Map<string, SecondSourceQuote>();
  for (const [assetId, root] of Object.entries(FEED_GUARD_ROOTS)) {
    const f = byTicker.get(`${root}1!`);
    const nx = byTicker.get(`${root}2!`);
    if (!f && !nx) continue;
    out.set(assetId, { front: f?.close ?? null, next: nx?.close ?? null, frontDailyChangePct: f?.change ?? null });
  }
  return out;
}

export function scannerTickers(): string[] {
  return Object.values(FEED_GUARD_ROOTS).flatMap((root) => [`${root}1!`, `${root}2!`]);
}

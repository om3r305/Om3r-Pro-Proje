// Prospective shadow test of the Polymarket political-calibration result (arXiv:2602.19520 C-01,
// run-4: slope 1.162 [1.114, 1.216]). Pre-registered in
// research/research_lab/studies/pm_calibration_shadow/PREREGISTRATION.md BEFORE any data is
// collected. Every number below is frozen; changing one changes RULES_FINGERPRINT and starts a
// new study. Nothing here trades: entries are hypothetical, priced at the observed best ask.

export const PM_CALIBRATION_RULES_VERSION = "pm-calibration-shadow.v1";

export const PM_CALIBRATION_RULES = Object.freeze({
  // Calibration map fitted on 16,062 political markets resolving 2026-01-01 .. 2026-10-01
  // (results/pm-political-slope-oos-v1/run-4-weekly, code 4c62754): q = sigmoid(a + b * logit(p)).
  slope: 1.1624733790827435,
  intercept: 0.1356427546398957,
  priceClip: [0.01, 0.99] as const,
  // Entry: first capture inside [end - 31 h, end - 29 h] (run-4 measured the price 30 h before).
  entryWindowHours: [29, 31] as const,
  // Market quality gates on the YES book at capture time.
  maxYesSpread: 0.10,
  minAskShares: 5,
  askRange: [0.02, 0.98] as const,
  // Hypothetical taker entry at the best ask; fee from the market's own feeSchedule, never assumed.
  minEdge: 0.02,
  // Decision (see PREREGISTRATION.md): event-clustered bootstrap of mean return per $1 staked.
  minTradesForDecision: 100,
  targetTrades: 300,
  decisionDeadline: "2027-01-31T00:00:00Z",
  bootstrapResamples: 2000,
  bootstrapSeed: 20261021,
});

export async function pmCalibrationRulesFingerprint(): Promise<string> {
  const text = JSON.stringify({ PM_CALIBRATION_RULES_VERSION, PM_CALIBRATION_RULES });
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text)));
  return [...d].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function calibratedYes(p: number): number {
  const [lo, hi] = PM_CALIBRATION_RULES.priceClip;
  const c = Math.min(Math.max(p, lo), hi);
  const z = Math.max(-30, Math.min(30, PM_CALIBRATION_RULES.intercept + PM_CALIBRATION_RULES.slope * Math.log(c / (1 - c))));
  return 1 / (1 + Math.exp(-z));
}

export interface BookTop { bid: number | null; ask: number | null; askShares: number | null }

// Polymarket taker fee per share: feeRate x p x (1 - p) (see _shared/prediction_market.ts).
const feePerShare = (price: number, feeRate: number) => feeRate * price * (1 - price);

export type EntryDecision =
  | { action: "SKIP"; reason: string; mid: number | null; q: number | null }
  | { action: "BUY_YES" | "BUY_NO"; mid: number; q: number; price: number; fee: number; edge: number; yesEdge: number; noEdge: number };

export function decideEntry(yes: BookTop, no: BookTop, feeRate: number | null): EntryDecision {
  const R = PM_CALIBRATION_RULES;
  if (feeRate === null || !Number.isFinite(feeRate) || feeRate < 0 || feeRate >= 1) return { action: "SKIP", reason: "FEE_UNKNOWN", mid: null, q: null };
  if (yes.bid === null || yes.ask === null || !(yes.ask > yes.bid)) return { action: "SKIP", reason: "YES_BOOK_ONE_SIDED", mid: null, q: null };
  const mid = (yes.bid + yes.ask) / 2, q = calibratedYes(mid);
  if (yes.ask - yes.bid > R.maxYesSpread) return { action: "SKIP", reason: "SPREAD_TOO_WIDE", mid, q };
  const usable = (b: BookTop) => b.ask !== null && b.ask >= R.askRange[0] && b.ask <= R.askRange[1] && (b.askShares ?? 0) >= R.minAskShares;
  const yesEdge = usable(yes) ? q - yes.ask! - feePerShare(yes.ask!, feeRate) : -Infinity;
  const noEdge = usable(no) ? (1 - q) - no.ask! - feePerShare(no.ask!, feeRate) : -Infinity;
  const best = yesEdge >= noEdge ? "BUY_YES" : "BUY_NO";
  const edge = Math.max(yesEdge, noEdge);
  if (!Number.isFinite(edge)) return { action: "SKIP", reason: "NO_USABLE_ASK", mid, q };
  if (edge < R.minEdge) return { action: "SKIP", reason: "EDGE_BELOW_MIN", mid, q };
  const price = best === "BUY_YES" ? yes.ask! : no.ask!;
  return { action: best, mid, q, price, fee: feePerShare(price, feeRate), edge, yesEdge, noEdge };
}

// Return per $1 staked on one share bought at `price` plus `fee`: payout 1 if the side won.
export function returnPerDollar(action: "BUY_YES" | "BUY_NO", price: number, fee: number, yesOutcome: 0 | 1): number {
  const won = action === "BUY_YES" ? yesOutcome === 1 : yesOutcome === 0;
  return (won ? 1 : 0) / (price + fee) - 1;
}

export function fromRawBook(raw: unknown): BookTop {
  const b = raw && typeof raw === "object" ? raw as { bids?: unknown; asks?: unknown } : {};
  const levels = (x: unknown) => (Array.isArray(x) ? x : []).flatMap((l) => {
    const p = Number((l as { price?: unknown })?.price), s = Number((l as { size?: unknown })?.size);
    return Number.isFinite(p) && Number.isFinite(s) && p > 0 && p < 1 && s > 0 ? [{ p, s }] : [];
  });
  const bids = levels(b.bids), asks = levels(b.asks);
  const bid = bids.length ? Math.max(...bids.map((l) => l.p)) : null;
  const ask = asks.length ? Math.min(...asks.map((l) => l.p)) : null;
  const askShares = ask === null ? null : asks.filter((l) => l.p === ask).reduce((n, l) => n + l.s, 0);
  return { bid, ask, askShares };
}

// Fee rate from the market's own schedule; null (unknown) unless the market states it.
export function marketFeeRate(m: { feesEnabled?: unknown; feeSchedule?: unknown }): number | null {
  if (m.feesEnabled === false) return 0;
  const rate = Number((m.feeSchedule as { rate?: unknown } | undefined)?.rate);
  return m.feesEnabled === true && Number.isFinite(rate) ? rate : null;
}

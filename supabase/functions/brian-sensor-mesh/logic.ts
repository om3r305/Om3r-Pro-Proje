// Pure, side-effect-free parsing/scoring logic for brian-sensor-mesh, split out of index.ts so it
// can be exercised directly by logic.test.ts without a live network/Supabase dependency.

export type Bar = { closeTime: number; open: number; high: number; low: number; close: number; volume: number };
export type Book = { bid: number; ask: number; mid: number; spreadBps: number };
export type RadarCandidate = {
  symbol: string; radar_score?: number; liquidity_score?: number; activity_score?: number; spread_bps?: number | null;
};

export const TEMPLATES = [
  { id: "structure-fast", family: "price_structure", group: "price_structure", ticket: 5.0 },
  { id: "momentum-fast", family: "price_structure", group: "price_momentum", ticket: 5.0 },
  { id: "mean-reversion-fast", family: "price_structure", group: "price_mean_reversion", ticket: 3.0 },
] as const;

export function finite(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function clip(value: number, low = 0, high = 1): number {
  return Math.max(low, Math.min(high, value));
}

export function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}

export function std(values: number[]): number {
  if (values.length <= 1) return 0;
  const m = mean(values);
  return Math.sqrt(mean(values.map((v) => (v - m) ** 2)));
}

/** Normalizes a raw Binance kline array into closed bars at or before cutoffMs, dropping malformed rows. */
export function parseBars(payload: unknown, cutoffMs: number): Bar[] {
  if (!Array.isArray(payload)) throw new Error("invalid kline payload");
  return payload
    .map((k) => {
      if (!Array.isArray(k) || k.length < 7) throw new Error("invalid kline row");
      return { closeTime: finite(k[6]), open: finite(k[1]), high: finite(k[2]), low: finite(k[3]), close: finite(k[4]), volume: finite(k[5]) };
    })
    .filter((bar) => bar.closeTime <= cutoffMs && bar.open > 0 && bar.high > 0 && bar.low > 0 && bar.close > 0);
}

export interface SignalResult {
  direction: number;
  strength: number;
  reason: string;
}

/** Closed-bar breakout/breakdown above/below the prior 8-12 bar range. */
export function signalStructure(bars: Bar[]): SignalResult {
  const current = bars.at(-1)!;
  const prior = bars.slice(-13, -1);
  if (prior.length < 8) return { direction: 0, strength: 0, reason: "insufficient structure context" };
  const high = Math.max(...prior.map((b) => b.high));
  const low = Math.min(...prior.map((b) => b.low));
  const ranges = prior.map((b) => Math.max(1e-12, b.high - b.low));
  const avgRange = mean(ranges);
  if (current.close > high * 1.0005) {
    return { direction: 1, strength: clip((current.close - high) / Math.max(avgRange * 2, 1e-12)), reason: "closed breakout above prior structure" };
  }
  if (current.close < low * 0.9995) {
    return { direction: -1, strength: clip((low - current.close) / Math.max(avgRange * 2, 1e-12)), reason: "closed breakdown below prior structure" };
  }
  return { direction: 0, strength: 0, reason: "no closed structure break" };
}

/** 4-bar log-return impulse against a volatility-adaptive threshold. */
export function signalMomentum(bars: Bar[]): SignalResult {
  const closes = bars.slice(-10).map((b) => b.close);
  if (closes.length < 6) return { direction: 0, strength: 0, reason: "insufficient momentum context" };
  const returns = closes.slice(1).map((v, i) => Math.log(v / closes[i]));
  const r4 = Math.log(closes.at(-1)! / closes.at(-5)!);
  const vol = Math.max(0.0004, std(returns));
  const threshold = Math.max(0.0015, vol * 1.5);
  if (Math.abs(r4) <= threshold) return { direction: 0, strength: 0, reason: "4-bar momentum below preregistered threshold" };
  return { direction: r4 > 0 ? 1 : -1, strength: clip(Math.abs(r4) / (threshold * 3)), reason: "4-bar momentum impulse" };
}

/** Z-score stretch of the latest close against its trailing 10-12 bar mean/std. */
export function signalMeanReversion(bars: Bar[]): SignalResult {
  const closes = bars.slice(-12).map((b) => b.close);
  if (closes.length < 10) return { direction: 0, strength: 0, reason: "insufficient mean-reversion context" };
  const m = mean(closes);
  const s = std(closes);
  if (s <= 1e-12) return { direction: 0, strength: 0, reason: "flat mean-reversion context" };
  const z = (closes.at(-1)! - m) / s;
  if (Math.abs(z) < 2.0) return { direction: 0, strength: 0, reason: "price not statistically stretched" };
  return { direction: z > 0 ? -1 : 1, strength: clip((Math.abs(z) - 1.5) / 2.5), reason: `mean-reversion stretch z=${z.toFixed(3)}` };
}

export function ticketFor(templateId: string): number {
  return Number(TEMPLATES.find((x) => x.id === templateId)?.ticket ?? 5);
}

/** Blends radar liquidity/activity scores with book spread quality into a [0,1] confidence. */
export function confidenceFromRadar(candidate: RadarCandidate, book: Book): number {
  const liquidity = clip(finite(candidate.liquidity_score, 0.5));
  const activity = clip(finite(candidate.activity_score, 0.5));
  const spreadQuality = 1 / (1 + Math.max(0, book.spreadBps) / 10);
  return clip(0.45 * liquidity + 0.35 * activity + 0.20 * spreadQuality);
}

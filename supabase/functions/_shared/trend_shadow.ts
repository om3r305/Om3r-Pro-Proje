// BTC/ETH trend-following paper portfolio (policy trend-ensemble-v1).
//
// Mirrors research/btc_eth_trend_study.py "ensemble weekly": for each asset and
// each lookback L in {50,100,200} days, the asset is "on" when its close is above
// its L-day simple moving average; its weight for that lookback is
// min(0.5, 0.5 * 0.40 / realised 30d vol). The target is the mean over the three
// lookbacks, set on Mondays (UTC) and held constant until the next Monday.
// A day's return is the held weights times that day's close-to-close returns;
// changing weights costs 15 bps per side. Paper only: nothing here places orders.

export const TREND_POLICY_VERSION = "trend-ensemble-v1";
export const TREND_ASSETS = ["BTCUSDT", "ETHUSDT"] as const;
export type TrendAsset = typeof TREND_ASSETS[number];
export type Weights = Record<TrendAsset, number>;

export const LOOKBACKS = [50, 100, 200] as const;
export const VOL_WINDOW = 30;
export const VOL_TARGET = 0.40;
export const MAX_ASSET_WEIGHT = 0.5;
export const SIDE_COST = 0.0015;
export const DAYS_PER_YEAR = 365;
export const MIN_HISTORY = Math.max(...LOOKBACKS) + 1;

export const ZERO_WEIGHTS: Weights = { BTCUSDT: 0, ETHUSDT: 0 };

export function sma(closes: number[], length: number): number {
  if (closes.length < length) throw new Error(`sma needs ${length} closes, got ${closes.length}`);
  const window = closes.slice(-length);
  return window.reduce((a, b) => a + b, 0) / length;
}

/** Annualised sample std (ddof=1) of the last `window` simple returns, as pandas rolling().std(). */
export function realisedVol(closes: number[], window = VOL_WINDOW): number {
  if (closes.length < window + 1) throw new Error(`vol needs ${window + 1} closes, got ${closes.length}`);
  const tail = closes.slice(-(window + 1));
  const rets = tail.slice(1).map((c, i) => c / tail[i] - 1);
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((a, r) => a + (r - mean) ** 2, 0) / (rets.length - 1);
  return Math.sqrt(variance) * Math.sqrt(DAYS_PER_YEAR);
}

export type AssetSignal = { close: number; vol: number; cap: number; on: Record<string, boolean>; weight: number };

export function assetSignal(closes: number[]): AssetSignal {
  if (closes.length < MIN_HISTORY) throw new Error(`need ${MIN_HISTORY} closes, got ${closes.length}`);
  if (closes.some((c) => !(Number.isFinite(c) && c > 0))) throw new Error("closes must be positive finite numbers");
  const close = closes[closes.length - 1];
  const vol = realisedVol(closes);
  const cap = vol > 0 ? Math.min(MAX_ASSET_WEIGHT, MAX_ASSET_WEIGHT * VOL_TARGET / vol) : 0;
  const on: Record<string, boolean> = {};
  let sum = 0;
  for (const L of LOOKBACKS) {
    on[`sma${L}`] = close > sma(closes, L);
    if (on[`sma${L}`]) sum += cap;
  }
  return { close, vol, cap, on, weight: sum / LOOKBACKS.length };
}

export function targetWeights(closes: Record<TrendAsset, number[]>): { weights: Weights; signals: Record<TrendAsset, AssetSignal> } {
  const signals = {} as Record<TrendAsset, AssetSignal>;
  const weights = { ...ZERO_WEIGHTS };
  for (const a of TREND_ASSETS) {
    signals[a] = assetSignal(closes[a]);
    weights[a] = signals[a].weight;
  }
  return { weights, signals };
}

/** Monday in UTC for a YYYY-MM-DD candle date. */
export function isRebalanceDay(day: string): boolean {
  return new Date(`${day}T00:00:00Z`).getUTCDay() === 1;
}

export type LedgerStep = {
  day: string;
  held: Weights;
  target: Weights;
  asset_returns: Weights;
  day_return: number;
  turnover: number;
  cost: number;
  nav: number;
  rebalanced: boolean;
};

/**
 * Advance the paper portfolio by one closed daily candle.
 * `closes` must end at `day` and include the previous close for the day return.
 */
export function step(
  day: string,
  prev: { held: Weights; nav: number } | null,
  closes: Record<TrendAsset, number[]>,
  startingNav = 10_000,
): LedgerStep & { signals: Record<TrendAsset, AssetSignal> } {
  const held = prev?.held ?? { ...ZERO_WEIGHTS };
  const navBefore = prev?.nav ?? startingNav;
  const asset_returns = { ...ZERO_WEIGHTS };
  let day_return = 0;
  for (const a of TREND_ASSETS) {
    const c = closes[a];
    asset_returns[a] = c[c.length - 1] / c[c.length - 2] - 1;
    day_return += held[a] * asset_returns[a];
  }
  const { weights, signals } = targetWeights(closes);
  const rebalanced = prev === null || isRebalanceDay(day);
  const target = rebalanced ? weights : { ...held };
  const turnover = TREND_ASSETS.reduce((s, a) => s + Math.abs(target[a] - held[a]), 0);
  const cost = turnover * SIDE_COST;
  const nav = navBefore * (1 + day_return - cost);
  return { day, held, target, asset_returns, day_return, turnover, cost, nav, rebalanced, signals };
}

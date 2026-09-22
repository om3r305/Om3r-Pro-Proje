// Pure, side-effect-free feature/allocation/accounting logic for brian-live-shadow, split out of
// index.ts so it can be exercised directly by logic.test.ts without a live network/Supabase/
// checkpoint dependency.

export const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"] as const;
export const POLICIES = ["NATIVE", "PROFIT"] as const;

export type PolicyKind = typeof POLICIES[number];
export type SymbolName = typeof SYMBOLS[number];
export type Bar = { closeTime: number; open: number; high: number; low: number; close: number; volume: number };
export type Book = { bid: number; ask: number; mid: number; spreadBps: number };
export type TickState = {
  observed_at: string; equity_after_costs: number; peak_equity_after: number; max_drawdown_pct_after: number;
  target_weights: Record<string, number>; observed_mid_prices: Record<string, number>;
};

export const GYM = { starting_equity: 500.0, fee_bps: 10.0, assumed_spread_bps: 2.0, slippage_bps: 1.0, max_gross_exposure: 1.0, max_asset_weight: 0.35 };
export const PROFIT = { round_trip_cost_multiplier: 2.0, max_positions: 3, max_asset_weight: 0.25, max_gross_exposure: 0.75, min_strength: 0.25 };

export function clip(value: number, low: number, high: number): number {
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

export function safeLogRatio(right: number, left: number): number {
  if (!(right > 0) || !(left > 0)) throw new Error("positive prices required");
  return Math.log(right / left);
}

export function finite(value: unknown): number {
  const n = Number(value);
  if (!Number.isFinite(n)) throw new Error("non-finite market value");
  return n;
}

export function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

/** Inner-joins each symbol's bar series on closeTime, keeping only times present for every symbol, sorted ascending. */
export function alignedFrames(bars: Record<SymbolName, Bar[]>): Array<{ closeTime: number; bars: Record<SymbolName, Bar> }> {
  const maps = {} as Record<SymbolName, Map<number, Bar>>;
  for (const symbol of SYMBOLS) maps[symbol] = new Map(bars[symbol].map((bar) => [bar.closeTime, bar]));
  let times = [...maps[SYMBOLS[0]].keys()];
  for (const symbol of SYMBOLS.slice(1)) times = times.filter((t) => maps[symbol].has(t));
  times.sort((a, b) => a - b);
  return times.map((closeTime) => {
    const out = {} as Record<SymbolName, Bar>;
    for (const symbol of SYMBOLS) out[symbol] = maps[symbol].get(closeTime)!;
    return { closeTime, bars: out };
  });
}

/** Builds the 10-wide per-symbol feature vector (bias + return/trend/vol/range/relative/market/volume features). */
export function featureMap(frames: Array<{ closeTime: number; bars: Record<SymbolName, Bar> }>, lookback: number): Record<SymbolName, number[]> {
  const visible = frames.slice(-lookback);
  if (visible.length < 2) throw new Error("feature map requires at least two frames");
  const current = visible[visible.length - 1];
  const previous = visible[visible.length - 2];
  const return1 = {} as Record<SymbolName, number>;
  for (const symbol of SYMBOLS) return1[symbol] = safeLogRatio(current.bars[symbol].close, previous.bars[symbol].close);
  const marketReturns = SYMBOLS.map((s) => return1[s]);
  const marketMomentum = mean(marketReturns);
  const marketDispersion = std(marketReturns);
  const out = {} as Record<SymbolName, number[]>;
  for (const symbol of SYMBOLS) {
    const assetBars = visible.map((f) => f.bars[symbol]);
    const closeReturns = assetBars.slice(1).map((bar, i) => safeLogRatio(bar.close, assetBars[i].close));
    const r1 = return1[symbol];
    const r3 = assetBars.length >= 4
      ? safeLogRatio(assetBars.at(-1)!.close, assetBars.at(-4)!.close)
      : safeLogRatio(assetBars.at(-1)!.close, assetBars[0].close);
    const trend = mean(closeReturns.slice(-5));
    const vol = std(closeReturns.slice(-lookback));
    const currentBar = current.bars[symbol];
    const prevBar = previous.bars[symbol];
    const barRange = Math.max(0, currentBar.high / currentBar.low - 1);
    const relative = r1 - marketMomentum;
    let volumeChange = 0;
    if (currentBar.volume > 0 && prevBar.volume > 0) volumeChange = clip(Math.log(currentBar.volume / prevBar.volume) / 3, -1, 1);
    out[symbol] = [1, clip(r1 * 20, -3, 3), clip(r3 * 10, -3, 3), clip(trend * 20, -3, 3), clip(vol * 20, 0, 3), clip(barRange * 10, 0, 3), clip(relative * 20, -3, 3), clip(marketMomentum * 20, -3, 3), clip(marketDispersion * 20, 0, 3), volumeChange];
  }
  return out;
}

export interface DrawdownConfig {
  drawdown_flatten: number;
  drawdown_throttle_2: number;
  drawdown_throttle_1: number;
  max_gross_exposure: number;
}

/** Steps down the tradeable gross-exposure budget as drawdown from starting equity deepens. */
export function grossBudget(equity: number, cfg: DrawdownConfig): number {
  const dd = Math.max(0, 1 - equity / GYM.starting_equity);
  if (dd >= cfg.drawdown_flatten) return 0;
  if (dd >= cfg.drawdown_throttle_2) return cfg.max_gross_exposure * 0.25;
  if (dd >= cfg.drawdown_throttle_1) return cfg.max_gross_exposure * 0.5;
  return cfg.max_gross_exposure;
}

export interface ModelWeights {
  weights: number[];
  weighted_samples: number;
  error_ewma: number;
}

export interface AllocationConfig extends DrawdownConfig {
  max_label_abs: number;
  min_uncertainty: number;
  risk_aversion: number;
  min_weighted_samples_per_asset: number;
  min_abs_edge: number;
  turnover_penalty_bps: number;
  max_positions: number;
  max_asset_weight: number;
}

/** Turns per-symbol linear model predictions into a risk-budgeted target-weight allocation. */
export function chooseAllocation(
  policy: PolicyKind,
  features: Record<SymbolName, number[]>,
  current: Record<string, number>,
  equity: number,
  models: Record<SymbolName, ModelWeights>,
  cfg: AllocationConfig,
) {
  const budget = Math.min(grossBudget(equity, cfg), policy === "PROFIT" ? PROFIT.max_gross_exposure : cfg.max_gross_exposure);
  const candidates: Array<{ score: number; asset: SymbolName; desired: number }> = [];
  const diagnostics: Record<string, unknown> = {};
  const fixedOneWayCostRate = (GYM.fee_bps + GYM.assumed_spread_bps / 2 + GYM.slippage_bps) / 10000;
  for (const symbol of SYMBOLS) {
    const model = models[symbol];
    const prediction = clip(dot(model.weights.map(Number), features[symbol]), -cfg.max_label_abs, cfg.max_label_abs);
    const uncertainty = Math.max(cfg.min_uncertainty, Number(model.error_ewma));
    const longEdge = prediction - cfg.risk_aversion * uncertainty;
    const shortEdge = -prediction - cfg.risk_aversion * uncertainty;
    const direction = longEdge >= shortEdge ? 1 : -1;
    const rawEdge = Math.max(longEdge, shortEdge);
    let score: number | null = null;
    let desired = 0;
    if (Number(model.weighted_samples) >= cfg.min_weighted_samples_per_asset && rawEdge > cfg.min_abs_edge) {
      if (policy === "NATIVE") {
        const strength = clip(rawEdge / Math.max(cfg.min_abs_edge * 4, 1e-12), 0.25, 1);
        desired = direction * cfg.max_asset_weight * strength;
        score = rawEdge - cfg.turnover_penalty_bps / 10000 * Math.abs(desired - (current[symbol] ?? 0));
        if (score > cfg.min_abs_edge) candidates.push({ score, asset: symbol, desired });
      } else {
        const nativeStrength = clip(rawEdge / Math.max(cfg.min_abs_edge * 4, 1e-12), PROFIT.min_strength, 1);
        desired = direction * Math.min(PROFIT.max_asset_weight, cfg.max_asset_weight) * nativeStrength;
        const incrementalTurnover = Math.abs(desired - (current[symbol] ?? 0));
        const netEdge = rawEdge - fixedOneWayCostRate * PROFIT.round_trip_cost_multiplier * incrementalTurnover;
        score = netEdge;
        if (netEdge > cfg.min_abs_edge) {
          const netStrength = clip(netEdge / Math.max(cfg.min_abs_edge * 4, 1e-12), PROFIT.min_strength, 1);
          desired = direction * Math.min(PROFIT.max_asset_weight, cfg.max_asset_weight) * netStrength;
          candidates.push({ score: netEdge, asset: symbol, desired });
        }
      }
    }
    diagnostics[symbol] = { prediction, uncertainty, long_edge: longEdge, short_edge: shortEdge, raw_edge: rawEdge, selection_score: score, direction, selected_weight: 0 };
  }
  candidates.sort((a, b) => b.score - a.score || a.asset.localeCompare(b.asset));
  const maxPositions = policy === "PROFIT" ? PROFIT.max_positions : cfg.max_positions;
  const maxAssetWeight = policy === "PROFIT" ? PROFIT.max_asset_weight : cfg.max_asset_weight;
  const weights: Record<string, number> = {};
  let remaining = budget;
  for (const candidate of candidates.slice(0, maxPositions)) {
    const magnitude = Math.min(Math.abs(candidate.desired), remaining, maxAssetWeight);
    if (magnitude <= 1e-12) break;
    weights[candidate.asset] = Math.sign(candidate.desired) * magnitude;
    (diagnostics[candidate.asset] as Record<string, unknown>).selected_weight = weights[candidate.asset];
    remaining -= magnitude;
  }
  return { weights, diagnostics };
}

/** Marks a prior tick's weights to the current book and returns drifted (post-return) weights. */
export function markAndDrift(previous: TickState | null, books: Record<SymbolName, Book>) {
  if (!previous) return { equityAfterMark: GYM.starting_equity, drifted: {} as Record<string, number>, periodPnl: 0 };
  const priorWeights = previous.target_weights ?? {};
  const priorPrices = previous.observed_mid_prices ?? {};
  let portfolioReturn = 0;
  for (const [asset, weight] of Object.entries(priorWeights)) {
    const now = books[asset as SymbolName]?.mid;
    const before = Number(priorPrices[asset]);
    if (!(now > 0) || !(before > 0)) throw new Error(`missing mark price for ${asset}`);
    portfolioReturn += Number(weight) * (now / before - 1);
  }
  const equityBefore = Number(previous.equity_after_costs);
  const equityAfterMark = Math.max(0, equityBefore * (1 + portfolioReturn));
  const drifted: Record<string, number> = {};
  if (equityAfterMark > 0) {
    for (const [asset, weight] of Object.entries(priorWeights)) {
      const ratio = books[asset as SymbolName].mid / Number(priorPrices[asset]);
      drifted[asset] = Number(weight) * equityBefore * ratio / equityAfterMark;
    }
  }
  return { equityAfterMark, drifted, periodPnl: equityAfterMark - equityBefore };
}

/** Fee + slippage + half-spread cost of moving from one weight vector to another. */
export function executionCost(equity: number, from: Record<string, number>, to: Record<string, number>, books: Record<SymbolName, Book>) {
  let turnoverFraction = 0;
  let cost = 0;
  const assets = new Set([...Object.keys(from), ...Object.keys(to)]);
  for (const asset of assets) {
    const delta = Math.abs((to[asset] ?? 0) - (from[asset] ?? 0));
    if (delta <= 1e-15) continue;
    turnoverFraction += delta;
    const spreadBps = books[asset as SymbolName]?.spreadBps;
    if (!Number.isFinite(spreadBps)) throw new Error(`missing spread for ${asset}`);
    cost += equity * delta * (GYM.fee_bps + GYM.slippage_bps + Number(spreadBps) / 2) / 10000;
  }
  return { turnoverNotional: equity * turnoverFraction, cost };
}

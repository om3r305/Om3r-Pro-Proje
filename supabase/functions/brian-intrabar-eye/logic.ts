// Pure, import-safe Phase 4.0 intrabar logic: signal helpers, consensus/late-chase scoring,
// and micro-book accounting. No Deno.serve, no network fetch, no Supabase client here, so this
// module can be imported directly by `deno test` (and, via shared JSON fixtures, compared
// against the Python reference model in brian2026/intrabar_reaction.py) without triggering any
// runtime side effects. index.ts imports from this file; it does not duplicate this logic.
//
// This is a behavior-preserving extraction: every exported function here reproduces exactly
// what index.ts computed inline before this refactor. See PR description for the equivalence
// argument per function.

export const FEE_BPS = 10.0;
export const SLIPPAGE_BPS = 1.0;
export const EVIDENCE_CLASS = "PROSPECTIVE_DEVELOPMENT_SHADOW";

export type RadarCandidate = { symbol: string; radar_score?: number; liquidity_score?: number; activity_score?: number; spread_bps?: number | null };
export type Book = { bid: number; ask: number; mid: number; spreadBps: number };
export type Bar = {
  openTime: number; closeTime: number; open: number; high: number; low: number; close: number;
  baseVolume: number; quoteVolume: number; trades: number; takerBuyQuote: number;
};
export type AggTrade = { price: number; qty: number; quote: number; ts: number; buyerMaker: boolean };
export type Signal = { direction: number; strength: number; reason: string; metadata?: Record<string, unknown> };
export type MarketRow = {
  candidate: RadarCandidate; bars: Bar[]; trades: AggTrade[]; book: Book; degradedAggTrades: boolean;
  sigma: number; current: Bar; baseline: Bar[]; elapsedFraction: number; medianQuoteVolume: number;
  velocityCoverageSeconds: number; return30s: number; previous30sReturn: number; decelerating: boolean;
};

export function finite(value: unknown, fallback = 0): number { const n = Number(value); return Number.isFinite(n) ? n : fallback; }
export function clip(value: number, low = 0, high = 1): number { return Math.max(low, Math.min(high, value)); }
export function mean(values: number[]): number { return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0; }
export function median(values: number[]): number { if (!values.length) return 0; const x = [...values].sort((a, b) => a - b); const m = Math.floor(x.length / 2); return x.length % 2 ? x[m] : (x[m - 1] + x[m]) / 2; }
export function std(values: number[]): number { if (values.length <= 1) return 0; const m = mean(values); return Math.sqrt(mean(values.map((v) => (v - m) ** 2))); }
export function sign(value: number): number { return value > 0 ? 1 : value < 0 ? -1 : 0; }

export function logReturns(values: number[]): number[] { return values.slice(1).map((v, i) => Math.log(v / values[i])); }
export function windowReturn(trades: AggTrade[], observedMs: number, seconds: number, offsetSeconds = 0): number {
  const end = observedMs - offsetSeconds * 1000; const start = end - seconds * 1000;
  const rows = trades.filter((t) => t.ts >= start && t.ts <= end); if (rows.length < 2) return 0;
  return Math.log(rows.at(-1)!.price / rows[0].price);
}
export function confidenceFromRadar(candidate: RadarCandidate, book: Book): number {
  const liquidity = clip(finite(candidate.liquidity_score, 0.5)); const activity = clip(finite(candidate.activity_score, 0.5));
  const spreadQuality = 1 / (1 + Math.max(0, book.spreadBps) / 10);
  return clip(0.45 * liquidity + 0.35 * activity + 0.20 * spreadQuality);
}

export function parseBars(payload: unknown): Bar[] {
  if (!Array.isArray(payload)) throw new Error("invalid 1m kline payload");
  return payload.map((k) => {
    if (!Array.isArray(k) || k.length < 11) throw new Error("invalid 1m kline row");
    return {
      openTime: finite(k[0]), open: finite(k[1]), high: finite(k[2]), low: finite(k[3]), close: finite(k[4]),
      baseVolume: finite(k[5]), closeTime: finite(k[6]), quoteVolume: finite(k[7]), trades: Math.max(0, Math.trunc(finite(k[8]))),
      takerBuyQuote: finite(k[10]),
    };
  }).filter((b) => b.openTime > 0 && b.open > 0 && b.high > 0 && b.low > 0 && b.close > 0 && b.high >= b.low);
}
export function parseAggTrades(payload: unknown): AggTrade[] {
  if (!Array.isArray(payload)) return [];
  return payload.map((row) => {
    const r = row as Record<string, unknown>; const price = finite(r.p); const qty = finite(r.q); const ts = finite(r.T);
    return { price, qty, quote: price * qty, ts, buyerMaker: Boolean(r.m) };
  }).filter((r) => r.price > 0 && r.qty > 0 && r.ts > 0).sort((a, b) => a.ts - b.ts);
}

export function signalVelocity(row: MarketRow, observedMs: number): Signal {
  const coverage = row.velocityCoverageSeconds; if (coverage < 5) return { direction: 0, strength: 0, reason: "insufficient live trade velocity coverage", metadata: { coverage_seconds: coverage } };
  const live = windowReturn(row.trades, observedMs, Math.min(60, Math.max(5, coverage)));
  const adaptive = Math.max(0.00010, row.sigma * Math.sqrt(Math.max(5, Math.min(60, coverage)) / 60) * 0.70);
  if (Math.abs(live) < adaptive) return { direction: 0, strength: 0, reason: "live trade velocity below adaptive threshold", metadata: { live_return: live, threshold: adaptive, coverage_seconds: coverage } };
  return { direction: sign(live), strength: clip(Math.abs(live) / (adaptive * 3)), reason: "live trade velocity impulse", metadata: { live_return: live, threshold: adaptive, coverage_seconds: coverage } };
}
export function signalVolumeBurst(row: MarketRow): Signal {
  const pace = row.current.quoteVolume / Math.max(row.elapsedFraction, 0.15) / Math.max(row.medianQuoteVolume, 1e-12);
  const body = Math.log(row.book.mid / row.current.open); const threshold = Math.max(0.00015, row.sigma * 0.45);
  if (pace < 1.8 || Math.abs(body) < threshold) return { direction: 0, strength: 0, reason: "partial 1m volume/body not impulsive", metadata: { pace_ratio: pace, body_return: body, threshold } };
  const strength = clip(0.55 * clip(Math.abs(body) / (threshold * 3)) + 0.45 * clip((pace - 1) / 4));
  return { direction: sign(body), strength, reason: "partial 1m relative-volume acceleration", metadata: { pace_ratio: pace, body_return: body, threshold } };
}
export function signalBreakout(row: MarketRow): Signal {
  const prior = row.baseline.slice(-10); if (prior.length < 8) return { direction: 0, strength: 0, reason: "insufficient 1m breakout context" };
  const high = Math.max(...prior.map((b) => b.high)); const low = Math.min(...prior.map((b) => b.low));
  const oneWay = (FEE_BPS + SLIPPAGE_BPS + row.book.spreadBps / 2) / 10000; const buffer = Math.max(0.00012, oneWay * 0.30, row.sigma * 0.30);
  if (row.book.mid > high * (1 + buffer)) {
    const dist = Math.log(row.book.mid / high); return { direction: 1, strength: clip(dist / Math.max(buffer * 4, row.sigma * 2)), reason: "pre-close 1m breakout above prior structure", metadata: { prior_high: high, breakout_return: dist, buffer } };
  }
  if (row.book.mid < low * (1 - buffer)) {
    const dist = Math.log(low / row.book.mid); return { direction: -1, strength: clip(dist / Math.max(buffer * 4, row.sigma * 2)), reason: "pre-close 1m breakdown below prior structure", metadata: { prior_low: low, breakout_return: -dist, buffer } };
  }
  return { direction: 0, strength: 0, reason: "no pre-close 1m structure break", metadata: { prior_high: high, prior_low: low, buffer } };
}
export function signalReclaim(row: MarketRow): Signal {
  const bars = row.bars; if (bars.length < 13) return { direction: 0, strength: 0, reason: "insufficient reclaim context" };
  const buffer = Math.max(0.00010, row.sigma * 0.20); const context = bars.slice(-13, -1); const priorLow = Math.min(...context.map((b) => b.low)); const priorHigh = Math.max(...context.map((b) => b.high));
  const current = row.current; const currentMidpoint = (current.high + current.low) / 2;
  if (current.low < priorLow * (1 - buffer) && row.book.mid > priorLow && row.book.mid > currentMidpoint) {
    const sweep = Math.log(priorLow / current.low); const reclaim = Math.log(row.book.mid / priorLow);
    return { direction: 1, strength: clip((sweep + reclaim) / Math.max(row.sigma * 3, buffer * 5)), reason: "same-minute downside sweep reclaimed", metadata: { prior_low: priorLow, sweep_depth: sweep, reclaim_return: reclaim } };
  }
  if (current.high > priorHigh * (1 + buffer) && row.book.mid < priorHigh && row.book.mid < currentMidpoint) {
    const sweep = Math.log(current.high / priorHigh); const reclaim = Math.log(priorHigh / row.book.mid);
    return { direction: -1, strength: clip((sweep + reclaim) / Math.max(row.sigma * 3, buffer * 5)), reason: "same-minute upside sweep rejected", metadata: { prior_high: priorHigh, sweep_depth: sweep, reclaim_return: -reclaim } };
  }
  const previous = bars.at(-2); const beforePrevious = bars.slice(-13, -2);
  if (previous && beforePrevious.length >= 8) {
    const low = Math.min(...beforePrevious.map((b) => b.low)); const high = Math.max(...beforePrevious.map((b) => b.high));
    if (previous.low < low * (1 - buffer) && row.book.mid > low && row.book.mid > previous.close) {
      return { direction: 1, strength: clip(Math.log(row.book.mid / Math.max(previous.low, 1e-12)) / Math.max(row.sigma * 4, buffer * 6)), reason: "prior-minute downside sweep reclaimed", metadata: { prior_low: low, sweep_low: previous.low } };
    }
    if (previous.high > high * (1 + buffer) && row.book.mid < high && row.book.mid < previous.close) {
      return { direction: -1, strength: clip(Math.log(previous.high / row.book.mid) / Math.max(row.sigma * 4, buffer * 6)), reason: "prior-minute upside sweep rejected", metadata: { prior_high: high, sweep_high: previous.high } };
    }
  }
  return { direction: 0, strength: 0, reason: "no liquidity sweep/reclaim pattern" };
}
export function signalTakerFlow(row: MarketRow): Signal {
  const q = row.current.quoteVolume; if (q <= 0) return { direction: 0, strength: 0, reason: "no partial 1m quote volume" };
  const buyShare = clip(row.current.takerBuyQuote / q); const pace = q / Math.max(row.elapsedFraction, 0.15) / Math.max(row.medianQuoteVolume, 1e-12);
  if (pace < 0.35 || (buyShare >= 0.42 && buyShare <= 0.58)) return { direction: 0, strength: 0, reason: "partial 1m taker flow balanced", metadata: { buy_share: buyShare, pace_ratio: pace } };
  const direction = buyShare > 0.58 ? 1 : -1; const imbalance = Math.abs(buyShare - 0.5) * 2;
  return { direction, strength: clip(imbalance * Math.min(1, 0.5 + pace / 2)), reason: "partial 1m taker-flow imbalance", metadata: { buy_share: buyShare, pace_ratio: pace } };
}

/** Bars/mid/sigma in, five-minute extension in realized-volatility units out. Feeds the late-chase veto. */
export function computeExtensionSigma(baseline: Bar[], mid: number, sigma: number): number {
  const close5 = baseline.at(-5)?.close ?? baseline[0].close;
  return Math.abs(Math.log(mid / close5)) / Math.max(sigma * Math.sqrt(5), 1e-12);
}

// Field names/shape intentionally mirror brian2026/global_sensor_mesh.py's SensorObservation
// (independent_group/direction/strength/confidence/reliability) so the same JSON fixtures can
// drive both this function and brian2026/intrabar_reaction.py::build_intrabar_consensus.
export interface IntrabarSignalRow {
  independentGroup: string;
  direction: number;
  strength: number;
  confidence: number;
  reliability: number;
}

export interface IntrabarConsensusParams {
  minSupportGroups: number;
  minConsensusScore: number;
  overextensionSigma: number;
  extensionSigma: number;
  decelerating: boolean;
  spreadBps: number;
}

export type IntrabarStatus = "WATCH" | "ACTIONABLE_SHADOW" | "VETOED_LATE_CHASE";

export interface IntrabarConsensusOutcome {
  direction: number;
  score: number;
  supportGroups: string[];
  conflictGroups: string[];
  eligible: boolean;
  staleVelocity: boolean;
  flowConflict: boolean;
  lateChase: boolean;
  status: IntrabarStatus;
  roundTripCostBps: number;
}

/**
 * Consensus + late-chase veto scoring for the Phase 4.0 intrabar reaction layer.
 *
 * Behavior-preserving extraction of the block that used to live inline in index.ts's request
 * handler (per-asset loop). For today's production call site — exactly one signal per
 * independent_group, already pre-filtered to direction !== 0 — the by-group dedup below is a
 * no-op and this function's output is numerically identical to the prior inline computation.
 * The dedup is kept (mirroring brian2026/intrabar_reaction.py::build_intrabar_consensus) so the
 * function is also correct, and testable, for fixture inputs with duplicate-group observations.
 */
export function buildIntrabarConsensus(
  rows: IntrabarSignalRow[],
  params: IntrabarConsensusParams,
): IntrabarConsensusOutcome {
  const byGroup = new Map<string, IntrabarSignalRow>();
  for (const row of rows) {
    if (row.direction === 0) continue;
    const quality = row.strength * row.confidence * row.reliability;
    const current = byGroup.get(row.independentGroup);
    const currentQuality = current ? current.strength * current.confidence * current.reliability : Number.NEGATIVE_INFINITY;
    if (!current || quality > currentQuality) byGroup.set(row.independentGroup, row);
  }
  const evidence = [...byGroup.values()];
  const roundTripCostBps = 2 * (FEE_BPS + SLIPPAGE_BPS + params.spreadBps / 2);
  if (!evidence.length) {
    return { direction: 0, score: 0, supportGroups: [], conflictGroups: [], eligible: false, staleVelocity: false, flowConflict: false, lateChase: false, status: "WATCH", roundTripCostBps };
  }

  const signed = evidence.map((row) => row.direction * row.strength * row.confidence * row.reliability);
  const aggregate = mean(signed);
  const direction = sign(aggregate);
  const support = evidence.filter((row) => row.direction === direction);
  const conflicts = evidence.filter((row) => row.direction !== direction);
  const supportRatio = support.length / evidence.length;
  const breadth = 0.4 + 0.6 * Math.min(1, evidence.length / 3);
  const score = clip(Math.abs(aggregate) * (0.5 + 0.5 * supportRatio) * breadth);
  const eligible = direction !== 0 && support.length >= params.minSupportGroups && score >= params.minConsensusScore;

  const velocity = byGroup.get("micro_velocity");
  const taker = byGroup.get("micro_taker_flow");
  const staleVelocity = velocity ? (velocity.direction !== direction || velocity.strength < 0.25) : true;
  const flowConflict = taker ? taker.direction === -direction : false;
  const lateChase = eligible && params.extensionSigma >= params.overextensionSigma && (params.decelerating || staleVelocity || flowConflict);
  const status: IntrabarStatus = eligible && !lateChase ? "ACTIONABLE_SHADOW" : lateChase ? "VETOED_LATE_CHASE" : "WATCH";

  return {
    direction,
    score,
    supportGroups: support.map((row) => row.independentGroup).sort(),
    conflictGroups: conflicts.map((row) => row.independentGroup).sort(),
    eligible,
    staleVelocity,
    flowConflict,
    lateChase,
    status,
    roundTripCostBps,
  };
}

export type PriorTick = {
  eye_id: string; starting_equity: string | number; equity_after: string | number; peak_equity_after: string | number;
  max_drawdown_pct_after: number; target_direction: number; observed_mid_price: string | number; observed_at?: string;
};

export interface MicroTickParams {
  eyeId: string; templateId: string; assetId: string; observedAt: string; mid: number; spreadBps: number;
  strength: number; confidence: number; targetDirection: number; startingTicket: number; rawCaptureId: string;
  prior?: PriorTick; metadata: Record<string, unknown>;
}

/** Cost-aware virtual micro-book tick: reads the prior tick, marks P&L, charges turnover cost. Pure — no I/O. */
export function microTick(params: MicroTickParams): Record<string, unknown> | null {
  const priorDirection = params.prior ? Number(params.prior.target_direction) : 0;
  if (!params.prior && params.targetDirection === 0) return null;
  if (params.prior && priorDirection === 0 && params.targetDirection === 0) return null;
  const starting = params.prior ? Number(params.prior.starting_equity) : params.startingTicket;
  const equityBefore = params.prior ? Number(params.prior.equity_after) : starting;
  const priorMid = params.prior ? Number(params.prior.observed_mid_price) : params.mid;
  const periodPnl = params.prior ? equityBefore * priorDirection * (params.mid / priorMid - 1) : 0;
  const marked = Math.max(0, equityBefore + periodPnl); const turnover = Math.abs(params.targetDirection - priorDirection);
  const oneWay = (FEE_BPS + SLIPPAGE_BPS + params.spreadBps / 2) / 10000; const tradingCost = marked * turnover * oneWay;
  const equityAfter = Math.max(0, marked - tradingCost); const peak = Math.max(params.prior ? Number(params.prior.peak_equity_after) : starting, equityAfter);
  const dd = 100 * Math.max(0, peak - equityAfter) / Math.max(peak, 1e-12); const maxDd = Math.max(params.prior ? Number(params.prior.max_drawdown_pct_after) : 0, dd);
  return { eye_id: params.eyeId, template_id: params.templateId, asset_id: params.assetId, horizon: "MICRO_1_5M", observed_at: params.observedAt, feature_close_at: params.observedAt, starting_equity: starting, equity_before: equityBefore, period_pnl: periodPnl, trading_cost: tradingCost, equity_after: equityAfter, peak_equity_after: peak, max_drawdown_pct_after: maxDd, prior_direction: priorDirection, target_direction: params.targetDirection, observed_mid_price: params.mid, observed_spread_bps: params.spreadBps, signal_strength: params.strength, signal_confidence: params.confidence, raw_capture_id: params.rawCaptureId, evidence_class: EVIDENCE_CLASS, shadow_only: true, live_execution: false, metadata: params.metadata };
}

// Brian 2026 Emergent Mover scout.
//
// Observation only: this module ranks where market behaviour is changing relative to the
// immediately-prior prospective frame and to the current cross-section. It does not produce
// BUY/SELL actions, target weights, order instructions, or promotion decisions.
//
// Important measurement note: Binance 24h quote volume / trade count are rolling-window
// quantities. Their frame-to-frame change is therefore a rolling-window displacement, NOT an
// exact interval flow. The report preserves that distinction for downstream consumers.

export const EMERGENT_MOVER_FRAME_SCHEMA = "brian.emergent-mover-frame.v1" as const;
export const EMERGENT_MOVER_REPORT_SCHEMA = "brian.emergent-mover-report.v1" as const;

export interface EmergentMarketRow {
  symbol: string;
  last_price: number;
  quote_volume_24h: number;
  trades_24h: number;
  price_change_pct_24h: number;
  high_price_24h: number;
  low_price_24h: number;
  spread_bps: number | null;
}

export interface EmergentMoverStateRow {
  symbol: string;
  last_price: number;
  quote_volume_24h: number;
  trades_24h: number;
  price_change_pct_24h: number;
  range_pct_24h: number;
  spread_bps: number | null;
  liquidity_rank: number;
  activity_rank: number;
  volatility_rank: number;
  momentum_rank: number;
  spread_quality: number;
}

export interface EmergentMoverFrame {
  schema_version: typeof EMERGENT_MOVER_FRAME_SCHEMA;
  observed_at: string;
  source: string;
  rows: EmergentMoverStateRow[];
  shadow_only: true;
}

export interface EmergentMoverFeatures {
  price_change_delta_pct: number;
  price_impulse_abs_pct: number;
  range_delta_pct: number;
  liquidity_rank_delta: number;
  activity_rank_delta: number;
  volatility_rank_delta: number;
  momentum_rank_delta: number;
  quote_volume_log_delta: number;
  trades_log_delta: number;
  current_spread_bps: number | null;
  current_spread_quality: number;
}

export interface EmergentMoverCandidate {
  rank: number;
  symbol: string;
  attention_score: number;
  observed_change_direction: -1 | 0 | 1;
  features: EmergentMoverFeatures;
  contribution_ranks: {
    price_impulse: number;
    momentum_acceleration: number;
    volatility_acceleration: number;
    activity_acceleration: number;
    liquidity_acceleration: number;
    rolling_quote_volume_gain: number;
    rolling_trade_count_gain: number;
  };
  reasons: string[];
  evidence_class: "PROSPECTIVE_DEVELOPMENT_SHADOW";
  shadow_only: true;
}

export interface EmergentMoverReport {
  schema_version: typeof EMERGENT_MOVER_REPORT_SCHEMA;
  source: string;
  observed_at: string;
  baseline_observed_at: string | null;
  comparison_age_ms: number | null;
  comparable: boolean;
  comparison_issue: null | "no_baseline" | "comparison_too_old";
  compared_symbol_count: number;
  newly_observed_symbols: string[];
  disappeared_symbols: string[];
  candidates: EmergentMoverCandidate[];
  measurement_notes: string[];
  evidence_class: "PROSPECTIVE_DEVELOPMENT_SHADOW";
  shadow_only: true;
}

export interface EmergentMoverConfig {
  max_candidates: number;
  max_comparison_age_ms: number;
}

export const DEFAULT_EMERGENT_MOVER_CONFIG: EmergentMoverConfig = Object.freeze({
  // Attention budget, not a trading threshold. No score cutoff is used.
  max_candidates: 15,
  // Data-quality guard: do not compare a current frame to an arbitrarily old market regime.
  max_comparison_age_ms: 30 * 60 * 1000,
});

function finite(value: number, name: string): number {
  if (!Number.isFinite(value)) throw new Error(`${name} must be finite`);
  return value;
}

function nonNegative(value: number, name: string): number {
  const out = finite(value, name);
  if (out < 0) throw new Error(`${name} cannot be negative`);
  return out;
}

function score01(value: number, name: string): number {
  const out = finite(value, name);
  if (out < 0 || out > 1) throw new Error(`${name} must be in [0,1]`);
  return out;
}

function validTimestamp(value: string, name: string): number {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${name} is required`);
  const ms = Date.parse(value);
  if (!Number.isFinite(ms)) throw new Error(`${name} must be a valid timestamp`);
  return ms;
}

function validateSymbol(symbol: string): string {
  if (typeof symbol !== "string" || !symbol.trim()) throw new Error("symbol is required");
  return symbol.trim().toUpperCase();
}

/** Tie-aware [0,1] cross-sectional percentile ranks. Equal values receive equal average rank. */
export function percentileRanks(values: number[]): number[] {
  values.forEach((value, index) => finite(value, `values[${index}]`));
  if (!values.length) return [];
  if (values.length === 1) return [1];

  const order = values.map((value, index) => ({ value, index }));
  order.sort((a, b) => a.value - b.value || a.index - b.index);
  const result = new Array<number>(values.length).fill(0);
  const denominator = values.length - 1;

  let start = 0;
  while (start < order.length) {
    let end = start;
    while (end + 1 < order.length && order[end + 1].value === order[start].value) end++;
    const averageRank = (start + end) / 2;
    const percentile = averageRank / denominator;
    for (let i = start; i <= end; i++) result[order[i].index] = percentile;
    start = end + 1;
  }
  return result;
}

/**
 * Cross-sectional rank used for acceleration features.
 * Zero/non-positive acceleration contributes exactly zero. Positive values are ranked only
 * against other positive observations, so an unchanged market never receives a fake 0.5 rank.
 */
export function positivePercentileRanks(values: number[]): number[] {
  values.forEach((value, index) => finite(value, `values[${index}]`));
  const result = new Array<number>(values.length).fill(0);
  const positives = values
    .map((value, index) => ({ value, index }))
    .filter((item) => item.value > 0)
    .sort((a, b) => a.value - b.value || a.index - b.index);
  if (!positives.length) return result;

  let start = 0;
  while (start < positives.length) {
    let end = start;
    while (end + 1 < positives.length && positives[end + 1].value === positives[start].value) end++;
    // Rank from 1/N through 1. Ties receive the same average positive rank.
    const averageOrdinal = ((start + 1) + (end + 1)) / 2;
    const rank = averageOrdinal / positives.length;
    for (let i = start; i <= end; i++) result[positives[i].index] = rank;
    start = end + 1;
  }
  return result;
}

export function buildEmergentMoverFrame(
  rows: EmergentMarketRow[],
  options: { observed_at: string; source: string },
): EmergentMoverFrame {
  validTimestamp(options.observed_at, "observed_at");
  if (!options.source?.trim()) throw new Error("source is required");

  const seen = new Set<string>();
  const normalized = rows.map((input, index) => {
    const symbol = validateSymbol(input.symbol);
    if (seen.has(symbol)) throw new Error(`duplicate symbol: ${symbol}`);
    seen.add(symbol);

    const lastPrice = finite(input.last_price, `rows[${index}].last_price`);
    if (!(lastPrice > 0)) throw new Error(`rows[${index}].last_price must be positive`);
    const quoteVolume = nonNegative(input.quote_volume_24h, `rows[${index}].quote_volume_24h`);
    const trades = nonNegative(input.trades_24h, `rows[${index}].trades_24h`);
    if (!Number.isInteger(trades)) throw new Error(`rows[${index}].trades_24h must be an integer`);
    const change = finite(input.price_change_pct_24h, `rows[${index}].price_change_pct_24h`);
    const high = nonNegative(input.high_price_24h, `rows[${index}].high_price_24h`);
    const low = nonNegative(input.low_price_24h, `rows[${index}].low_price_24h`);
    if (high < low) throw new Error(`rows[${index}] high cannot be below low`);
    const spread = input.spread_bps === null
      ? null
      : nonNegative(input.spread_bps, `rows[${index}].spread_bps`);
    const range = 100 * Math.max(0, high - low) / lastPrice;
    return { symbol, lastPrice, quoteVolume, trades, change, range, spread };
  });

  // Sort before deriving ranks so frame JSON is deterministic for identical market content.
  normalized.sort((a, b) => a.symbol.localeCompare(b.symbol));
  const liquidity = percentileRanks(normalized.map((row) => Math.log1p(row.quoteVolume)));
  const activity = percentileRanks(normalized.map((row) => Math.log1p(row.trades)));
  const volatility = percentileRanks(normalized.map((row) => row.range));
  const momentum = percentileRanks(normalized.map((row) => Math.abs(row.change)));

  const stateRows: EmergentMoverStateRow[] = normalized.map((row, i) => ({
    symbol: row.symbol,
    last_price: row.lastPrice,
    quote_volume_24h: row.quoteVolume,
    trades_24h: row.trades,
    price_change_pct_24h: row.change,
    range_pct_24h: row.range,
    spread_bps: row.spread,
    liquidity_rank: liquidity[i],
    activity_rank: activity[i],
    volatility_rank: volatility[i],
    momentum_rank: momentum[i],
    // Missing top-of-book is explicitly neutral, never magically perfect.
    spread_quality: row.spread === null ? 0.5 : 1 / (1 + row.spread / 10),
  }));

  return {
    schema_version: EMERGENT_MOVER_FRAME_SCHEMA,
    observed_at: options.observed_at,
    source: options.source.trim(),
    rows: stateRows,
    shadow_only: true,
  };
}

export function parseEmergentMoverFrame(value: unknown): EmergentMoverFrame {
  if (!value || typeof value !== "object") throw new Error("emergent frame must be an object");
  const raw = value as Record<string, unknown>;
  if (raw.schema_version !== EMERGENT_MOVER_FRAME_SCHEMA) throw new Error("unsupported emergent frame schema");
  if (raw.shadow_only !== true) throw new Error("emergent frame must be shadow-only");
  const observedAt = String(raw.observed_at ?? "");
  validTimestamp(observedAt, "observed_at");
  const source = String(raw.source ?? "").trim();
  if (!source) throw new Error("source is required");
  if (!Array.isArray(raw.rows)) throw new Error("rows must be an array");

  const seen = new Set<string>();
  const rows: EmergentMoverStateRow[] = raw.rows.map((item, index) => {
    if (!item || typeof item !== "object") throw new Error(`rows[${index}] must be an object`);
    const row = item as Record<string, unknown>;
    const symbol = validateSymbol(String(row.symbol ?? ""));
    if (seen.has(symbol)) throw new Error(`duplicate symbol: ${symbol}`);
    seen.add(symbol);
    const lastPrice = finite(Number(row.last_price), `rows[${index}].last_price`);
    if (!(lastPrice > 0)) throw new Error(`rows[${index}].last_price must be positive`);
    const quoteVolume = nonNegative(Number(row.quote_volume_24h), `rows[${index}].quote_volume_24h`);
    const trades = nonNegative(Number(row.trades_24h), `rows[${index}].trades_24h`);
    if (!Number.isInteger(trades)) throw new Error(`rows[${index}].trades_24h must be an integer`);
    const spreadRaw = row.spread_bps;
    const spread = spreadRaw === null ? null : nonNegative(Number(spreadRaw), `rows[${index}].spread_bps`);
    return {
      symbol,
      last_price: lastPrice,
      quote_volume_24h: quoteVolume,
      trades_24h: trades,
      price_change_pct_24h: finite(Number(row.price_change_pct_24h), `rows[${index}].price_change_pct_24h`),
      range_pct_24h: nonNegative(Number(row.range_pct_24h), `rows[${index}].range_pct_24h`),
      spread_bps: spread,
      liquidity_rank: score01(Number(row.liquidity_rank), `rows[${index}].liquidity_rank`),
      activity_rank: score01(Number(row.activity_rank), `rows[${index}].activity_rank`),
      volatility_rank: score01(Number(row.volatility_rank), `rows[${index}].volatility_rank`),
      momentum_rank: score01(Number(row.momentum_rank), `rows[${index}].momentum_rank`),
      spread_quality: score01(Number(row.spread_quality), `rows[${index}].spread_quality`),
    };
  });
  rows.sort((a, b) => a.symbol.localeCompare(b.symbol));
  return { schema_version: EMERGENT_MOVER_FRAME_SCHEMA, observed_at: observedAt, source, rows, shadow_only: true };
}

function directionOf(value: number): -1 | 0 | 1 {
  if (value > 0) return 1;
  if (value < 0) return -1;
  return 0;
}

function topReasons(values: Array<{ label: string; rank: number }>): string[] {
  return [...values]
    .filter((item) => item.rank > 0)
    .sort((a, b) => b.rank - a.rank || a.label.localeCompare(b.label))
    .slice(0, 3)
    .map((item) => item.label);
}

export function buildEmergentMoverReport(
  previous: EmergentMoverFrame | null,
  current: EmergentMoverFrame,
  config: EmergentMoverConfig = DEFAULT_EMERGENT_MOVER_CONFIG,
): EmergentMoverReport {
  const now = parseEmergentMoverFrame(current);
  if (!Number.isInteger(config.max_candidates) || config.max_candidates <= 0) {
    throw new Error("max_candidates must be a positive integer");
  }
  if (!Number.isFinite(config.max_comparison_age_ms) || config.max_comparison_age_ms <= 0) {
    throw new Error("max_comparison_age_ms must be positive");
  }

  const baseReport = {
    schema_version: EMERGENT_MOVER_REPORT_SCHEMA,
    source: now.source,
    observed_at: now.observed_at,
    measurement_notes: [
      "research_attention_only_no_trade_action",
      "24h_quote_volume_and_trade_count_deltas_are_rolling_window_displacement_not_interval_flow",
      "newly_observed_symbols_require_a_later_frame_before_mover_ranking",
    ],
    evidence_class: "PROSPECTIVE_DEVELOPMENT_SHADOW" as const,
    shadow_only: true as const,
  };

  if (previous === null) {
    return {
      ...baseReport,
      baseline_observed_at: null,
      comparison_age_ms: null,
      comparable: false,
      comparison_issue: "no_baseline",
      compared_symbol_count: 0,
      newly_observed_symbols: [],
      disappeared_symbols: [],
      candidates: [],
    };
  }

  const before = parseEmergentMoverFrame(previous);
  if (before.source !== now.source) throw new Error("emergent frames must have the same source");
  const beforeMs = validTimestamp(before.observed_at, "baseline observed_at");
  const nowMs = validTimestamp(now.observed_at, "current observed_at");
  if (nowMs <= beforeMs) throw new Error("emergent frames must be chronological");
  const ageMs = nowMs - beforeMs;
  if (ageMs > config.max_comparison_age_ms) {
    return {
      ...baseReport,
      baseline_observed_at: before.observed_at,
      comparison_age_ms: ageMs,
      comparable: false,
      comparison_issue: "comparison_too_old",
      compared_symbol_count: 0,
      newly_observed_symbols: [],
      disappeared_symbols: [],
      candidates: [],
    };
  }

  const previousBySymbol = new Map(before.rows.map((row) => [row.symbol, row]));
  const currentBySymbol = new Map(now.rows.map((row) => [row.symbol, row]));
  const commonSymbols = now.rows.map((row) => row.symbol).filter((symbol) => previousBySymbol.has(symbol));
  const newlyObserved = now.rows.map((row) => row.symbol).filter((symbol) => !previousBySymbol.has(symbol)).sort();
  const disappeared = before.rows.map((row) => row.symbol).filter((symbol) => !currentBySymbol.has(symbol)).sort();

  const featureRows = commonSymbols.map((symbol) => {
    const prev = previousBySymbol.get(symbol)!;
    const curr = currentBySymbol.get(symbol)!;
    const priceChangeDelta = curr.price_change_pct_24h - prev.price_change_pct_24h;
    return {
      symbol,
      features: {
        price_change_delta_pct: priceChangeDelta,
        price_impulse_abs_pct: Math.abs(priceChangeDelta),
        range_delta_pct: curr.range_pct_24h - prev.range_pct_24h,
        liquidity_rank_delta: curr.liquidity_rank - prev.liquidity_rank,
        activity_rank_delta: curr.activity_rank - prev.activity_rank,
        volatility_rank_delta: curr.volatility_rank - prev.volatility_rank,
        momentum_rank_delta: curr.momentum_rank - prev.momentum_rank,
        quote_volume_log_delta: Math.log1p(curr.quote_volume_24h) - Math.log1p(prev.quote_volume_24h),
        trades_log_delta: Math.log1p(curr.trades_24h) - Math.log1p(prev.trades_24h),
        current_spread_bps: curr.spread_bps,
        current_spread_quality: curr.spread_quality,
      } satisfies EmergentMoverFeatures,
    };
  });

  const priceRanks = positivePercentileRanks(featureRows.map((row) => row.features.price_impulse_abs_pct));
  const momentumRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.momentum_rank_delta)));
  const volatilityRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.volatility_rank_delta)));
  const activityRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.activity_rank_delta)));
  const liquidityRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.liquidity_rank_delta)));
  const volumeRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.quote_volume_log_delta)));
  const tradeRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.trades_log_delta)));

  const candidates = featureRows.map((row, index) => {
    const contributionRanks = {
      price_impulse: priceRanks[index],
      momentum_acceleration: momentumRanks[index],
      volatility_acceleration: volatilityRanks[index],
      activity_acceleration: activityRanks[index],
      liquidity_acceleration: liquidityRanks[index],
      rolling_quote_volume_gain: volumeRanks[index],
      rolling_trade_count_gain: tradeRanks[index],
    };
    const values = Object.values(contributionRanks);
    const attentionScore = values.reduce((sum, value) => sum + value, 0) / values.length;
    const reasons = topReasons([
      { label: "short-horizon price displacement", rank: contributionRanks.price_impulse },
      { label: "momentum rank acceleration", rank: contributionRanks.momentum_acceleration },
      { label: "range/volatility rank acceleration", rank: contributionRanks.volatility_acceleration },
      { label: "trade-activity rank acceleration", rank: contributionRanks.activity_acceleration },
      { label: "liquidity rank acceleration", rank: contributionRanks.liquidity_acceleration },
      { label: "24h rolling quote-volume window gained", rank: contributionRanks.rolling_quote_volume_gain },
      { label: "24h rolling trade-count window gained", rank: contributionRanks.rolling_trade_count_gain },
    ]);
    return {
      rank: 0,
      symbol: row.symbol,
      attention_score: attentionScore,
      observed_change_direction: directionOf(row.features.price_change_delta_pct),
      features: row.features,
      contribution_ranks: contributionRanks,
      reasons,
      evidence_class: "PROSPECTIVE_DEVELOPMENT_SHADOW" as const,
      shadow_only: true as const,
    };
  });

  candidates.sort((a, b) =>
    b.attention_score - a.attention_score ||
    b.features.price_impulse_abs_pct - a.features.price_impulse_abs_pct ||
    a.symbol.localeCompare(b.symbol)
  );

  const selected = candidates
    .filter((candidate) => candidate.attention_score > 0)
    .slice(0, config.max_candidates)
    .map((candidate, index) => ({ ...candidate, rank: index + 1 }));

  return {
    ...baseReport,
    baseline_observed_at: before.observed_at,
    comparison_age_ms: ageMs,
    comparable: true,
    comparison_issue: null,
    compared_symbol_count: commonSymbols.length,
    newly_observed_symbols: newlyObserved,
    disappeared_symbols: disappeared,
    candidates: selected,
  };
}

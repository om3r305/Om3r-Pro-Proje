// Brian 2026 Emergent Mover scout.
// Observation only: research attention, never BUY/SELL/order/target-weight output.
//
// `last_price` across consecutive prospective frames is used for the actual short-horizon
// price displacement. Binance 24h price-change/volume/trade-count fields are rolling-window
// measurements; their deltas are diagnostics and are never mislabeled as interval flow.

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
  short_return_pct: number;
  price_impulse_abs_pct: number;
  rolling_price_change_delta_pct_24h: number;
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
  // Data-quality guard: never compare to an arbitrarily old market regime after an outage.
  max_comparison_age_ms: 30 * 60 * 1000,
});

function requiredNumber(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${name} must be a finite number`);
  }
  return value;
}

function nonNegative(value: unknown, name: string): number {
  const out = requiredNumber(value, name);
  if (out < 0) throw new Error(`${name} cannot be negative`);
  return out;
}

function score01(value: unknown, name: string): number {
  const out = requiredNumber(value, name);
  if (out < 0 || out > 1) throw new Error(`${name} must be in [0,1]`);
  return out;
}

function timestampMs(value: unknown, name: string): number {
  if (typeof value !== "string" || !value.trim()) throw new Error(`${name} is required`);
  const ms = Date.parse(value);
  if (!Number.isFinite(ms)) throw new Error(`${name} must be a valid timestamp`);
  return ms;
}

function symbolValue(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) throw new Error("symbol is required");
  return value.trim().toUpperCase();
}

/** Tie-aware [0,1] percentile ranks. Equal values get the same average rank. */
export function percentileRanks(values: number[]): number[] {
  values.forEach((value, i) => requiredNumber(value, `values[${i}]`));
  if (!values.length) return [];
  if (values.length === 1) return [1];
  const order = values.map((value, index) => ({ value, index }))
    .sort((a, b) => a.value - b.value || a.index - b.index);
  const out = new Array<number>(values.length).fill(0);
  const denominator = values.length - 1;
  for (let start = 0; start < order.length;) {
    let end = start;
    while (end + 1 < order.length && order[end + 1].value === order[start].value) end++;
    const rank = ((start + end) / 2) / denominator;
    for (let i = start; i <= end; i++) out[order[i].index] = rank;
    start = end + 1;
  }
  return out;
}

/** Non-positive acceleration contributes exactly zero; positive observations are tie-aware ranked. */
export function positivePercentileRanks(values: number[]): number[] {
  values.forEach((value, i) => requiredNumber(value, `values[${i}]`));
  const out = new Array<number>(values.length).fill(0);
  const positive = values.map((value, index) => ({ value, index }))
    .filter((item) => item.value > 0)
    .sort((a, b) => a.value - b.value || a.index - b.index);
  if (!positive.length) return out;
  for (let start = 0; start < positive.length;) {
    let end = start;
    while (end + 1 < positive.length && positive[end + 1].value === positive[start].value) end++;
    const averageOrdinal = ((start + 1) + (end + 1)) / 2;
    const rank = averageOrdinal / positive.length;
    for (let i = start; i <= end; i++) out[positive[i].index] = rank;
    start = end + 1;
  }
  return out;
}

export function buildEmergentMoverFrame(
  rows: EmergentMarketRow[],
  options: { observed_at: string; source: string },
): EmergentMoverFrame {
  timestampMs(options.observed_at, "observed_at");
  if (typeof options.source !== "string" || !options.source.trim()) throw new Error("source is required");
  const seen = new Set<string>();
  const normalized = rows.map((input, i) => {
    if (!input || typeof input !== "object") throw new Error(`rows[${i}] must be an object`);
    const symbol = symbolValue(input.symbol);
    if (seen.has(symbol)) throw new Error(`duplicate symbol: ${symbol}`);
    seen.add(symbol);
    const last = requiredNumber(input.last_price, `rows[${i}].last_price`);
    if (!(last > 0)) throw new Error(`rows[${i}].last_price must be positive`);
    const volume = nonNegative(input.quote_volume_24h, `rows[${i}].quote_volume_24h`);
    const trades = nonNegative(input.trades_24h, `rows[${i}].trades_24h`);
    if (!Number.isInteger(trades)) throw new Error(`rows[${i}].trades_24h must be an integer`);
    const change = requiredNumber(input.price_change_pct_24h, `rows[${i}].price_change_pct_24h`);
    const high = nonNegative(input.high_price_24h, `rows[${i}].high_price_24h`);
    const low = nonNegative(input.low_price_24h, `rows[${i}].low_price_24h`);
    if (high < low) throw new Error(`rows[${i}] high cannot be below low`);
    const spread = input.spread_bps === null
      ? null
      : nonNegative(input.spread_bps, `rows[${i}].spread_bps`);
    return { symbol, last, volume, trades, change, range: 100 * (high - low) / last, spread };
  }).sort((a, b) => a.symbol.localeCompare(b.symbol));

  const liquidity = percentileRanks(normalized.map((row) => Math.log1p(row.volume)));
  const activity = percentileRanks(normalized.map((row) => Math.log1p(row.trades)));
  const volatility = percentileRanks(normalized.map((row) => row.range));
  const momentum = percentileRanks(normalized.map((row) => Math.abs(row.change)));
  return {
    schema_version: EMERGENT_MOVER_FRAME_SCHEMA,
    observed_at: options.observed_at,
    source: options.source.trim(),
    rows: normalized.map((row, i) => ({
      symbol: row.symbol,
      last_price: row.last,
      quote_volume_24h: row.volume,
      trades_24h: row.trades,
      price_change_pct_24h: row.change,
      range_pct_24h: row.range,
      spread_bps: row.spread,
      liquidity_rank: liquidity[i],
      activity_rank: activity[i],
      volatility_rank: volatility[i],
      momentum_rank: momentum[i],
      spread_quality: row.spread === null ? 0.5 : 1 / (1 + row.spread / 10),
    })),
    shadow_only: true,
  };
}

/** Re-validate persisted JSON. Numeric strings/nulls are deliberately rejected rather than coerced. */
export function parseEmergentMoverFrame(value: unknown): EmergentMoverFrame {
  if (!value || typeof value !== "object") throw new Error("emergent frame must be an object");
  const raw = value as Record<string, unknown>;
  if (raw.schema_version !== EMERGENT_MOVER_FRAME_SCHEMA) throw new Error("unsupported emergent frame schema");
  if (raw.shadow_only !== true) throw new Error("emergent frame must be shadow-only");
  timestampMs(raw.observed_at, "observed_at");
  if (typeof raw.source !== "string" || !raw.source.trim()) throw new Error("source is required");
  if (!Array.isArray(raw.rows)) throw new Error("rows must be an array");
  const seen = new Set<string>();
  const rows: EmergentMoverStateRow[] = raw.rows.map((item, i) => {
    if (!item || typeof item !== "object") throw new Error(`rows[${i}] must be an object`);
    const row = item as Record<string, unknown>;
    const symbol = symbolValue(row.symbol);
    if (seen.has(symbol)) throw new Error(`duplicate symbol: ${symbol}`);
    seen.add(symbol);
    const last = requiredNumber(row.last_price, `rows[${i}].last_price`);
    if (!(last > 0)) throw new Error(`rows[${i}].last_price must be positive`);
    const trades = nonNegative(row.trades_24h, `rows[${i}].trades_24h`);
    if (!Number.isInteger(trades)) throw new Error(`rows[${i}].trades_24h must be an integer`);
    return {
      symbol,
      last_price: last,
      quote_volume_24h: nonNegative(row.quote_volume_24h, `rows[${i}].quote_volume_24h`),
      trades_24h: trades,
      price_change_pct_24h: requiredNumber(row.price_change_pct_24h, `rows[${i}].price_change_pct_24h`),
      range_pct_24h: nonNegative(row.range_pct_24h, `rows[${i}].range_pct_24h`),
      spread_bps: row.spread_bps === null ? null : nonNegative(row.spread_bps, `rows[${i}].spread_bps`),
      liquidity_rank: score01(row.liquidity_rank, `rows[${i}].liquidity_rank`),
      activity_rank: score01(row.activity_rank, `rows[${i}].activity_rank`),
      volatility_rank: score01(row.volatility_rank, `rows[${i}].volatility_rank`),
      momentum_rank: score01(row.momentum_rank, `rows[${i}].momentum_rank`),
      spread_quality: score01(row.spread_quality, `rows[${i}].spread_quality`),
    };
  }).sort((a, b) => a.symbol.localeCompare(b.symbol));
  return {
    schema_version: EMERGENT_MOVER_FRAME_SCHEMA,
    observed_at: raw.observed_at as string,
    source: raw.source.trim(),
    rows,
    shadow_only: true,
  };
}

function direction(value: number): -1 | 0 | 1 {
  return value > 0 ? 1 : value < 0 ? -1 : 0;
}

function topReasons(items: Array<{ label: string; rank: number }>): string[] {
  return items.filter((item) => item.rank > 0)
    .sort((a, b) => b.rank - a.rank || a.label.localeCompare(b.label))
    .slice(0, 3)
    .map((item) => item.label);
}

/**
 * Relative-rank movement can strengthen evidence, but it cannot create a mover by itself.
 * At least one observable on the symbol itself must change in an emergence-like direction.
 */
export function hasSelfEmergenceEvidence(features: EmergentMoverFeatures): boolean {
  return features.price_impulse_abs_pct > 0 ||
    features.range_delta_pct > 0 ||
    features.quote_volume_log_delta > 0 ||
    features.trades_log_delta > 0;
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
  const base = {
    schema_version: EMERGENT_MOVER_REPORT_SCHEMA,
    source: now.source,
    observed_at: now.observed_at,
    measurement_notes: [
      "research_attention_only_no_trade_action",
      "short_return_uses_consecutive_frame_last_price",
      "relative_rank_change_cannot_create_candidate_without_symbol_self_evidence",
      "24h_price_change_quote_volume_and_trade_count_deltas_are_rolling_window_displacement_not_interval_flow",
      "newly_observed_symbols_require_a_later_frame_before_mover_ranking",
    ],
    evidence_class: "PROSPECTIVE_DEVELOPMENT_SHADOW" as const,
    shadow_only: true as const,
  };
  if (previous === null) {
    return {
      ...base,
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
  const beforeMs = timestampMs(before.observed_at, "baseline observed_at");
  const nowMs = timestampMs(now.observed_at, "current observed_at");
  if (nowMs <= beforeMs) throw new Error("emergent frames must be chronological");
  const ageMs = nowMs - beforeMs;
  if (ageMs > config.max_comparison_age_ms) {
    return {
      ...base,
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

  const oldMap = new Map(before.rows.map((row) => [row.symbol, row]));
  const newMap = new Map(now.rows.map((row) => [row.symbol, row]));
  const common = now.rows.map((row) => row.symbol).filter((symbol) => oldMap.has(symbol));
  const newlyObserved = now.rows.map((row) => row.symbol).filter((symbol) => !oldMap.has(symbol)).sort();
  const disappeared = before.rows.map((row) => row.symbol).filter((symbol) => !newMap.has(symbol)).sort();

  const featureRows = common.map((symbol) => {
    const oldRow = oldMap.get(symbol)!;
    const newRow = newMap.get(symbol)!;
    const shortReturn = 100 * (newRow.last_price / oldRow.last_price - 1);
    const features: EmergentMoverFeatures = {
      short_return_pct: shortReturn,
      price_impulse_abs_pct: Math.abs(shortReturn),
      rolling_price_change_delta_pct_24h: newRow.price_change_pct_24h - oldRow.price_change_pct_24h,
      range_delta_pct: newRow.range_pct_24h - oldRow.range_pct_24h,
      liquidity_rank_delta: newRow.liquidity_rank - oldRow.liquidity_rank,
      activity_rank_delta: newRow.activity_rank - oldRow.activity_rank,
      volatility_rank_delta: newRow.volatility_rank - oldRow.volatility_rank,
      momentum_rank_delta: newRow.momentum_rank - oldRow.momentum_rank,
      quote_volume_log_delta: Math.log1p(newRow.quote_volume_24h) - Math.log1p(oldRow.quote_volume_24h),
      trades_log_delta: Math.log1p(newRow.trades_24h) - Math.log1p(oldRow.trades_24h),
      current_spread_bps: newRow.spread_bps,
      current_spread_quality: newRow.spread_quality,
    };
    return { symbol, features };
  });

  const priceRanks = positivePercentileRanks(featureRows.map((row) => row.features.price_impulse_abs_pct));
  const momentumRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.momentum_rank_delta)));
  const volatilityRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.volatility_rank_delta)));
  const activityRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.activity_rank_delta)));
  const liquidityRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.liquidity_rank_delta)));
  const volumeRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.quote_volume_log_delta)));
  const tradeRanks = positivePercentileRanks(featureRows.map((row) => Math.max(0, row.features.trades_log_delta)));

  const ranked = featureRows.map((row, i) => {
    const contribution_ranks = {
      price_impulse: priceRanks[i],
      momentum_acceleration: momentumRanks[i],
      volatility_acceleration: volatilityRanks[i],
      activity_acceleration: activityRanks[i],
      liquidity_acceleration: liquidityRanks[i],
      rolling_quote_volume_gain: volumeRanks[i],
      rolling_trade_count_gain: tradeRanks[i],
    };
    const contributions = Object.values(contribution_ranks);
    const attention = contributions.reduce((sum, value) => sum + value, 0) / contributions.length;
    return {
      rank: 0,
      symbol: row.symbol,
      attention_score: attention,
      observed_change_direction: direction(row.features.short_return_pct),
      features: row.features,
      contribution_ranks,
      reasons: topReasons([
        { label: "short-horizon last-price displacement", rank: contribution_ranks.price_impulse },
        { label: "momentum rank acceleration", rank: contribution_ranks.momentum_acceleration },
        { label: "range/volatility rank acceleration", rank: contribution_ranks.volatility_acceleration },
        { label: "trade-activity rank acceleration", rank: contribution_ranks.activity_acceleration },
        { label: "liquidity rank acceleration", rank: contribution_ranks.liquidity_acceleration },
        { label: "24h rolling quote-volume window gained", rank: contribution_ranks.rolling_quote_volume_gain },
        { label: "24h rolling trade-count window gained", rank: contribution_ranks.rolling_trade_count_gain },
      ]),
      evidence_class: "PROSPECTIVE_DEVELOPMENT_SHADOW" as const,
      shadow_only: true as const,
    };
  }).sort((a, b) =>
    b.attention_score - a.attention_score ||
    b.features.price_impulse_abs_pct - a.features.price_impulse_abs_pct ||
    a.symbol.localeCompare(b.symbol)
  );

  const candidates = ranked
    .filter((candidate) => candidate.attention_score > 0 && hasSelfEmergenceEvidence(candidate.features))
    .slice(0, config.max_candidates)
    .map((candidate, i) => ({ ...candidate, rank: i + 1 }));

  return {
    ...base,
    baseline_observed_at: before.observed_at,
    comparison_age_ms: ageMs,
    comparable: true,
    comparison_issue: null,
    compared_symbol_count: common.length,
    newly_observed_symbols: newlyObserved,
    disappeared_symbols: disappeared,
    candidates,
  };
}

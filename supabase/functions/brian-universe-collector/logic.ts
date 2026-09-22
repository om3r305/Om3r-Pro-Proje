// Pure, side-effect-free filtering/scoring logic for brian-universe-collector, split out of
// index.ts so it can be exercised directly by logic.test.ts without a live network/Supabase dependency.

export interface UniverseConfig {
  quote_asset: string;
  min_quote_volume: number;
  min_trades_24h: number;
  min_price: number;
  top_n: number;
  max_abs_change_pct: number;
  excluded_base_assets: string[];
}

export const CONFIG: UniverseConfig = {
  quote_asset: "USDT",
  min_quote_volume: 5_000_000,
  min_trades_24h: 1_000,
  min_price: 1e-8,
  top_n: 50,
  max_abs_change_pct: 200,
  excluded_base_assets: ["USDT", "USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "EUR", "TRY"],
};

export function finiteNumber(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

export function clip01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/** Maps each value to its rank fraction in [0,1] (0 = smallest, 1 = largest); ties break by input order. */
export function rankPercentiles(values: number[]): number[] {
  if (!values.length) return [];
  const indexed = values.map((value, index) => ({ value, index }));
  indexed.sort((a, b) => a.value - b.value || a.index - b.index);
  const out = new Array(values.length).fill(0);
  const denominator = Math.max(1, values.length - 1);
  indexed.forEach((item, rank) => { out[item.index] = rank / denominator; });
  return out;
}

/** Indexes a Binance array-of-objects response by its `symbol` field. */
export function indexRows(payload: unknown): Map<string, Record<string, unknown>> {
  if (!Array.isArray(payload)) throw new Error("expected Binance array response");
  const out = new Map<string, Record<string, unknown>>();
  for (const row of payload) {
    if (row && typeof row === "object" && "symbol" in row) {
      const symbol = String((row as Record<string, unknown>).symbol ?? "");
      if (symbol) out.set(symbol, row as Record<string, unknown>);
    }
  }
  return out;
}

export interface UniverseRow {
  symbol: string;
  base_asset: string;
  last_price: number;
  quote_volume: number;
  trades_24h: number;
  price_change_pct: number;
  range_pct: number;
  spread_bps: number | null;
}

/** Applies the eligibility gate (status/quote/exclusions/liquidity/trade-count/price sanity) to raw exchange symbol metadata. */
export function buildEligibleRows(
  exchangeSymbols: unknown[],
  tickerMap: Map<string, Record<string, unknown>>,
  bookMap: Map<string, Record<string, unknown>>,
  config: UniverseConfig = CONFIG,
): UniverseRow[] {
  const excluded = new Set(config.excluded_base_assets);
  const rows: UniverseRow[] = [];
  for (const metadataValue of exchangeSymbols) {
    if (!metadataValue || typeof metadataValue !== "object") continue;
    const metadata = metadataValue as Record<string, unknown>;
    const symbol = String(metadata.symbol ?? "");
    const baseAsset = String(metadata.baseAsset ?? "");
    const quoteAsset = String(metadata.quoteAsset ?? "");
    const tickerRow = tickerMap.get(symbol);
    if (!symbol || !baseAsset || !quoteAsset || !tickerRow) continue;
    if (String(metadata.status ?? "") !== "TRADING" || metadata.isSpotTradingAllowed === false) continue;
    if (quoteAsset.toUpperCase() !== config.quote_asset || excluded.has(baseAsset.toUpperCase())) continue;

    const lastPrice = finiteNumber(tickerRow.lastPrice);
    const quoteVolume = finiteNumber(tickerRow.quoteVolume);
    const trades24h = Math.max(0, Math.trunc(finiteNumber(tickerRow.count)));
    const priceChangePct = finiteNumber(tickerRow.priceChangePercent);
    const highPrice = finiteNumber(tickerRow.highPrice);
    const lowPrice = finiteNumber(tickerRow.lowPrice);
    if (lastPrice < config.min_price || quoteVolume < config.min_quote_volume || trades24h < config.min_trades_24h) continue;
    if (highPrice < lowPrice || lowPrice < 0) continue;

    const bookRow = bookMap.get(symbol);
    const bid = bookRow ? finiteNumber(bookRow.bidPrice) : 0;
    const ask = bookRow ? finiteNumber(bookRow.askPrice) : 0;
    const spreadBps = bid > 0 && ask > 0
      ? 10_000 * Math.max(0, ask - bid) / Math.max((ask + bid) / 2, 1e-12)
      : null;
    const rangePct = 100 * Math.max(0, highPrice - lowPrice) / Math.max(lastPrice, 1e-12);
    rows.push({
      symbol, base_asset: baseAsset, last_price: lastPrice, quote_volume: quoteVolume, trades_24h: trades24h,
      price_change_pct: priceChangePct, range_pct: rangePct, spread_bps: spreadBps,
    });
  }
  return rows;
}

export interface UniverseCandidate {
  symbol: string;
  base_asset: string;
  liquidity_score: number;
  activity_score: number;
  volatility_score: number;
  momentum_score: number;
  spread_quality: number;
  radar_score: number;
  quote_volume: number;
  trades_24h: number;
  price_change_pct: number;
  range_pct: number;
  spread_bps: number | null;
  reasons: string[];
}

/** Scores each eligible row on liquidity/activity/volatility/momentum/spread percentile rank, sorted best-first. */
export function scoreCandidates(rows: UniverseRow[], config: UniverseConfig = CONFIG): UniverseCandidate[] {
  const liquidity = rankPercentiles(rows.map((r) => Math.log1p(r.quote_volume)));
  const activity = rankPercentiles(rows.map((r) => Math.log1p(r.trades_24h)));
  const volatility = rankPercentiles(rows.map((r) => r.range_pct));
  const momentum = rankPercentiles(rows.map((r) => Math.min(config.max_abs_change_pct, Math.abs(r.price_change_pct))));

  const candidates = rows.map((row, i) => {
    const spread = row.spread_bps;
    const spreadQuality = spread === null ? 0.50 : 1 / (1 + Math.max(0, spread) / 10);
    const radarScore = clip01(0.34 * liquidity[i] + 0.20 * activity[i] + 0.20 * volatility[i] + 0.16 * momentum[i] + 0.10 * spreadQuality);
    const reasons: string[] = [];
    if (liquidity[i] >= 0.80) reasons.push("high relative liquidity");
    if (activity[i] >= 0.80) reasons.push("high trade activity");
    if (volatility[i] >= 0.80) reasons.push("elevated 24h range");
    if (momentum[i] >= 0.80) reasons.push("large absolute 24h move");
    if (spread !== null && spread <= 5) reasons.push("tight top-of-book spread");
    return {
      symbol: row.symbol, base_asset: row.base_asset, liquidity_score: clip01(liquidity[i]),
      activity_score: clip01(activity[i]), volatility_score: clip01(volatility[i]), momentum_score: clip01(momentum[i]),
      spread_quality: clip01(spreadQuality), radar_score: radarScore, quote_volume: row.quote_volume,
      trades_24h: row.trades_24h, price_change_pct: row.price_change_pct, range_pct: row.range_pct,
      spread_bps: spread, reasons,
    };
  });
  candidates.sort((a, b) => b.radar_score - a.radar_score || b.quote_volume - a.quote_volume || a.symbol.localeCompare(b.symbol));
  return candidates;
}

export interface EligibilityDelta {
  comparable: boolean;
  newlyObserved: string[];
  disappeared: string[];
}

/** Diffs this run's eligible symbol set against the previous snapshot's, when one exists. */
export function diffEligibility(previousEligible: string[] | null, eligibleSymbols: string[]): EligibilityDelta {
  const before = new Set(previousEligible ?? []);
  const after = new Set(eligibleSymbols);
  const comparable = previousEligible !== null;
  return {
    comparable,
    newlyObserved: comparable ? eligibleSymbols.filter((x) => !before.has(x)) : [],
    disappeared: comparable ? [...before].filter((x) => !after.has(x)).sort() : [],
  };
}

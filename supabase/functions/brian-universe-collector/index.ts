import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const RAW_BUCKET = "brian-intelligence-raw";
const PROVIDER = "binance_public";
const MIN_INTERVAL_SECONDS = 780;
const CONFIG = {
  quote_asset: "USDT",
  min_quote_volume: 5_000_000,
  min_trades_24h: 1_000,
  min_price: 1e-8,
  top_n: 50,
  max_abs_change_pct: 200,
  excluded_base_assets: ["USDT", "USDC", "FDUSD", "TUSD", "DAI", "BUSD", "USDP", "EUR", "TRY"],
};

const EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo";
const TICKER_24H = "https://api.binance.com/api/v3/ticker/24hr";
const BOOK_TICKER = "https://api.binance.com/api/v3/ticker/bookTicker";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

function finiteNumber(value: unknown, fallback = 0): number {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clip01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function utf8(value: string): Uint8Array {
  return new TextEncoder().encode(value);
}

async function sha256Hex(value: string | Uint8Array): Promise<string> {
  const bytes = typeof value === "string" ? utf8(value) : value;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchJson(url: string, required: boolean): Promise<{payload: unknown; observedAt: string; degraded: boolean}> {
  try {
    const response = await fetch(url, {
      method: "GET",
      headers: { "accept": "application/json", "user-agent": "Brian-2026-Shadow-Research/2.0" },
      signal: AbortSignal.timeout(8_000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    return { payload, observedAt: new Date().toISOString(), degraded: false };
  } catch (error) {
    if (required) throw error;
    return { payload: [], observedAt: new Date().toISOString(), degraded: true };
  }
}

async function persistRaw(recordType: string, payload: unknown, observedAt: string, provenanceUri: string): Promise<string> {
  const canonical = JSON.stringify(payload);
  const bytes = utf8(canonical);
  const payloadHash = await sha256Hex(bytes);
  const compressed = gzip(bytes, { level: 6 });
  const date = observedAt.slice(0, 10);
  const path = `${PROVIDER}/${recordType}/${date}/${payloadHash}.json.gz`;

  const upload = await supabase.storage.from(RAW_BUCKET).upload(path, compressed, {
    contentType: "application/gzip",
    upsert: false,
    cacheControl: "31536000",
  });
  if (upload.error) {
    const status = String((upload.error as {statusCode?: string | number}).statusCode ?? "");
    const msg = String(upload.error.message ?? "");
    if (status !== "409" && !msg.toLowerCase().includes("exist") && !msg.toLowerCase().includes("duplicate")) {
      throw upload.error;
    }
  }

  const capturedAt = new Date().toISOString();
  const captureId = await sha256Hex(`${PROVIDER}|${recordType}|${observedAt}|${payloadHash}`);
  const payloadPointer = {
    storage_bucket: RAW_BUCKET,
    storage_path: path,
    uncompressed_byte_length: bytes.byteLength,
    compressed_byte_length: compressed.byteLength,
    content_type: "application/json",
    content_encoding: "gzip",
  };

  const insert = await supabase.from("brian_raw_captures").insert({
    capture_id: captureId,
    provider: PROVIDER,
    record_type: recordType,
    observed_at: observedAt,
    captured_at: capturedAt,
    provenance_uri: provenanceUri,
    payload_hash: payloadHash,
    payload: payloadPointer,
  });
  if (insert.error) throw insert.error;
  return captureId;
}

function rankPercentiles(values: number[]): number[] {
  if (!values.length) return [];
  const indexed = values.map((value, index) => ({ value, index }));
  indexed.sort((a, b) => a.value - b.value || a.index - b.index);
  const out = new Array(values.length).fill(0);
  const denominator = Math.max(1, values.length - 1);
  indexed.forEach((item, rank) => { out[item.index] = rank / denominator; });
  return out;
}

function indexRows(payload: unknown): Map<string, Record<string, unknown>> {
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

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return jsonResponse({ error: "POST required" }, 405);

  try {
    const last = await supabase
      .from("brian_universe_snapshots")
      .select("observed_at,candidates")
      .order("observed_at", { ascending: false })
      .limit(1)
      .maybeSingle();
    if (last.error) throw last.error;
    if (last.data?.observed_at) {
      const ageSeconds = (Date.now() - Date.parse(last.data.observed_at)) / 1000;
      if (Number.isFinite(ageSeconds) && ageSeconds < MIN_INTERVAL_SECONDS) {
        return jsonResponse({ status: "SKIPPED_RATE_GUARD", age_seconds: Math.max(0, ageSeconds), shadow_only: true });
      }
    }

    const exchange = await fetchJson(EXCHANGE_INFO, true);
    const ticker = await fetchJson(TICKER_24H, true);
    const book = await fetchJson(BOOK_TICKER, false);
    const snapshotObservedAt = new Date().toISOString();

    const exchangeObj = exchange.payload as Record<string, unknown>;
    if (!exchangeObj || typeof exchangeObj !== "object" || !Array.isArray(exchangeObj.symbols)) {
      throw new Error("invalid Binance exchangeInfo response");
    }
    const tickerMap = indexRows(ticker.payload);
    const bookMap = book.degraded ? new Map<string, Record<string, unknown>>() : indexRows(book.payload);

    const captureIds: string[] = [];
    captureIds.push(await persistRaw("exchange_info", exchange.payload, exchange.observedAt, EXCHANGE_INFO));
    captureIds.push(await persistRaw("ticker_24h", ticker.payload, ticker.observedAt, TICKER_24H));
    if (!book.degraded) captureIds.push(await persistRaw("book_ticker", book.payload, book.observedAt, BOOK_TICKER));

    const excluded = new Set(CONFIG.excluded_base_assets);
    const rows: Array<Record<string, number | string | null>> = [];
    for (const metadataValue of exchangeObj.symbols as unknown[]) {
      if (!metadataValue || typeof metadataValue !== "object") continue;
      const metadata = metadataValue as Record<string, unknown>;
      const symbol = String(metadata.symbol ?? "");
      const baseAsset = String(metadata.baseAsset ?? "");
      const quoteAsset = String(metadata.quoteAsset ?? "");
      const tickerRow = tickerMap.get(symbol);
      if (!symbol || !baseAsset || !quoteAsset || !tickerRow) continue;
      if (String(metadata.status ?? "") !== "TRADING" || metadata.isSpotTradingAllowed === false) continue;
      if (quoteAsset.toUpperCase() !== CONFIG.quote_asset || excluded.has(baseAsset.toUpperCase())) continue;

      const lastPrice = finiteNumber(tickerRow.lastPrice);
      const quoteVolume = finiteNumber(tickerRow.quoteVolume);
      const trades24h = Math.max(0, Math.trunc(finiteNumber(tickerRow.count)));
      const priceChangePct = finiteNumber(tickerRow.priceChangePercent);
      const highPrice = finiteNumber(tickerRow.highPrice);
      const lowPrice = finiteNumber(tickerRow.lowPrice);
      if (lastPrice < CONFIG.min_price || quoteVolume < CONFIG.min_quote_volume || trades24h < CONFIG.min_trades_24h) continue;
      if (highPrice < lowPrice || lowPrice < 0) continue;

      const bookRow = bookMap.get(symbol);
      const bid = bookRow ? finiteNumber(bookRow.bidPrice) : 0;
      const ask = bookRow ? finiteNumber(bookRow.askPrice) : 0;
      const spreadBps = bid > 0 && ask > 0
        ? 10_000 * Math.max(0, ask - bid) / Math.max((ask + bid) / 2, 1e-12)
        : null;
      const rangePct = 100 * Math.max(0, highPrice - lowPrice) / Math.max(lastPrice, 1e-12);
      rows.push({ symbol, base_asset: baseAsset, last_price: lastPrice, quote_volume: quoteVolume,
        trades_24h: trades24h, price_change_pct: priceChangePct, range_pct: rangePct, spread_bps: spreadBps });
    }

    const liquidity = rankPercentiles(rows.map((r) => Math.log1p(Number(r.quote_volume))));
    const activity = rankPercentiles(rows.map((r) => Math.log1p(Number(r.trades_24h))));
    const volatility = rankPercentiles(rows.map((r) => Number(r.range_pct)));
    const momentum = rankPercentiles(rows.map((r) => Math.min(CONFIG.max_abs_change_pct, Math.abs(Number(r.price_change_pct)))));

    const candidates = rows.map((row, i) => {
      const spread = row.spread_bps === null ? null : Number(row.spread_bps);
      const spreadQuality = spread === null ? 0.50 : 1 / (1 + Math.max(0, spread) / 10);
      const radarScore = clip01(0.34 * liquidity[i] + 0.20 * activity[i] + 0.20 * volatility[i] + 0.16 * momentum[i] + 0.10 * spreadQuality);
      const reasons: string[] = [];
      if (liquidity[i] >= 0.80) reasons.push("high relative liquidity");
      if (activity[i] >= 0.80) reasons.push("high trade activity");
      if (volatility[i] >= 0.80) reasons.push("elevated 24h range");
      if (momentum[i] >= 0.80) reasons.push("large absolute 24h move");
      if (spread !== null && spread <= 5) reasons.push("tight top-of-book spread");
      return {
        symbol: row.symbol,
        base_asset: row.base_asset,
        liquidity_score: clip01(liquidity[i]),
        activity_score: clip01(activity[i]),
        volatility_score: clip01(volatility[i]),
        momentum_score: clip01(momentum[i]),
        spread_quality: clip01(spreadQuality),
        radar_score: radarScore,
        quote_volume: row.quote_volume,
        trades_24h: row.trades_24h,
        price_change_pct: row.price_change_pct,
        range_pct: row.range_pct,
        spread_bps: spread,
        reasons,
      };
    });
    candidates.sort((a, b) => b.radar_score - a.radar_score || Number(b.quote_volume) - Number(a.quote_volume) || String(a.symbol).localeCompare(String(b.symbol)));
    const selected = candidates.slice(0, CONFIG.top_n);
    const eligibleSymbols = rows.map((r) => String(r.symbol)).sort();

    const previousPayload = (last.data?.candidates && typeof last.data.candidates === "object")
      ? last.data.candidates as Record<string, unknown>
      : null;
    const previousEligible = previousPayload && Array.isArray(previousPayload.eligible_symbols)
      ? previousPayload.eligible_symbols.map(String)
      : null;
    const before = new Set(previousEligible ?? []);
    const after = new Set(eligibleSymbols);
    const comparable = previousEligible !== null;
    const newlyObserved = comparable ? eligibleSymbols.filter((x) => !before.has(x)) : [];
    const disappeared = comparable ? [...before].filter((x) => !after.has(x)).sort() : [];

    const snapshotPayload = {
      schema_version: "brian.universe-snapshot.v1",
      collector_version: "2",
      source: PROVIDER,
      config: CONFIG,
      eligible_symbols: eligibleSymbols,
      rejected_count: (exchangeObj.symbols as unknown[]).length - rows.length,
      degraded_sources: book.degraded ? ["book_ticker"] : [],
      comparable,
      newly_observed_symbols: newlyObserved,
      disappeared_symbols: disappeared,
      candidates: selected,
    };
    const snapshotId = await sha256Hex(`universe|${snapshotObservedAt}|${captureIds.join("|")}`);
    const snapshotInsert = await supabase.from("brian_universe_snapshots").insert({
      snapshot_id: snapshotId,
      provider: PROVIDER,
      observed_at: snapshotObservedAt,
      eligible_count: eligibleSymbols.length,
      candidates: snapshotPayload,
      raw_capture_ids: captureIds,
    });
    if (snapshotInsert.error) throw snapshotInsert.error;

    return jsonResponse({
      status: "CAPTURED",
      snapshot_id: snapshotId,
      observed_at: snapshotObservedAt,
      eligible_count: eligibleSymbols.length,
      top_candidates: selected.slice(0, 10).map((x) => ({ symbol: x.symbol, radar_score: x.radar_score, reasons: x.reasons })),
      newly_observed_symbols: newlyObserved,
      degraded_sources: snapshotPayload.degraded_sources,
      raw_encoding: "gzip",
      shadow_only: true,
    });
  } catch (error) {
    console.error("brian-universe-collector failed", error);
    return jsonResponse({ status: "FAILED_CLOSED", error: String(error instanceof Error ? error.message : error), shadow_only: true }, 500);
  }
});

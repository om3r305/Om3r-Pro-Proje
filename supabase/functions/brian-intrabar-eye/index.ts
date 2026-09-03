import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });

const SCHEMA_VERSION = "brian.phase40-intrabar-reaction.v1";
const EXPERIMENT_ID = "phase40-intrabar-reaction-v1";
const EVIDENCE_CLASS = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const RAW_BUCKET = "brian-intelligence-raw";
const COLLECTOR_ID = "brian-intrabar-eye";
const TOP_N = 50;
const CORE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"] as const;
const KLINE_LIMIT = 32;
const AGG_TRADE_LIMIT = 200;
const MIN_INTERVAL_SECONDS = 50;
const PRIOR_LOOKUP_BATCH = 40;
const FEE_BPS = 10.0;
const SLIPPAGE_BPS = 1.0;
const MIN_SUPPORT_GROUPS = 2;
const MIN_CONSENSUS_SCORE = 0.18;
const OVEREXTENSION_SIGMA = 3.5;

const TEMPLATES = [
  { id: "velocity-micro", family: "price_structure", group: "micro_velocity", ticket: 3.0 },
  { id: "volume-burst-micro", family: "price_structure", group: "micro_volume", ticket: 3.0 },
  { id: "breakout-micro", family: "price_structure", group: "micro_breakout", ticket: 5.0 },
  { id: "reclaim-micro", family: "price_structure", group: "micro_reclaim", ticket: 3.0 },
  { id: "taker-flow-micro", family: "orderbook", group: "micro_taker_flow", ticket: 3.0 },
] as const;

type RadarCandidate = { symbol: string; radar_score?: number; liquidity_score?: number; activity_score?: number; spread_bps?: number | null };
type Book = { bid: number; ask: number; mid: number; spreadBps: number };
type Bar = {
  openTime: number; closeTime: number; open: number; high: number; low: number; close: number;
  baseVolume: number; quoteVolume: number; trades: number; takerBuyQuote: number;
};
type AggTrade = { price: number; qty: number; quote: number; ts: number; buyerMaker: boolean };
type Signal = { direction: number; strength: number; reason: string; metadata?: Record<string, unknown> };
type MarketRow = {
  candidate: RadarCandidate; bars: Bar[]; trades: AggTrade[]; book: Book; degradedAggTrades: boolean;
  sigma: number; current: Bar; baseline: Bar[]; elapsedFraction: number; medianQuoteVolume: number;
  velocityCoverageSeconds: number; return30s: number; previous30sReturn: number; decelerating: boolean;
};
type PriorTick = {
  eye_id: string; starting_equity: string | number; equity_after: string | number; peak_equity_after: string | number;
  max_drawdown_pct_after: number; target_direction: number; observed_mid_price: string | number; observed_at?: string;
};
type Observation = {
  observation_id: string; eye_id: string; template_id: string; asset_id: string; market_domain: string;
  sensor_family: string; horizon: string; independent_group: string; observed_at: string; direction: number;
  strength: number; confidence: number; reliability: number; available: boolean; source_ids: string[];
  reason: string; evidence_class: string; shadow_only: boolean; live_execution: boolean; metadata: Record<string, unknown>;
};
type SignalRow = {
  template: typeof TEMPLATES[number]; assetId: string; candidate: RadarCandidate; market: MarketRow; signal: Signal; eyeId: string;
};

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}
function errorText(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  if (error && typeof error === "object") {
    try { return JSON.stringify(error); } catch { return Object.prototype.toString.call(error); }
  }
  return String(error);
}
function finite(value: unknown, fallback = 0): number { const n = Number(value); return Number.isFinite(n) ? n : fallback; }
function clip(value: number, low = 0, high = 1): number { return Math.max(low, Math.min(high, value)); }
function mean(values: number[]): number { return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0; }
function median(values: number[]): number { if (!values.length) return 0; const x = [...values].sort((a, b) => a - b); const m = Math.floor(x.length / 2); return x.length % 2 ? x[m] : (x[m - 1] + x[m]) / 2; }
function std(values: number[]): number { if (values.length <= 1) return 0; const m = mean(values); return Math.sqrt(mean(values.map((v) => (v - m) ** 2))); }
function sign(value: number): number { return value > 0 ? 1 : value < 0 ? -1 : 0; }
function utf8(text: string): Uint8Array { return new TextEncoder().encode(text); }
async function sha256(value: string | Uint8Array): Promise<string> {
  const bytes = typeof value === "string" ? utf8(value) : value;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}
async function fetchJson(url: string): Promise<unknown> {
  const r = await fetch(url, { headers: { accept: "application/json", "user-agent": "Brian-2026-Intrabar-Eye/1.0" }, signal: AbortSignal.timeout(8_000) });
  if (!r.ok) throw new Error(`public market fetch failed ${r.status}: ${url}`);
  return await r.json();
}
async function mapLimit<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length); let cursor = 0;
  async function worker() { while (true) { const i = cursor++; if (i >= items.length) return; out[i] = await fn(items[i]); } }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, () => worker())); return out;
}

function parseBars(payload: unknown): Bar[] {
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
function parseAggTrades(payload: unknown): AggTrade[] {
  if (!Array.isArray(payload)) return [];
  return payload.map((row) => {
    const r = row as Record<string, unknown>; const price = finite(r.p); const qty = finite(r.q); const ts = finite(r.T);
    return { price, qty, quote: price * qty, ts, buyerMaker: Boolean(r.m) };
  }).filter((r) => r.price > 0 && r.qty > 0 && r.ts > 0).sort((a, b) => a.ts - b.ts);
}
function logReturns(values: number[]): number[] { return values.slice(1).map((v, i) => Math.log(v / values[i])); }
function windowReturn(trades: AggTrade[], observedMs: number, seconds: number, offsetSeconds = 0): number {
  const end = observedMs - offsetSeconds * 1000; const start = end - seconds * 1000;
  const rows = trades.filter((t) => t.ts >= start && t.ts <= end); if (rows.length < 2) return 0;
  return Math.log(rows.at(-1)!.price / rows[0].price);
}
function confidenceFromRadar(candidate: RadarCandidate, book: Book): number {
  const liquidity = clip(finite(candidate.liquidity_score, 0.5)); const activity = clip(finite(candidate.activity_score, 0.5));
  const spreadQuality = 1 / (1 + Math.max(0, book.spreadBps) / 10);
  return clip(0.45 * liquidity + 0.35 * activity + 0.20 * spreadQuality);
}

function signalVelocity(row: MarketRow, observedMs: number): Signal {
  const coverage = row.velocityCoverageSeconds; if (coverage < 5) return { direction: 0, strength: 0, reason: "insufficient live trade velocity coverage", metadata: { coverage_seconds: coverage } };
  const live = windowReturn(row.trades, observedMs, Math.min(60, Math.max(5, coverage)));
  const adaptive = Math.max(0.00010, row.sigma * Math.sqrt(Math.max(5, Math.min(60, coverage)) / 60) * 0.70);
  if (Math.abs(live) < adaptive) return { direction: 0, strength: 0, reason: "live trade velocity below adaptive threshold", metadata: { live_return: live, threshold: adaptive, coverage_seconds: coverage } };
  return { direction: sign(live), strength: clip(Math.abs(live) / (adaptive * 3)), reason: "live trade velocity impulse", metadata: { live_return: live, threshold: adaptive, coverage_seconds: coverage } };
}
function signalVolumeBurst(row: MarketRow): Signal {
  const pace = row.current.quoteVolume / Math.max(row.elapsedFraction, 0.15) / Math.max(row.medianQuoteVolume, 1e-12);
  const body = Math.log(row.book.mid / row.current.open); const threshold = Math.max(0.00015, row.sigma * 0.45);
  if (pace < 1.8 || Math.abs(body) < threshold) return { direction: 0, strength: 0, reason: "partial 1m volume/body not impulsive", metadata: { pace_ratio: pace, body_return: body, threshold } };
  const strength = clip(0.55 * clip(Math.abs(body) / (threshold * 3)) + 0.45 * clip((pace - 1) / 4));
  return { direction: sign(body), strength, reason: "partial 1m relative-volume acceleration", metadata: { pace_ratio: pace, body_return: body, threshold } };
}
function signalBreakout(row: MarketRow): Signal {
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
function signalReclaim(row: MarketRow): Signal {
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
function signalTakerFlow(row: MarketRow): Signal {
  const q = row.current.quoteVolume; if (q <= 0) return { direction: 0, strength: 0, reason: "no partial 1m quote volume" };
  const buyShare = clip(row.current.takerBuyQuote / q); const pace = q / Math.max(row.elapsedFraction, 0.15) / Math.max(row.medianQuoteVolume, 1e-12);
  if (pace < 0.35 || (buyShare >= 0.42 && buyShare <= 0.58)) return { direction: 0, strength: 0, reason: "partial 1m taker flow balanced", metadata: { buy_share: buyShare, pace_ratio: pace } };
  const direction = buyShare > 0.58 ? 1 : -1; const imbalance = Math.abs(buyShare - 0.5) * 2;
  return { direction, strength: clip(imbalance * Math.min(1, 0.5 + pace / 2)), reason: "partial 1m taker-flow imbalance", metadata: { buy_share: buyShare, pace_ratio: pace } };
}

async function persistRaw(payload: unknown, observedAt: string): Promise<string> {
  const canonical = JSON.stringify(payload); const bytes = utf8(canonical); const payloadHash = await sha256(bytes); const compressed = gzip(bytes, { level: 6 });
  const path = `binance_public/phase40_intrabar/${observedAt.slice(0, 10)}/${payloadHash}.json.gz`;
  const upload = await supabase.storage.from(RAW_BUCKET).upload(path, compressed, { contentType: "application/gzip", upsert: false, cacheControl: "31536000" });
  if (upload.error) { const msg = String(upload.error.message ?? "").toLowerCase(); const status = String((upload.error as {statusCode?: string | number}).statusCode ?? ""); if (status !== "409" && !msg.includes("duplicate") && !msg.includes("exist")) throw upload.error; }
  const captureId = await sha256(`binance_public|phase40_intrabar|${observedAt}|${payloadHash}`);
  const ins = await supabase.from("brian_raw_captures").insert({
    capture_id: captureId, provider: "binance_public", record_type: "phase40_intrabar_reaction", observed_at: observedAt, captured_at: new Date().toISOString(),
    provenance_uri: "https://api.binance.com/api/v3/klines?interval=1m + /api/v3/aggTrades + /api/v3/ticker/bookTicker",
    payload_hash: payloadHash, payload: { storage_bucket: RAW_BUCKET, storage_path: path, content_type: "application/json", content_encoding: "gzip", uncompressed_byte_length: bytes.byteLength, compressed_byte_length: compressed.byteLength },
  });
  if (ins.error) throw ins.error; return captureId;
}

function microTick(params: { eyeId: string; templateId: string; assetId: string; observedAt: string; mid: number; spreadBps: number; strength: number; confidence: number; targetDirection: number; startingTicket: number; rawCaptureId: string; prior?: PriorTick; metadata: Record<string, unknown> }): Record<string, unknown> | null {
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

async function recordCollectorRun(startedAt: string, status: "SUCCESS" | "DEGRADED" | "FAILED" | "SKIPPED", observed: number, stored: number, degraded: string[], metadata: Record<string, unknown>, error?: unknown) {
  try {
    const finishedAt = new Date().toISOString(); const runId = await sha256(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
    await supabase.from("brian_collector_runs").insert({ run_id: runId, collector_id: COLLECTOR_ID, started_at: startedAt, finished_at: finishedAt, status, observed_records: observed, stored_records: stored, degraded_sources: degraded, error_class: error ? "INTRABAR_RUNTIME" : null, error_message: error ? errorText(error).slice(0, 2000) : null, evidence_class: EVIDENCE_CLASS, shadow_only: true, live_execution: false, metadata });
  } catch (logError) { console.error("collector run logging failed", errorText(logError)); }
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return jsonResponse({ error: "POST required" }, 405);
  const startedAt = new Date().toISOString();
  try {
    const lastRun = await supabase.from("brian_collector_runs").select("finished_at").eq("collector_id", COLLECTOR_ID).in("status", ["SUCCESS", "DEGRADED"]).order("finished_at", { ascending: false }).limit(1).maybeSingle();
    if (lastRun.error) throw lastRun.error;
    if (lastRun.data?.finished_at) {
      const age = (Date.now() - Date.parse(lastRun.data.finished_at)) / 1000;
      if (Number.isFinite(age) && age < MIN_INTERVAL_SECONDS) return jsonResponse({ status: "SKIPPED_RATE_GUARD", age_seconds: Math.max(0, age), shadow_only: true, live_execution: false });
    }

    const latest = await supabase.from("brian_universe_snapshots").select("snapshot_id,observed_at,candidates").order("observed_at", { ascending: false }).limit(1).maybeSingle();
    if (latest.error || !latest.data) throw latest.error ?? new Error("universe snapshot unavailable");
    const universePayload = latest.data.candidates as Record<string, unknown>; const radar = Array.isArray(universePayload?.candidates) ? universePayload.candidates as RadarCandidate[] : [];
    const selectedMap = new Map<string, RadarCandidate>();
    for (const candidate of radar.slice(0, TOP_N)) if (candidate?.symbol?.endsWith("USDT")) selectedMap.set(candidate.symbol, candidate);
    for (const symbol of CORE_SYMBOLS) if (!selectedMap.has(symbol)) selectedMap.set(symbol, { symbol, radar_score: 0.5, liquidity_score: 0.5, activity_score: 0.5 });
    const selected = [...selectedMap.values()]; if (!selected.length) throw new Error("no symbols selected for intrabar scan");

    const allBookPayload = await fetchJson("https://api.binance.com/api/v3/ticker/bookTicker"); if (!Array.isArray(allBookPayload)) throw new Error("invalid bookTicker payload");
    const bookMap = new Map<string, Book>();
    for (const value of allBookPayload) {
      if (!value || typeof value !== "object") continue; const r = value as Record<string, unknown>; const symbol = String(r.symbol ?? ""); const bid = finite(r.bidPrice); const ask = finite(r.askPrice);
      if (!symbol || bid <= 0 || ask < bid) continue; const mid = (bid + ask) / 2; bookMap.set(symbol, { bid, ask, mid, spreadBps: 10000 * (ask - bid) / Math.max(mid, 1e-12) });
    }

    const observedAt = new Date().toISOString(); const observedMs = Date.parse(observedAt);
    const fetched = await mapLimit(selected, 10, async (candidate): Promise<MarketRow | null> => {
      const book = bookMap.get(candidate.symbol); if (!book) return null;
      const klineUrl = `https://api.binance.com/api/v3/klines?symbol=${candidate.symbol}&interval=1m&limit=${KLINE_LIMIT}`;
      const aggUrl = `https://api.binance.com/api/v3/aggTrades?symbol=${candidate.symbol}&limit=${AGG_TRADE_LIMIT}`;
      const klinePayload = await fetchJson(klineUrl); let aggPayload: unknown = []; let degradedAggTrades = false;
      try { aggPayload = await fetchJson(aggUrl); } catch { degradedAggTrades = true; }
      const bars = parseBars(klinePayload); const trades = parseAggTrades(aggPayload); if (bars.length < 22) return null;
      const current = bars.at(-1)!; const baseline = bars.filter((b) => b.openTime < current.openTime).slice(-20); if (baseline.length < 15) return null;
      const sigma = Math.max(0.00008, std(logReturns(baseline.map((b) => b.close)))); const elapsedFraction = clip((observedMs - current.openTime) / 60_000, 0.05, 1);
      const medianQuoteVolume = Math.max(1e-12, median(baseline.map((b) => b.quoteVolume)));
      const velocityCoverageSeconds = trades.length >= 2 ? Math.max(0, (trades.at(-1)!.ts - trades[0].ts) / 1000) : 0;
      const return30s = windowReturn(trades, observedMs, 30); const previous30sReturn = windowReturn(trades, observedMs, 30, 30);
      const decelerating = return30s !== 0 && previous30sReturn !== 0 && sign(return30s) === sign(previous30sReturn) && Math.abs(return30s) < Math.abs(previous30sReturn) * 0.5;
      return { candidate, bars, trades, book, degradedAggTrades, sigma, current, baseline, elapsedFraction, medianQuoteVolume, velocityCoverageSeconds, return30s, previous30sReturn, decelerating };
    });
    const usable = fetched.filter((x): x is MarketRow => x !== null); const degradedSources = usable.some((x) => x.degradedAggTrades) ? ["aggTrades_partial"] : [];
    const rawCaptureId = await persistRaw({ schema_version: SCHEMA_VERSION, experiment_id: EXPERIMENT_ID, observed_at: observedAt, universe_snapshot_id: latest.data.snapshot_id, selected_symbols: selected.map((x) => x.symbol), market: usable.map((x) => ({ symbol: x.candidate.symbol, candidate: x.candidate, bars: x.bars, agg_trades: x.trades, book: x.book, degraded_agg_trades: x.degradedAggTrades })) }, observedAt);

    const signalRows: SignalRow[] = [];
    for (const market of usable) {
      const signals = [signalVelocity(market, observedMs), signalVolumeBurst(market), signalBreakout(market), signalReclaim(market), signalTakerFlow(market)];
      const assetId = `crypto:${market.candidate.symbol}`;
      for (let i = 0; i < TEMPLATES.length; i++) {
        const template = TEMPLATES[i]; const eyeId = await sha256(`${template.id}|${assetId}`);
        signalRows.push({ template, assetId, candidate: market.candidate, market, signal: signals[i], eyeId });
      }
    }
    const consensusEyeIds = new Map<string, string>();
    for (const market of usable) consensusEyeIds.set(`crypto:${market.candidate.symbol}`, await sha256(`intrabar-consensus|crypto:${market.candidate.symbol}`));
    const allEyeIds = [...signalRows.map((x) => x.eyeId), ...consensusEyeIds.values()];
    const priorRows: PriorTick[] = [];
    for (let i = 0; i < allEyeIds.length; i += PRIOR_LOOKUP_BATCH) {
      const chunk = allEyeIds.slice(i, i + PRIOR_LOOKUP_BATCH);
      const priorResp = await supabase.from("brian_micro_book_ticks")
        .select("eye_id,starting_equity,equity_after,peak_equity_after,max_drawdown_pct_after,target_direction,observed_mid_price,observed_at")
        .in("eye_id", chunk)
        .order("observed_at", { ascending: false })
        .limit(1000);
      if (priorResp.error) throw priorResp.error;
      priorRows.push(...((priorResp.data ?? []) as PriorTick[]));
    }
    const latestByEye = new Map<string, PriorTick>();
    priorRows.sort((a, b) => Date.parse(String(b.observed_at ?? 0)) - Date.parse(String(a.observed_at ?? 0)));
    for (const row of priorRows) if (!latestByEye.has(row.eye_id)) latestByEye.set(row.eye_id, row);

    const observations: Observation[] = []; const microTicks: Record<string, unknown>[] = [];
    for (const row of signalRows) {
      const prior = latestByEye.get(row.eyeId); if (row.signal.direction === 0 && (!prior || Number(prior.target_direction) === 0)) continue;
      const confidence = confidenceFromRadar(row.candidate, row.market.book); const observationId = await sha256(`${row.eyeId}|${observedAt}|${row.signal.direction}|${row.signal.strength.toFixed(12)}`);
      const observation: Observation = { observation_id: observationId, eye_id: row.eyeId, template_id: row.template.id, asset_id: row.assetId, market_domain: "crypto", sensor_family: row.template.family, horizon: "MICRO_1_5M", independent_group: row.template.group, observed_at: observedAt, direction: row.signal.direction, strength: row.signal.strength, confidence, reliability: 0.5, available: true, source_ids: [rawCaptureId], reason: row.signal.reason, evidence_class: EVIDENCE_CLASS, shadow_only: true, live_execution: false, metadata: { ...row.signal.metadata, feature_snapshot_at: observedAt, mid: row.market.book.mid, spread_bps: row.market.book.spreadBps, current_1m_open_time: new Date(row.market.current.openTime).toISOString(), current_1m_partial: row.market.current.closeTime > observedMs, universe_snapshot_id: latest.data.snapshot_id, schema_version: SCHEMA_VERSION } };
      observations.push(observation);
      const tick = microTick({ eyeId: row.eyeId, templateId: row.template.id, assetId: row.assetId, observedAt, mid: row.market.book.mid, spreadBps: row.market.book.spreadBps, strength: row.signal.strength, confidence, targetDirection: row.signal.direction, startingTicket: row.template.ticket, rawCaptureId, prior, metadata: { reason: row.signal.reason, schema_version: SCHEMA_VERSION } });
      if (tick) { tick.tick_id = await sha256(`${row.eyeId}|${observedAt}|${row.signal.direction}|${row.market.book.mid}`); microTicks.push(tick); }
    }
    if (observations.length) { const ins = await supabase.from("brian_sensor_observations").insert(observations); if (ins.error) throw ins.error; }

    const events: Record<string, unknown>[] = [];
    for (const market of usable) {
      const assetId = `crypto:${market.candidate.symbol}`; const rows = signalRows.filter((x) => x.assetId === assetId && x.signal.direction !== 0); if (!rows.length) {
        const consensusEyeId = consensusEyeIds.get(assetId)!; const prior = latestByEye.get(consensusEyeId);
        if (prior && Number(prior.target_direction) !== 0) {
          const tick = microTick({ eyeId: consensusEyeId, templateId: "intrabar-consensus", assetId, observedAt, mid: market.book.mid, spreadBps: market.book.spreadBps, strength: 0, confidence: confidenceFromRadar(market.candidate, market.book), targetDirection: 0, startingTicket: 5, rawCaptureId, prior, metadata: { reason: "intrabar consensus cleared", schema_version: SCHEMA_VERSION } });
          if (tick) { tick.tick_id = await sha256(`${consensusEyeId}|${observedAt}|0|${market.book.mid}`); microTicks.push(tick); }
        }
        continue;
      }
      const signed = rows.map((x) => x.signal.direction * x.signal.strength * confidenceFromRadar(x.candidate, x.market.book) * 0.5); const aggregate = mean(signed); const direction = sign(aggregate);
      const support = rows.filter((x) => x.signal.direction === direction); const conflicts = rows.filter((x) => x.signal.direction !== direction); const supportRatio = support.length / rows.length; const breadth = 0.4 + 0.6 * Math.min(1, rows.length / 3);
      const score = clip(Math.abs(aggregate) * (0.5 + 0.5 * supportRatio) * breadth); const eligible = direction !== 0 && support.length >= MIN_SUPPORT_GROUPS && score >= MIN_CONSENSUS_SCORE;
      const close5 = market.baseline.at(-5)?.close ?? market.baseline[0].close; const extensionSigma = Math.abs(Math.log(market.book.mid / close5)) / Math.max(market.sigma * Math.sqrt(5), 1e-12);
      const velocity = rows.find((x) => x.template.id === "velocity-micro")?.signal; const taker = rows.find((x) => x.template.id === "taker-flow-micro")?.signal;
      const staleVelocity = velocity ? velocity.direction !== direction || velocity.strength < 0.25 : true; const flowConflict = taker ? taker.direction === -direction : false;
      const lateChase = eligible && extensionSigma >= OVEREXTENSION_SIGMA && (market.decelerating || staleVelocity || flowConflict);
      const status = eligible && !lateChase ? "ACTIONABLE_SHADOW" : lateChase ? "VETOED_LATE_CHASE" : "WATCH"; const roundTripCostBps = 2 * (FEE_BPS + SLIPPAGE_BPS + market.book.spreadBps / 2);
      const eventId = await sha256(`${EXPERIMENT_ID}|${assetId}|${observedAt}|${direction}|${score.toFixed(12)}|${status}`);
      events.push({ event_id: eventId, experiment_id: EXPERIMENT_ID, observed_at: observedAt, asset_id: assetId, direction, score, support_groups: support.map((x) => x.template.group).sort(), conflict_groups: conflicts.map((x) => x.template.group).sort(), source_observation_ids: support.map((x) => observations.find((o) => o.eye_id === x.eyeId)?.observation_id).filter((value): value is string => Boolean(value)), observed_mid_price: market.book.mid, observed_spread_bps: market.book.spreadBps, estimated_round_trip_cost_bps: roundTripCostBps, extension_sigma: extensionSigma, late_chase: lateChase, status, reason: status === "ACTIONABLE_SHADOW" ? "independent intrabar evidence aligned before overextension veto" : status === "VETOED_LATE_CHASE" ? "signal arrived after statistically extended move without sufficient fresh continuation" : "intrabar evidence not strong/independent enough", evidence_class: EVIDENCE_CLASS, shadow_only: true, live_execution: false, metadata: { schema_version: SCHEMA_VERSION, decelerating: market.decelerating, return_30s: market.return30s, previous_30s_return: market.previous30sReturn, current_1m_partial: market.current.closeTime > observedMs, raw_capture_id: rawCaptureId, universe_snapshot_id: latest.data.snapshot_id } });

      const consensusEyeId = consensusEyeIds.get(assetId)!; const prior = latestByEye.get(consensusEyeId); const targetDirection = status === "ACTIONABLE_SHADOW" ? direction : 0;
      const tick = microTick({ eyeId: consensusEyeId, templateId: "intrabar-consensus", assetId, observedAt, mid: market.book.mid, spreadBps: market.book.spreadBps, strength: score, confidence: confidenceFromRadar(market.candidate, market.book), targetDirection, startingTicket: 5, rawCaptureId, prior, metadata: { status, late_chase: lateChase, support_groups: support.map((x) => x.template.group).sort(), conflict_groups: conflicts.map((x) => x.template.group).sort(), extension_sigma: extensionSigma, estimated_round_trip_cost_bps: roundTripCostBps, schema_version: SCHEMA_VERSION } });
      if (tick) { tick.tick_id = await sha256(`${consensusEyeId}|${observedAt}|${targetDirection}|${market.book.mid}`); microTicks.push(tick); }
    }
    if (events.length) { const ins = await supabase.from("brian_intrabar_reaction_events").insert(events); if (ins.error) throw ins.error; }
    if (microTicks.length) { const ins = await supabase.from("brian_micro_book_ticks").insert(microTicks); if (ins.error) throw ins.error; }

    const status = degradedSources.length ? "DEGRADED" : "SUCCESS"; const stored = observations.length + events.length + microTicks.length;
    await recordCollectorRun(startedAt, status, usable.length, stored, degradedSources, { schema_version: SCHEMA_VERSION, experiment_id: EXPERIMENT_ID, selected_count: selected.length, usable_count: usable.length, observation_count: observations.length, event_count: events.length, actionable_count: events.filter((e) => e.status === "ACTIONABLE_SHADOW").length, late_chase_veto_count: events.filter((e) => e.status === "VETOED_LATE_CHASE").length, micro_tick_count: microTicks.length, cadence_seconds: 60, top_n: TOP_N, core_symbols: CORE_SYMBOLS, prior_lookup_batch: PRIOR_LOOKUP_BATCH });
    return jsonResponse({ status, experiment_id: EXPERIMENT_ID, observed_at: observedAt, scanned_symbols: usable.length, signal_observations: observations.length, reaction_events: events.length, actionable_shadow: events.filter((e) => e.status === "ACTIONABLE_SHADOW").length, late_chase_vetoes: events.filter((e) => e.status === "VETOED_LATE_CHASE").length, shadow_only: true, live_execution: false });
  } catch (error) {
    const message = errorText(error);
    console.error("brian-intrabar-eye failed", message); await recordCollectorRun(startedAt, "FAILED", 0, 0, [], { schema_version: SCHEMA_VERSION, experiment_id: EXPERIMENT_ID }, error);
    return jsonResponse({ status: "FAILED_CLOSED", error: message, shadow_only: true, live_execution: false }, 500);
  }
});

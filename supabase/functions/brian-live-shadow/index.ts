import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";
import checkpoint from "./checkpoint.json" with { type: "json" };

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });

const EXPERIMENT_ID = "phase37-prospective-live-20260903";
const SCHEMA_VERSION = "brian.phase37-prospective-live-shadow.v1";
const EVIDENCE_CLASS = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"] as const;
const POLICIES = ["NATIVE", "PROFIT"] as const;
const TIMEFRAME = "5m";
const KLINE_LIMIT = 40;
const RAW_BUCKET = "brian-intelligence-raw";
const CHECKPOINT_RAW_STATE_ID = "de90c35af3525d591f17e2489e64e9c5ebd84f8124e344927d7c829623688d36";
const CHECKPOINT_PORTABLE_FINGERPRINT = "b534b611543fcf449a371faad208be20ccf7782343996d08b2bd554ed7f720b9";
const GYM = { starting_equity: 500.0, fee_bps: 10.0, assumed_spread_bps: 2.0, slippage_bps: 1.0, max_gross_exposure: 1.0, max_asset_weight: 0.35 };
const PROFIT = { round_trip_cost_multiplier: 2.0, max_positions: 3, max_asset_weight: 0.25, max_gross_exposure: 0.75, min_strength: 0.25 };

type PolicyKind = typeof POLICIES[number];
type SymbolName = typeof SYMBOLS[number];
type Bar = { closeTime: number; open: number; high: number; low: number; close: number; volume: number };
type Book = { bid: number; ask: number; mid: number; spreadBps: number };
type TickState = { observed_at: string; equity_after_costs: number; peak_equity_after: number; max_drawdown_pct_after: number; target_weights: Record<string, number>; observed_mid_prices: Record<string, number> };

function jsonResponse(body: unknown, status = 200): Response { return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } }); }
function clip(value: number, low: number, high: number): number { return Math.max(low, Math.min(high, value)); }
function mean(values: number[]): number { return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0; }
function std(values: number[]): number { if (values.length <= 1) return 0; const m = mean(values); return Math.sqrt(mean(values.map((v) => (v - m) ** 2))); }
function safeLogRatio(right: number, left: number): number { if (!(right > 0) || !(left > 0)) throw new Error("positive prices required"); return Math.log(right / left); }
function finite(value: unknown): number { const n = Number(value); if (!Number.isFinite(n)) throw new Error("non-finite market value"); return n; }
function utf8(value: string): Uint8Array { return new TextEncoder().encode(value); }
async function sha256Hex(value: string | Uint8Array): Promise<string> { const bytes = typeof value === "string" ? utf8(value) : value; const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)); return [...digest].map((b) => b.toString(16).padStart(2, "0")).join(""); }

async function fetchMarket(): Promise<{ bars: Record<SymbolName, Bar[]>; books: Record<SymbolName, Book>; observedAt: string; raw: unknown }> {
  const responses = await Promise.all(SYMBOLS.map(async (symbol) => {
    const klineUrl = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${TIMEFRAME}&limit=${KLINE_LIMIT}`;
    const bookUrl = `https://api.binance.com/api/v3/ticker/bookTicker?symbol=${symbol}`;
    const [klineResponse, bookResponse] = await Promise.all([
      fetch(klineUrl, { headers: { "accept": "application/json", "user-agent": "Brian-2026-Prospective-Shadow/1.0" }, signal: AbortSignal.timeout(8000) }),
      fetch(bookUrl, { headers: { "accept": "application/json", "user-agent": "Brian-2026-Prospective-Shadow/1.0" }, signal: AbortSignal.timeout(8000) }),
    ]);
    if (!klineResponse.ok || !bookResponse.ok) throw new Error(`Binance public fetch failed for ${symbol}: ${klineResponse.status}/${bookResponse.status}`);
    const klines = await klineResponse.json();
    const book = await bookResponse.json();
    if (!Array.isArray(klines)) throw new Error(`invalid kline payload for ${symbol}`);
    return { symbol, klines, book, klineUrl, bookUrl };
  }));
  const observedAt = new Date().toISOString();
  const cutoffMs = Date.parse(observedAt) - 2000;
  const bars = {} as Record<SymbolName, Bar[]>;
  const books = {} as Record<SymbolName, Book>;
  for (const row of responses) {
    const parsed: Bar[] = row.klines.map((k: unknown) => {
      if (!Array.isArray(k) || k.length < 7) throw new Error(`invalid kline row for ${row.symbol}`);
      return { closeTime: finite(k[6]), open: finite(k[1]), high: finite(k[2]), low: finite(k[3]), close: finite(k[4]), volume: finite(k[5]) };
    }).filter((bar: Bar) => bar.closeTime <= cutoffMs);
    if (parsed.length < Number(checkpoint.config.lookback) + 1) throw new Error(`insufficient closed bars for ${row.symbol}`);
    bars[row.symbol as SymbolName] = parsed;
    const bid = finite(row.book.bidPrice); const ask = finite(row.book.askPrice);
    if (!(bid > 0) || !(ask >= bid)) throw new Error(`invalid book for ${row.symbol}`);
    const mid = (bid + ask) / 2;
    books[row.symbol as SymbolName] = { bid, ask, mid, spreadBps: 10000 * (ask - bid) / Math.max(mid, 1e-12) };
  }
  return { bars, books, observedAt, raw: { schema_version: SCHEMA_VERSION, observed_at: observedAt, responses } };
}

async function persistRaw(raw: unknown, observedAt: string): Promise<string> {
  const canonical = JSON.stringify(raw); const bytes = utf8(canonical); const payloadHash = await sha256Hex(bytes); const compressed = gzip(bytes, { level: 6 });
  const path = `binance_public/phase37_live_market/${observedAt.slice(0, 10)}/${payloadHash}.json.gz`;
  const upload = await supabase.storage.from(RAW_BUCKET).upload(path, compressed, { contentType: "application/gzip", upsert: false, cacheControl: "31536000" });
  if (upload.error) { const msg = String(upload.error.message ?? "").toLowerCase(); const status = String((upload.error as {statusCode?: string | number}).statusCode ?? ""); if (status !== "409" && !msg.includes("exist") && !msg.includes("duplicate")) throw upload.error; }
  const captureId = await sha256Hex(`binance_public|phase37_live_market|${observedAt}|${payloadHash}`);
  const insert = await supabase.from("brian_raw_captures").insert({ capture_id: captureId, provider: "binance_public", record_type: "phase37_live_market", observed_at: observedAt, captured_at: new Date().toISOString(), provenance_uri: "https://api.binance.com/api/v3/klines + /api/v3/ticker/bookTicker", payload_hash: payloadHash, payload: { storage_bucket: RAW_BUCKET, storage_path: path, uncompressed_byte_length: bytes.byteLength, compressed_byte_length: compressed.byteLength, content_type: "application/json", content_encoding: "gzip" } });
  if (insert.error) throw insert.error;
  return captureId;
}

function alignedFrames(bars: Record<SymbolName, Bar[]>): Array<{closeTime: number; bars: Record<SymbolName, Bar>}> {
  const maps = {} as Record<SymbolName, Map<number, Bar>>; for (const symbol of SYMBOLS) maps[symbol] = new Map(bars[symbol].map((bar) => [bar.closeTime, bar]));
  let times = [...maps[SYMBOLS[0]].keys()]; for (const symbol of SYMBOLS.slice(1)) times = times.filter((t) => maps[symbol].has(t)); times.sort((a, b) => a - b);
  return times.map((closeTime) => { const out = {} as Record<SymbolName, Bar>; for (const symbol of SYMBOLS) out[symbol] = maps[symbol].get(closeTime)!; return { closeTime, bars: out }; });
}

function featureMap(frames: Array<{closeTime: number; bars: Record<SymbolName, Bar>}>): Record<SymbolName, number[]> {
  const lookback = Number(checkpoint.config.lookback); const visible = frames.slice(-lookback); if (visible.length < 2) throw new Error("feature map requires at least two frames");
  const current = visible[visible.length - 1]; const previous = visible[visible.length - 2]; const return1 = {} as Record<SymbolName, number>;
  for (const symbol of SYMBOLS) return1[symbol] = safeLogRatio(current.bars[symbol].close, previous.bars[symbol].close);
  const marketReturns = SYMBOLS.map((s) => return1[s]); const marketMomentum = mean(marketReturns); const marketDispersion = std(marketReturns); const out = {} as Record<SymbolName, number[]>;
  for (const symbol of SYMBOLS) {
    const assetBars = visible.map((f) => f.bars[symbol]); const closeReturns = assetBars.slice(1).map((bar, i) => safeLogRatio(bar.close, assetBars[i].close)); const r1 = return1[symbol];
    const r3 = assetBars.length >= 4 ? safeLogRatio(assetBars.at(-1)!.close, assetBars.at(-4)!.close) : safeLogRatio(assetBars.at(-1)!.close, assetBars[0].close);
    const trend = mean(closeReturns.slice(-5)); const vol = std(closeReturns.slice(-lookback)); const currentBar = current.bars[symbol]; const prevBar = previous.bars[symbol];
    const barRange = Math.max(0, currentBar.high / currentBar.low - 1); const relative = r1 - marketMomentum; let volumeChange = 0; if (currentBar.volume > 0 && prevBar.volume > 0) volumeChange = clip(Math.log(currentBar.volume / prevBar.volume) / 3, -1, 1);
    out[symbol] = [1, clip(r1 * 20, -3, 3), clip(r3 * 10, -3, 3), clip(trend * 20, -3, 3), clip(vol * 20, 0, 3), clip(barRange * 10, 0, 3), clip(relative * 20, -3, 3), clip(marketMomentum * 20, -3, 3), clip(marketDispersion * 20, 0, 3), volumeChange];
  }
  return out;
}
function dot(a: number[], b: number[]): number { return a.reduce((sum, value, i) => sum + value * b[i], 0); }
function grossBudget(equity: number): number { const cfg = checkpoint.config; const dd = Math.max(0, 1 - equity / GYM.starting_equity); if (dd >= Number(cfg.drawdown_flatten)) return 0; if (dd >= Number(cfg.drawdown_throttle_2)) return Number(cfg.max_gross_exposure) * 0.25; if (dd >= Number(cfg.drawdown_throttle_1)) return Number(cfg.max_gross_exposure) * 0.5; return Number(cfg.max_gross_exposure); }

function chooseAllocation(policy: PolicyKind, features: Record<SymbolName, number[]>, current: Record<string, number>, equity: number) {
  const cfg = checkpoint.config; const budget = Math.min(grossBudget(equity), policy === "PROFIT" ? PROFIT.max_gross_exposure : Number(cfg.max_gross_exposure)); const candidates: Array<{score:number;asset:SymbolName;desired:number}> = []; const diagnostics: Record<string, unknown> = {};
  const fixedOneWayCostRate = (GYM.fee_bps + GYM.assumed_spread_bps / 2 + GYM.slippage_bps) / 10000;
  for (const symbol of SYMBOLS) {
    const model = (checkpoint.models as Record<string, {weights:number[];weighted_samples:number;error_ewma:number}>)[symbol]; const prediction = clip(dot(model.weights.map(Number), features[symbol]), -Number(cfg.max_label_abs), Number(cfg.max_label_abs)); const uncertainty = Math.max(Number(cfg.min_uncertainty), Number(model.error_ewma));
    const longEdge = prediction - Number(cfg.risk_aversion) * uncertainty; const shortEdge = -prediction - Number(cfg.risk_aversion) * uncertainty; const direction = longEdge >= shortEdge ? 1 : -1; const rawEdge = Math.max(longEdge, shortEdge); let score: number | null = null; let desired = 0;
    if (Number(model.weighted_samples) >= Number(cfg.min_weighted_samples_per_asset) && rawEdge > Number(cfg.min_abs_edge)) {
      if (policy === "NATIVE") { const strength = clip(rawEdge / Math.max(Number(cfg.min_abs_edge) * 4, 1e-12), 0.25, 1); desired = direction * Number(cfg.max_asset_weight) * strength; score = rawEdge - Number(cfg.turnover_penalty_bps) / 10000 * Math.abs(desired - (current[symbol] ?? 0)); if (score > Number(cfg.min_abs_edge)) candidates.push({ score, asset: symbol, desired }); }
      else { const nativeStrength = clip(rawEdge / Math.max(Number(cfg.min_abs_edge) * 4, 1e-12), PROFIT.min_strength, 1); desired = direction * Math.min(PROFIT.max_asset_weight, Number(cfg.max_asset_weight)) * nativeStrength; const incrementalTurnover = Math.abs(desired - (current[symbol] ?? 0)); const netEdge = rawEdge - fixedOneWayCostRate * PROFIT.round_trip_cost_multiplier * incrementalTurnover; score = netEdge; if (netEdge > Number(cfg.min_abs_edge)) { const netStrength = clip(netEdge / Math.max(Number(cfg.min_abs_edge) * 4, 1e-12), PROFIT.min_strength, 1); desired = direction * Math.min(PROFIT.max_asset_weight, Number(cfg.max_asset_weight)) * netStrength; candidates.push({ score: netEdge, asset: symbol, desired }); } }
    }
    diagnostics[symbol] = { prediction, uncertainty, long_edge: longEdge, short_edge: shortEdge, raw_edge: rawEdge, selection_score: score, direction, selected_weight: 0 };
  }
  candidates.sort((a, b) => b.score - a.score || a.asset.localeCompare(b.asset)); const maxPositions = policy === "PROFIT" ? PROFIT.max_positions : Number(cfg.max_positions); const maxAssetWeight = policy === "PROFIT" ? PROFIT.max_asset_weight : Number(cfg.max_asset_weight); const weights: Record<string, number> = {}; let remaining = budget;
  for (const candidate of candidates.slice(0, maxPositions)) { const magnitude = Math.min(Math.abs(candidate.desired), remaining, maxAssetWeight); if (magnitude <= 1e-12) break; weights[candidate.asset] = Math.sign(candidate.desired) * magnitude; (diagnostics[candidate.asset] as Record<string, unknown>).selected_weight = weights[candidate.asset]; remaining -= magnitude; }
  return { weights, diagnostics };
}

function markAndDrift(previous: TickState | null, books: Record<SymbolName, Book>) {
  if (!previous) return { equityAfterMark: GYM.starting_equity, drifted: {} as Record<string, number>, periodPnl: 0 };
  const priorWeights = previous.target_weights ?? {}; const priorPrices = previous.observed_mid_prices ?? {}; let portfolioReturn = 0;
  for (const [asset, weight] of Object.entries(priorWeights)) { const now = books[asset as SymbolName]?.mid; const before = Number(priorPrices[asset]); if (!(now > 0) || !(before > 0)) throw new Error(`missing mark price for ${asset}`); portfolioReturn += Number(weight) * (now / before - 1); }
  const equityBefore = Number(previous.equity_after_costs); const equityAfterMark = Math.max(0, equityBefore * (1 + portfolioReturn)); const drifted: Record<string, number> = {};
  if (equityAfterMark > 0) for (const [asset, weight] of Object.entries(priorWeights)) { const ratio = books[asset as SymbolName].mid / Number(priorPrices[asset]); drifted[asset] = Number(weight) * equityBefore * ratio / equityAfterMark; }
  return { equityAfterMark, drifted, periodPnl: equityAfterMark - equityBefore };
}
function executionCost(equity: number, from: Record<string, number>, to: Record<string, number>, books: Record<SymbolName, Book>) {
  let turnoverFraction = 0; let cost = 0; const assets = new Set([...Object.keys(from), ...Object.keys(to)]);
  for (const asset of assets) { const delta = Math.abs((to[asset] ?? 0) - (from[asset] ?? 0)); if (delta <= 1e-15) continue; turnoverFraction += delta; const spreadBps = books[asset as SymbolName]?.spreadBps; if (!Number.isFinite(spreadBps)) throw new Error(`missing spread for ${asset}`); cost += equity * delta * (GYM.fee_bps + GYM.slippage_bps + Number(spreadBps) / 2) / 10000; }
  return { turnoverNotional: equity * turnoverFraction, cost };
}

async function ensureExperiment(observedAt: string): Promise<void> {
  const existing = await supabase.from("brian_live_shadow_experiments").select("experiment_id").eq("experiment_id", EXPERIMENT_ID).maybeSingle(); if (existing.error) throw existing.error; if (existing.data) return;
  const insert = await supabase.from("brian_live_shadow_experiments").insert({ experiment_id: EXPERIMENT_ID, started_at: observedAt, evidence_class: EVIDENCE_CLASS, checkpoint_raw_state_id: CHECKPOINT_RAW_STATE_ID, checkpoint_portable_fingerprint: CHECKPOINT_PORTABLE_FINGERPRINT, checkpoint_source_run_id: "33766345728", symbols: [...SYMBOLS], timeframe: TIMEFRAME, starting_equity: GYM.starting_equity, policies: [...POLICIES], config: { schema_version: SCHEMA_VERSION, gym: GYM, profit: PROFIT, learner_config: checkpoint.config, historical_backfill_allowed: false, learning_enabled: false, evaluation_gate: { min_elapsed_days: 7, min_active_ticks: 20, max_drawdown_pct: 10, net_return_must_be_positive: true, no_automatic_promotion: true } }, shadow_only: true });
  if (insert.error) throw insert.error;
}
async function latestTick(policy: PolicyKind): Promise<TickState | null> { const result = await supabase.from("brian_live_shadow_ticks").select("observed_at,equity_after_costs,peak_equity_after,max_drawdown_pct_after,target_weights,observed_mid_prices").eq("experiment_id", EXPERIMENT_ID).eq("policy_kind", policy).order("observed_at", { ascending: false }).limit(1).maybeSingle(); if (result.error) throw result.error; return result.data as TickState | null; }

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return jsonResponse({ error: "POST required" }, 405);
  try {
    const market = await fetchMarket(); const captureId = await persistRaw(market.raw, market.observedAt); await ensureExperiment(market.observedAt); const frames = alignedFrames(market.bars); const lookback = Number(checkpoint.config.lookback); if (frames.length < lookback) throw new Error("insufficient aligned closed frames");
    const latestFeatureClose = frames.at(-1)!.closeTime; const features = featureMap(frames.slice(-lookback)); const midPrices = Object.fromEntries(SYMBOLS.map((s) => [s, market.books[s].mid])); const spreadBps = Object.fromEntries(SYMBOLS.map((s) => [s, market.books[s].spreadBps])); const outputs: Record<string, unknown> = {};
    for (const policy of POLICIES) {
      const previous = await latestTick(policy); if (previous && Date.parse(previous.observed_at) >= Date.parse(market.observedAt) - 240000) { outputs[policy] = { status: "SKIPPED_RATE_GUARD", previous_observed_at: previous.observed_at }; continue; }
      const marked = markAndDrift(previous, market.books); const allocation = chooseAllocation(policy, features, marked.drifted, marked.equityAfterMark); const costs = executionCost(marked.equityAfterMark, marked.drifted, allocation.weights, market.books); const equityAfterCosts = Math.max(0, marked.equityAfterMark - costs.cost); const previousPeak = previous ? Number(previous.peak_equity_after) : GYM.starting_equity; const peak = Math.max(previousPeak, equityAfterCosts); const drawdown = 100 * Math.max(0, peak - equityAfterCosts) / Math.max(peak, 1e-12); const maxDd = Math.max(previous ? Number(previous.max_drawdown_pct_after) : 0, drawdown); const active = Object.values(allocation.weights).some((w) => Math.abs(Number(w)) > 1e-12); const tickId = await sha256Hex(`${EXPERIMENT_ID}|${policy}|${market.observedAt}|${captureId}`);
      const insert = await supabase.from("brian_live_shadow_ticks").insert({ tick_id: tickId, experiment_id: EXPERIMENT_ID, policy_kind: policy, observed_at: market.observedAt, feature_close_at: new Date(latestFeatureClose).toISOString(), raw_capture_id: captureId, equity_before_mark: previous ? Number(previous.equity_after_costs) : GYM.starting_equity, period_pnl: marked.periodPnl, equity_after_mark: marked.equityAfterMark, trading_cost: costs.cost, equity_after_costs: equityAfterCosts, peak_equity_after: peak, drawdown_pct: drawdown, max_drawdown_pct_after: maxDd, turnover_notional: costs.turnoverNotional, prior_weights: previous?.target_weights ?? {}, drifted_weights: marked.drifted, target_weights: allocation.weights, observed_mid_prices: midPrices, observed_spread_bps: spreadBps, feature_hash: await sha256Hex(JSON.stringify(features)), diagnostics: { schema_version: SCHEMA_VERSION, model: allocation.diagnostics, active, learning_enabled: false, historical_backfill: false }, evidence_class: EVIDENCE_CLASS, shadow_only: true });
      if (insert.error) throw insert.error; outputs[policy] = { status: "CAPTURED", active, target_weights: allocation.weights, equity: equityAfterCosts, period_pnl: marked.periodPnl, trading_cost: costs.cost, drawdown_pct: drawdown };
    }
    return jsonResponse({ status: "CAPTURED", experiment_id: EXPERIMENT_ID, observed_at: market.observedAt, feature_close_at: new Date(latestFeatureClose).toISOString(), checkpoint_portable_fingerprint: CHECKPOINT_PORTABLE_FINGERPRINT, policies: outputs, evidence_class: EVIDENCE_CLASS, historical_backfill: false, learning_enabled: false, live_execution: false, shadow_only: true });
  } catch (error) { console.error("brian-live-shadow failed", error); return jsonResponse({ status: "FAILED_CLOSED", error: String(error instanceof Error ? error.message : error), live_execution: false, shadow_only: true }, 500); }
});

import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";
import checkpoint from "./checkpoint.json" with { type: "json" };
import {
  alignedFrames, type Bar, type Book, chooseAllocation, executionCost, featureMap, finite, GYM, markAndDrift,
  type ModelWeights, POLICIES, type PolicyKind, PROFIT, SYMBOLS, type SymbolName, type TickState,
} from "./logic.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });

const EXPERIMENT_ID = "phase37-prospective-live-20260903";
const SCHEMA_VERSION = "brian.phase37-prospective-live-shadow.v1";
const EVIDENCE_CLASS = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const TIMEFRAME = "5m";
const KLINE_LIMIT = 40;
const RAW_BUCKET = "brian-intelligence-raw";
const CHECKPOINT_RAW_STATE_ID = "de90c35af3525d591f17e2489e64e9c5ebd84f8124e344927d7c829623688d36";
const CHECKPOINT_PORTABLE_FINGERPRINT = "b534b611543fcf449a371faad208be20ccf7782343996d08b2bd554ed7f720b9";

function jsonResponse(body: unknown, status = 200): Response { return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } }); }
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
    const latestFeatureClose = frames.at(-1)!.closeTime; const features = featureMap(frames.slice(-lookback), lookback); const midPrices = Object.fromEntries(SYMBOLS.map((s) => [s, market.books[s].mid])); const spreadBps = Object.fromEntries(SYMBOLS.map((s) => [s, market.books[s].spreadBps])); const outputs: Record<string, unknown> = {};
    const models = checkpoint.models as unknown as Record<SymbolName, ModelWeights>;
    for (const policy of POLICIES) {
      const previous = await latestTick(policy); if (previous && Date.parse(previous.observed_at) >= Date.parse(market.observedAt) - 240000) { outputs[policy] = { status: "SKIPPED_RATE_GUARD", previous_observed_at: previous.observed_at }; continue; }
      const marked = markAndDrift(previous, market.books); const allocation = chooseAllocation(policy, features, marked.drifted, marked.equityAfterMark, models, checkpoint.config); const costs = executionCost(marked.equityAfterMark, marked.drifted, allocation.weights, market.books); const equityAfterCosts = Math.max(0, marked.equityAfterMark - costs.cost); const previousPeak = previous ? Number(previous.peak_equity_after) : GYM.starting_equity; const peak = Math.max(previousPeak, equityAfterCosts); const drawdown = 100 * Math.max(0, peak - equityAfterCosts) / Math.max(peak, 1e-12); const maxDd = Math.max(previous ? Number(previous.max_drawdown_pct_after) : 0, drawdown); const active = Object.values(allocation.weights).some((w) => Math.abs(Number(w)) > 1e-12); const tickId = await sha256Hex(`${EXPERIMENT_ID}|${policy}|${market.observedAt}|${captureId}`);
      const insert = await supabase.from("brian_live_shadow_ticks").insert({ tick_id: tickId, experiment_id: EXPERIMENT_ID, policy_kind: policy, observed_at: market.observedAt, feature_close_at: new Date(latestFeatureClose).toISOString(), raw_capture_id: captureId, equity_before_mark: previous ? Number(previous.equity_after_costs) : GYM.starting_equity, period_pnl: marked.periodPnl, equity_after_mark: marked.equityAfterMark, trading_cost: costs.cost, equity_after_costs: equityAfterCosts, peak_equity_after: peak, drawdown_pct: drawdown, max_drawdown_pct_after: maxDd, turnover_notional: costs.turnoverNotional, prior_weights: previous?.target_weights ?? {}, drifted_weights: marked.drifted, target_weights: allocation.weights, observed_mid_prices: midPrices, observed_spread_bps: spreadBps, feature_hash: await sha256Hex(JSON.stringify(features)), diagnostics: { schema_version: SCHEMA_VERSION, model: allocation.diagnostics, active, learning_enabled: false, historical_backfill: false }, evidence_class: EVIDENCE_CLASS, shadow_only: true });
      if (insert.error) throw insert.error; outputs[policy] = { status: "CAPTURED", active, target_weights: allocation.weights, equity: equityAfterCosts, period_pnl: marked.periodPnl, trading_cost: costs.cost, drawdown_pct: drawdown };
    }
    return jsonResponse({ status: "CAPTURED", experiment_id: EXPERIMENT_ID, observed_at: market.observedAt, feature_close_at: new Date(latestFeatureClose).toISOString(), checkpoint_portable_fingerprint: CHECKPOINT_PORTABLE_FINGERPRINT, policies: outputs, evidence_class: EVIDENCE_CLASS, historical_backfill: false, learning_enabled: false, live_execution: false, shadow_only: true });
  } catch (error) { console.error("brian-live-shadow failed", error); return jsonResponse({ status: "FAILED_CLOSED", error: String(error instanceof Error ? error.message : error), live_execution: false, shadow_only: true }, 500); }
});

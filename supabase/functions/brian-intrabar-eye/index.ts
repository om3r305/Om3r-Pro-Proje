import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";
import {
  EVIDENCE_CLASS,
  buildIntrabarConsensus,
  clip,
  computeExtensionSigma,
  confidenceFromRadar,
  finite,
  logReturns,
  median,
  microTick,
  parseAggTrades,
  parseBars,
  sign,
  signalBreakout,
  signalReclaim,
  signalTakerFlow,
  signalVelocity,
  signalVolumeBurst,
  std,
  windowReturn,
} from "./logic.ts";
import type { Book, IntrabarSignalRow, MarketRow, PriorTick, RadarCandidate, Signal } from "./logic.ts";
import { withCollectorLease } from "../_shared/collector_lease.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });

const SCHEMA_VERSION = "brian.phase40-intrabar-reaction.v1";
const EXPERIMENT_ID = "phase40-intrabar-reaction-v1";
const RAW_BUCKET = "brian-intelligence-raw";
const COLLECTOR_ID = "brian-intrabar-eye";
const TOP_N = 50;
const CORE_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"] as const;
const KLINE_LIMIT = 32;
const AGG_TRADE_LIMIT = 200;
const MIN_INTERVAL_SECONDS = 50;
const LEASE_SECONDS = 55;
const PRIOR_LOOKUP_BATCH = 40;
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
    // Cadence is anchored to the prior run start, not its finish. A 10–15s runtime must not suppress the next minute's cron tick.
    const lastRun = await supabase.from("brian_collector_runs").select("started_at").eq("collector_id", COLLECTOR_ID).in("status", ["SUCCESS", "DEGRADED"]).order("started_at", { ascending: false }).limit(1).maybeSingle();
    if (lastRun.error) throw lastRun.error;
    if (lastRun.data?.started_at) {
      const age = (Date.now() - Date.parse(lastRun.data.started_at)) / 1000;
      if (Number.isFinite(age) && age < MIN_INTERVAL_SECONDS) return jsonResponse({ status: "SKIPPED_RATE_GUARD", age_seconds: Math.max(0, age), shadow_only: true, live_execution: false });
    }

    const lease = await withCollectorLease(supabase, COLLECTOR_ID, LEASE_SECONDS, async () => {
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
      const consensusRows: IntrabarSignalRow[] = rows.map((x) => ({ independentGroup: x.template.group, direction: x.signal.direction, strength: x.signal.strength, confidence: confidenceFromRadar(x.candidate, x.market.book), reliability: 0.5 }));
      const extensionSigma = computeExtensionSigma(market.baseline, market.book.mid, market.sigma);
      const outcome = buildIntrabarConsensus(consensusRows, { minSupportGroups: MIN_SUPPORT_GROUPS, minConsensusScore: MIN_CONSENSUS_SCORE, overextensionSigma: OVEREXTENSION_SIGMA, extensionSigma, decelerating: market.decelerating, spreadBps: market.book.spreadBps });
      const supportRows = rows.filter((x) => x.signal.direction === outcome.direction);
      const eventId = await sha256(`${EXPERIMENT_ID}|${assetId}|${observedAt}|${outcome.direction}|${outcome.score.toFixed(12)}|${outcome.status}`);
      events.push({ event_id: eventId, experiment_id: EXPERIMENT_ID, observed_at: observedAt, asset_id: assetId, direction: outcome.direction, score: outcome.score, support_groups: outcome.supportGroups, conflict_groups: outcome.conflictGroups, source_observation_ids: supportRows.map((x) => observations.find((o) => o.eye_id === x.eyeId)?.observation_id).filter((value): value is string => Boolean(value)), observed_mid_price: market.book.mid, observed_spread_bps: market.book.spreadBps, estimated_round_trip_cost_bps: outcome.roundTripCostBps, extension_sigma: extensionSigma, late_chase: outcome.lateChase, status: outcome.status, reason: outcome.status === "ACTIONABLE_SHADOW" ? "independent intrabar evidence aligned before overextension veto" : outcome.status === "VETOED_LATE_CHASE" ? "signal arrived after statistically extended move without sufficient fresh continuation" : "intrabar evidence not strong/independent enough", evidence_class: EVIDENCE_CLASS, shadow_only: true, live_execution: false, metadata: { schema_version: SCHEMA_VERSION, decelerating: market.decelerating, return_30s: market.return30s, previous_30s_return: market.previous30sReturn, current_1m_partial: market.current.closeTime > observedMs, raw_capture_id: rawCaptureId, universe_snapshot_id: latest.data.snapshot_id } });

      const consensusEyeId = consensusEyeIds.get(assetId)!; const prior = latestByEye.get(consensusEyeId); const targetDirection = outcome.status === "ACTIONABLE_SHADOW" ? outcome.direction : 0;
      const tick = microTick({ eyeId: consensusEyeId, templateId: "intrabar-consensus", assetId, observedAt, mid: market.book.mid, spreadBps: market.book.spreadBps, strength: outcome.score, confidence: confidenceFromRadar(market.candidate, market.book), targetDirection, startingTicket: 5, rawCaptureId, prior, metadata: { status: outcome.status, late_chase: outcome.lateChase, support_groups: outcome.supportGroups, conflict_groups: outcome.conflictGroups, extension_sigma: extensionSigma, estimated_round_trip_cost_bps: outcome.roundTripCostBps, schema_version: SCHEMA_VERSION } });
      if (tick) { tick.tick_id = await sha256(`${consensusEyeId}|${observedAt}|${targetDirection}|${market.book.mid}`); microTicks.push(tick); }
    }
    if (events.length) { const ins = await supabase.from("brian_intrabar_reaction_events").insert(events); if (ins.error) throw ins.error; }
    if (microTicks.length) { const ins = await supabase.from("brian_micro_book_ticks").insert(microTicks); if (ins.error) throw ins.error; }

    const status = degradedSources.length ? "DEGRADED" : "SUCCESS"; const stored = observations.length + events.length + microTicks.length;
    await recordCollectorRun(startedAt, status, usable.length, stored, degradedSources, { schema_version: SCHEMA_VERSION, experiment_id: EXPERIMENT_ID, selected_count: selected.length, usable_count: usable.length, observation_count: observations.length, event_count: events.length, actionable_count: events.filter((e) => e.status === "ACTIONABLE_SHADOW").length, late_chase_veto_count: events.filter((e) => e.status === "VETOED_LATE_CHASE").length, micro_tick_count: microTicks.length, cadence_seconds: 60, top_n: TOP_N, core_symbols: CORE_SYMBOLS, prior_lookup_batch: PRIOR_LOOKUP_BATCH });
      return jsonResponse({ status, experiment_id: EXPERIMENT_ID, observed_at: observedAt, scanned_symbols: usable.length, signal_observations: observations.length, reaction_events: events.length, actionable_shadow: events.filter((e) => e.status === "ACTIONABLE_SHADOW").length, late_chase_vetoes: events.filter((e) => e.status === "VETOED_LATE_CHASE").length, shadow_only: true, live_execution: false });
    });
    // Contended: another invocation already owns this collector's lease. No collector work has
    // run and no data has been written -- see supabase/functions/_shared/collector_lease.ts.
    if (lease.contended) return jsonResponse({ status: "SKIPPED_LEASE_CONTENDED", collector_id: COLLECTOR_ID, shadow_only: true, live_execution: false });
    return lease.value!;
  } catch (error) {
    const message = errorText(error);
    console.error("brian-intrabar-eye failed", message); await recordCollectorRun(startedAt, "FAILED", 0, 0, [], { schema_version: SCHEMA_VERSION, experiment_id: EXPERIMENT_ID }, error);
    return jsonResponse({ status: "FAILED_CLOSED", error: message, shadow_only: true, live_execution: false }, 500);
  }
});

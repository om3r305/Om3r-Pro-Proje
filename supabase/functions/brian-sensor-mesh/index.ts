import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import {
  type Book, clip, confidenceFromRadar, finite, mean, parseBars, type RadarCandidate,
  signalMeanReversion, signalMomentum, signalStructure, TEMPLATES, ticketFor,
} from "./logic.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });

const SCHEMA_VERSION = "brian.phase38-crypto-sensor-mesh.v1";
const EVIDENCE_CLASS = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const RAW_BUCKET = "brian-intelligence-raw";
const COLLECTOR_ID = "brian-sensor-mesh";
const TOP_N = 25;
const KLINE_LIMIT = 30;
const MIN_INTERVAL_SECONDS = 240;
const LEASE_SECONDS = 240;
const FEE_BPS = 10.0;
const SLIPPAGE_BPS = 1.0;
const LOGICAL_TEMPLATE_COUNT = 10;

type Observation = {
  observation_id: string; eye_id: string; template_id: string; asset_id: string; market_domain: string;
  sensor_family: string; horizon: string; independent_group: string; observed_at: string; direction: number;
  strength: number; confidence: number; reliability: number; available: boolean; source_ids: string[];
  reason: string; evidence_class: string; shadow_only: boolean; live_execution: boolean; metadata: Record<string, unknown>;
};

type PriorTick = {
  eye_id: string; starting_equity: string | number; equity_after: string | number; peak_equity_after: string | number;
  max_drawdown_pct_after: number; target_direction: number; observed_mid_price: string | number;
};

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}
function utf8(text: string): Uint8Array { return new TextEncoder().encode(text); }
async function sha256(value: string | Uint8Array): Promise<string> {
  const bytes = typeof value === "string" ? utf8(value) : value;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}
async function fetchJson(url: string): Promise<unknown> {
  const r = await fetch(url, { headers: { accept: "application/json", "user-agent": "Brian-2026-Sensor-Mesh/1.0" }, signal: AbortSignal.timeout(8000) });
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
  const path = `binance_public/phase38_sensor_mesh/${observedAt.slice(0, 10)}/${payloadHash}.json.gz`;
  const upload = await supabase.storage.from(RAW_BUCKET).upload(path, compressed, { contentType: "application/gzip", upsert: false, cacheControl: "31536000" });
  if (upload.error) { const msg = String(upload.error.message ?? "").toLowerCase(); const status = String((upload.error as {statusCode?: string | number}).statusCode ?? ""); if (status !== "409" && !msg.includes("duplicate") && !msg.includes("exist")) throw upload.error; }
  const captureId = await sha256(`binance_public|phase38_sensor_mesh|${observedAt}|${payloadHash}`);
  const ins = await supabase.from("brian_raw_captures").insert({
    capture_id: captureId, provider: "binance_public", record_type: "phase38_sensor_mesh", observed_at: observedAt,
    captured_at: new Date().toISOString(), provenance_uri: "https://api.binance.com/api/v3/klines + /api/v3/ticker/bookTicker",
    payload_hash: payloadHash, payload: { storage_bucket: RAW_BUCKET, storage_path: path, content_type: "application/json", content_encoding: "gzip", uncompressed_byte_length: bytes.byteLength, compressed_byte_length: compressed.byteLength },
  });
  if (ins.error) throw ins.error; return captureId;
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return response({ error: "POST required" }, 405);
  try {
    const lastRound = await supabase.from("brian_opportunity_tournament_rounds").select("observed_at").order("observed_at", { ascending: false }).limit(1).maybeSingle();
    if (lastRound.error) throw lastRound.error;
    if (lastRound.data?.observed_at) { const age = (Date.now() - Date.parse(lastRound.data.observed_at)) / 1000; if (Number.isFinite(age) && age < MIN_INTERVAL_SECONDS) return response({ status: "SKIPPED_RATE_GUARD", age_seconds: Math.max(0, age), shadow_only: true }); }

    const lease = await withCollectorLease(supabase, COLLECTOR_ID, LEASE_SECONDS, async () => {
    const latest = await supabase.from("brian_universe_snapshots").select("snapshot_id,observed_at,candidates").order("observed_at", { ascending: false }).limit(1).maybeSingle();
    if (latest.error || !latest.data) throw latest.error ?? new Error("universe snapshot unavailable");
    const payload = latest.data.candidates as Record<string, unknown>; const radar = Array.isArray(payload?.candidates) ? payload.candidates as RadarCandidate[] : [];
    const selected = radar.filter((x) => typeof x.symbol === "string" && x.symbol.endsWith("USDT")).slice(0, TOP_N);
    if (!selected.length) throw new Error("universe radar produced no selected symbols");

    const allBookPayload = await fetchJson("https://api.binance.com/api/v3/ticker/bookTicker");
    if (!Array.isArray(allBookPayload)) throw new Error("invalid bookTicker payload");
    const bookMap = new Map<string, Book>();
    for (const row of allBookPayload) { if (!row || typeof row !== "object") continue; const r = row as Record<string, unknown>; const symbol = String(r.symbol ?? ""); const bid = finite(r.bidPrice); const ask = finite(r.askPrice); if (!symbol || bid <= 0 || ask < bid) continue; const mid = (bid + ask) / 2; bookMap.set(symbol, { bid, ask, mid, spreadBps: 10000 * (ask - bid) / Math.max(mid, 1e-12) }); }

    const observedAt = new Date().toISOString(); const cutoffMs = Date.parse(observedAt) - 2000;
    const marketRows = await mapLimit(selected, 8, async (candidate) => {
      const url = `https://api.binance.com/api/v3/klines?symbol=${candidate.symbol}&interval=5m&limit=${KLINE_LIMIT}`;
      const klinePayload = await fetchJson(url); const bars = parseBars(klinePayload, cutoffMs); const book = bookMap.get(candidate.symbol); if (bars.length < 15 || !book) return null;
      return { candidate, bars, book, url };
    });
    const usable = marketRows.filter((x): x is NonNullable<typeof x> => x !== null);
    const rawCaptureId = await persistRaw({ schema_version: SCHEMA_VERSION, observed_at: observedAt, universe_snapshot_id: latest.data.snapshot_id, selected_symbols: selected.map((x) => x.symbol), market: usable.map((x) => ({ symbol: x.candidate.symbol, bars: x.bars, book: x.book, kline_url: x.url })) }, observedAt);

    const observations: Observation[] = [];
    for (const row of usable) {
      const signals = [signalStructure(row.bars), signalMomentum(row.bars), signalMeanReversion(row.bars)];
      const confidence = confidenceFromRadar(row.candidate, row.book); const featureCloseAt = new Date(row.bars.at(-1)!.closeTime).toISOString();
      for (let i = 0; i < TEMPLATES.length; i++) {
        const template = TEMPLATES[i]; const signal = signals[i]; const eyeId = await sha256(`${template.id}|crypto:${row.candidate.symbol}`);
        const observationId = await sha256(`${eyeId}|${observedAt}|${featureCloseAt}|${signal.direction}|${signal.strength.toFixed(12)}`);
        observations.push({ observation_id: observationId, eye_id: eyeId, template_id: template.id, asset_id: `crypto:${row.candidate.symbol}`, market_domain: "crypto", sensor_family: template.family, horizon: "FAST_5_30M", independent_group: template.group, observed_at: observedAt, direction: signal.direction, strength: signal.strength, confidence, reliability: 0.5, available: true, source_ids: [rawCaptureId], reason: signal.reason, evidence_class: EVIDENCE_CLASS, shadow_only: true, live_execution: false, metadata: { feature_close_at: featureCloseAt, mid: row.book.mid, spread_bps: row.book.spreadBps, universe_snapshot_id: latest.data.snapshot_id } });
      }
    }
    if (observations.length) { const ins = await supabase.from("brian_sensor_observations").insert(observations); if (ins.error) throw ins.error; }

    const eyeIds = observations.map((x) => x.eye_id); const priorResp = eyeIds.length ? await supabase.from("brian_micro_book_ticks").select("eye_id,starting_equity,equity_after,peak_equity_after,max_drawdown_pct_after,target_direction,observed_mid_price,observed_at").in("eye_id", eyeIds).order("observed_at", { ascending: false }).limit(500) : { data: [], error: null };
    if (priorResp.error) throw priorResp.error; const latestByEye = new Map<string, PriorTick>(); for (const row of (priorResp.data ?? []) as PriorTick[]) if (!latestByEye.has(row.eye_id)) latestByEye.set(row.eye_id, row);

    const marketByAsset = new Map(usable.map((x) => [`crypto:${x.candidate.symbol}`, x])); const microTicks: Record<string, unknown>[] = [];
    for (const obs of observations) {
      const market = marketByAsset.get(obs.asset_id)!; const prior = latestByEye.get(obs.eye_id); const targetDirection = obs.direction; const priorDirection = prior ? Number(prior.target_direction) : 0;
      if (!prior && targetDirection === 0) continue; if (prior && priorDirection === 0 && targetDirection === 0) continue;
      const starting = prior ? Number(prior.starting_equity) : ticketFor(obs.template_id); const equityBefore = prior ? Number(prior.equity_after) : starting; const priorMid = prior ? Number(prior.observed_mid_price) : market.book.mid;
      const periodPnl = prior ? equityBefore * priorDirection * (market.book.mid / priorMid - 1) : 0; const marked = Math.max(0, equityBefore + periodPnl); const turnover = Math.abs(targetDirection - priorDirection); const oneWay = (FEE_BPS + SLIPPAGE_BPS + market.book.spreadBps / 2) / 10000; const tradingCost = marked * turnover * oneWay; const equityAfter = Math.max(0, marked - tradingCost); const peak = Math.max(prior ? Number(prior.peak_equity_after) : starting, equityAfter); const dd = 100 * Math.max(0, peak - equityAfter) / Math.max(peak, 1e-12); const maxDd = Math.max(prior ? Number(prior.max_drawdown_pct_after) : 0, dd);
      const tickId = await sha256(`${obs.eye_id}|${observedAt}|${targetDirection}|${market.book.mid}`);
      microTicks.push({ tick_id: tickId, eye_id: obs.eye_id, template_id: obs.template_id, asset_id: obs.asset_id, horizon: obs.horizon, observed_at: observedAt, feature_close_at: obs.metadata.feature_close_at, starting_equity: starting, equity_before: equityBefore, period_pnl: periodPnl, trading_cost: tradingCost, equity_after: equityAfter, peak_equity_after: peak, max_drawdown_pct_after: maxDd, prior_direction: priorDirection, target_direction: targetDirection, observed_mid_price: market.book.mid, observed_spread_bps: market.book.spreadBps, signal_strength: obs.strength, signal_confidence: obs.confidence, raw_capture_id: rawCaptureId, evidence_class: EVIDENCE_CLASS, shadow_only: true, live_execution: false, metadata: { reason: obs.reason } });
    }
    if (microTicks.length) { const ins = await supabase.from("brian_micro_book_ticks").insert(microTicks); if (ins.error) throw ins.error; }

    const byAsset = new Map<string, Observation[]>(); for (const obs of observations) if (obs.direction !== 0) { const rows = byAsset.get(obs.asset_id) ?? []; rows.push(obs); byAsset.set(obs.asset_id, rows); }
    const candidateRows: Array<Record<string, unknown>> = [];
    for (const [assetId, rows] of byAsset.entries()) {
      const unique = new Map<string, Observation>(); for (const row of rows) { const old = unique.get(row.independent_group); if (!old || row.strength * row.confidence > old.strength * old.confidence) unique.set(row.independent_group, row); }
      const evidence = [...unique.values()]; if (!evidence.length) continue; const signed = evidence.map((x) => x.direction * x.strength * x.confidence * x.reliability); const aggregate = mean(signed); const direction = aggregate > 0 ? 1 : aggregate < 0 ? -1 : 0; const support = evidence.filter((x) => x.direction === direction).length / evidence.length; const score = clip(Math.abs(aggregate) * (0.5 + 0.5 * support) * (0.5 + 0.5 * Math.min(1, evidence.length / 4))); const eligible = evidence.length >= 2 && direction !== 0 && score >= 0.20; const ticket = !eligible ? 0 : score >= 0.8 ? 20 : score >= 0.6 ? 10 : score >= 0.4 ? 5 : 3;
      candidateRows.push({ asset_id: assetId, direction, opportunity_score: score, independent_groups: evidence.map((x) => x.independent_group).sort(), supporting_observation_ids: evidence.filter((x) => x.direction === direction).map((x) => x.observation_id), conflicting_observation_ids: evidence.filter((x) => x.direction !== direction).map((x) => x.observation_id), requested_virtual_ticket_usd: ticket, eligible });
    }
    candidateRows.sort((a, b) => Number(b.eligible) - Number(a.eligible) || Number(b.opportunity_score) - Number(a.opportunity_score) || String(a.asset_id).localeCompare(String(b.asset_id)));
    let allocated = 0; const finalCandidates = candidateRows.slice(0, 25).map((row) => { let ticket = Number(row.requested_virtual_ticket_usd); if (row.eligible) { ticket = Math.min(ticket, Math.max(0, 500 - allocated)); allocated += ticket; } return { ...row, virtual_ticket_usd: ticket, eligible: Boolean(row.eligible) && ticket > 0 }; });
    const roundId = await sha256(`phase38|${observedAt}|${latest.data.snapshot_id}|${rawCaptureId}`); const roundIns = await supabase.from("brian_opportunity_tournament_rounds").insert({ round_id: roundId, observed_at: observedAt, logical_eye_count: usable.length * LOGICAL_TEMPLATE_COUNT, candidate_count: finalCandidates.length, eligible_count: finalCandidates.filter((x) => x.eligible).length, virtual_allocated_usd: allocated, virtual_unallocated_usd: 500 - allocated, candidates: { schema_version: SCHEMA_VERSION, universe_snapshot_id: latest.data.snapshot_id, scanned_symbols: usable.map((x) => x.candidate.symbol), active_templates: TEMPLATES.map((x) => x.id), unavailable_template_families: ["orderbook_direction", "derivatives", "onchain", "news", "social_psychology", "cross_asset", "macro"], candidates: finalCandidates }, evidence_class: EVIDENCE_CLASS, shadow_only: true, live_execution: false });
    if (roundIns.error) throw roundIns.error;

      return response({ status: "CAPTURED", observed_at: observedAt, universe_snapshot_id: latest.data.snapshot_id, scanned_symbols: usable.length, logical_eye_count: usable.length * LOGICAL_TEMPLATE_COUNT, stored_sensor_observations: observations.length, stored_micro_book_ticks: microTicks.length, eligible_candidates: finalCandidates.filter((x) => x.eligible).slice(0, 10), unavailable_families: ["derivatives", "onchain", "news", "social_psychology", "cross_asset", "macro"], learning_enabled: false, live_execution: false, shadow_only: true });
    });
    // Contended: another invocation already owns this collector's lease. No collector work has
    // run and no data has been written -- see supabase/functions/_shared/collector_lease.ts.
    if (lease.contended) return response({ status: "SKIPPED_LEASE_CONTENDED", collector_id: COLLECTOR_ID, shadow_only: true, live_execution: false });
    return lease.value!;
  } catch (error) {
    console.error("brian-sensor-mesh failed", error);
    return response({ status: "FAILED_CLOSED", error: String(error instanceof Error ? error.message : error), learning_enabled: false, live_execution: false, shadow_only: true }, 500);
  }
});

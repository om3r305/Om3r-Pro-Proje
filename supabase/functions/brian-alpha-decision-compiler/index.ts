import { createClient } from "npm:@supabase/supabase-js@2";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import {
  ALPHA_COMPILER_VERSION,
  compileAlphaDecision,
  type AlphaEvidenceRow,
  type IntrabarVetoContext,
} from "../_shared/alpha_decision.ts";
import { compileDegradedTopOfBookCost, type DynamicCostQuote } from "../_shared/dynamic_cost.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });

const COLLECTOR_ID = "brian-alpha-decision-compiler-v2";
const EVIDENCE = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const MIN_INTERVAL_SECONDS = 50;
const LEASE_SECONDS = 55;
const CORE_ASSETS = ["crypto:BTCUSDT", "crypto:ETHUSDT", "crypto:SOLUSDT", "crypto:BNBUSDT", "crypto:XRPUSDT"];
const MAX_ASSETS = 25;
const FALLBACK_FEE_BPS = 10;
const FALLBACK_SLIPPAGE_BPS = 1;

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}
function clip(v: number, lo = 0, hi = 1) { return Math.max(lo, Math.min(hi, v)); }
function finite(v: unknown, fallback = 0) { const n = Number(v); return Number.isFinite(n) ? n : fallback; }
function errorText(error: unknown): string { return error instanceof Error ? `${error.name}: ${error.message}` : String(error); }
function utf8(s: string) { return new TextEncoder().encode(s); }
async function sha(s: string) { const d = new Uint8Array(await crypto.subtle.digest("SHA-256", utf8(s))); return [...d].map((b) => b.toString(16).padStart(2, "0")).join(""); }

function freshnessMs(horizon: string): number {
  // Operational freshness derives from the declared horizon, not Sep-4 outcome tuning.
  if (horizon === "MICRO_1_5M") return 5 * 60_000;
  if (horizon === "FAST_5_30M") return 30 * 60_000;
  if (horizon === "EVENT_DRIVEN") return 60 * 60_000;
  if (horizon === "DAILY") return 36 * 60 * 60_000;
  return 0; // unknown horizon fails stale/closed
}
function isFresh(observedAt: string, horizon: string, nowMs: number): boolean {
  const t = Date.parse(observedAt); const max = freshnessMs(horizon);
  return Number.isFinite(t) && max > 0 && t <= nowMs + 5_000 && nowMs - t <= max;
}
function ticketForScore(score: number): number {
  // Reuses the existing Phase 3.8 virtual-ticket schedule; not tuned from Sep-4.
  return score >= 0.8 ? 20 : score >= 0.6 ? 10 : score >= 0.4 ? 5 : 3;
}

async function latestRadarAssets(): Promise<string[]> {
  const out = new Set<string>(CORE_ASSETS);
  const frame = await supabase.from("brian_emergent_mover_frames")
    .select("observed_at,report")
    .eq("provider", "binance_public")
    .order("observed_at", { ascending: false }).limit(1).maybeSingle();
  if (!frame.error && frame.data?.report && typeof frame.data.report === "object") {
    const candidates = (frame.data.report as Record<string, unknown>).candidates;
    if (Array.isArray(candidates)) {
      for (const raw of candidates.slice(0, MAX_ASSETS)) {
        if (!raw || typeof raw !== "object") continue;
        const symbol = String((raw as Record<string, unknown>).symbol ?? "").trim().toUpperCase();
        if (/^[A-Z0-9]+USDT$/.test(symbol)) out.add(`crypto:${symbol}`);
      }
    }
  }
  return [...out].slice(0, MAX_ASSETS);
}

async function loadSensorEvidence(assets: string[], nowMs: number): Promise<Map<string, AlphaEvidenceRow[]>> {
  const map = new Map<string, AlphaEvidenceRow[]>();
  const since = new Date(nowMs - 36 * 60 * 60_000).toISOString();
  const resp = await supabase.from("brian_sensor_observations")
    .select("observation_id,asset_id,sensor_family,horizon,independent_group,observed_at,direction,strength,confidence,reliability,available,reason")
    .in("asset_id", assets).gte("observed_at", since).eq("available", true)
    .order("observed_at", { ascending: false }).limit(6000);
  if (resp.error) throw resp.error;
  for (const r of resp.data ?? []) {
    const asset = String(r.asset_id);
    const rows = map.get(asset) ?? [];
    rows.push({
      observationId: String(r.observation_id), sourceKind: String(r.sensor_family), independentGroup: String(r.independent_group),
      direction: Number(r.direction), strength: finite(r.strength), confidence: finite(r.confidence), reliability: finite(r.reliability),
      observedAt: String(r.observed_at), horizon: String(r.horizon), fresh: isFresh(String(r.observed_at), String(r.horizon), nowMs), reason: String(r.reason ?? "sensor evidence"),
    });
    map.set(asset, rows);
  }
  return map;
}

async function addDipEvidence(map: Map<string, AlphaEvidenceRow[]>, assets: string[], nowMs: number) {
  const since = new Date(nowMs - 5 * 60_000).toISOString();
  const resp = await supabase.from("brian_dip_events")
    .select("event_id,symbol,observed_at,event_kind,metadata")
    .eq("event_kind", "BUY").gte("observed_at", since).order("observed_at", { ascending: false }).limit(100);
  if (resp.error) return; // optional experimental eye: absence cannot fail the compiler
  const allowed = new Set(assets);
  for (const r of resp.data ?? []) {
    const asset = `crypto:${String(r.symbol).toUpperCase()}`; if (!allowed.has(asset)) continue;
    const md = (r.metadata ?? {}) as Record<string, unknown>;
    const confidence = clip(finite(md.expert_confidence, 0.5));
    const rawScore = finite(md.dip_score, 0.5); const strength = clip(rawScore > 1 ? rawScore / 100 : rawScore);
    const rows = map.get(asset) ?? [];
    rows.push({ observationId: `dip:${r.event_id}`, sourceKind: "dip_trader", independentGroup: "dip_reclaim", direction: 1, strength, confidence, reliability: 0.5, observedAt: String(r.observed_at), horizon: "MICRO_1_5M", fresh: true, reason: "Dip Trader shadow BUY/reclaim evidence" });
    map.set(asset, rows);
  }
}

async function addFrozenPhase37Evidence(map: Map<string, AlphaEvidenceRow[]>, assets: string[], nowMs: number) {
  const latest = await supabase.from("brian_live_shadow_ticks")
    .select("tick_id,policy_kind,observed_at,target_weights")
    .order("observed_at", { ascending: false }).limit(2);
  if (latest.error) return;
  const allowed = new Set(assets);
  for (const r of latest.data ?? []) {
    const observedAt = String(r.observed_at); if (nowMs - Date.parse(observedAt) > 10 * 60_000) continue;
    const weights = (r.target_weights ?? {}) as Record<string, unknown>;
    for (const [symbol, rawWeight] of Object.entries(weights)) {
      const asset = `crypto:${symbol.toUpperCase()}`; if (!allowed.has(asset)) continue;
      const weight = finite(rawWeight); if (Math.abs(weight) <= 1e-12) continue;
      const rows = map.get(asset) ?? [];
      // NATIVE and PROFIT share one independent group because they descend from the same frozen model.
      rows.push({ observationId: `phase37:${r.tick_id}`, sourceKind: "phase37_frozen_control", independentGroup: "phase37_frozen_model", direction: weight > 0 ? 1 : -1, strength: clip(Math.abs(weight) / 0.35), confidence: 1, reliability: 0.5, observedAt, horizon: "FAST_5_30M", fresh: true, reason: `frozen Phase 3.7 ${r.policy_kind} target weight` });
      map.set(asset, rows);
    }
  }
}

async function loadIntrabarContexts(assets: string[], nowMs: number): Promise<Map<string, IntrabarVetoContext>> {
  const map = new Map<string, IntrabarVetoContext>();
  const since = new Date(nowMs - 5 * 60_000).toISOString();
  const resp = await supabase.from("brian_intrabar_reaction_events")
    .select("event_id,asset_id,observed_at,direction,status,late_chase,reason")
    .in("asset_id", assets).gte("observed_at", since).order("observed_at", { ascending: false }).limit(500);
  if (resp.error) throw resp.error;
  for (const r of resp.data ?? []) if (!map.has(String(r.asset_id))) {
    const status = String(r.status) as IntrabarVetoContext["status"];
    map.set(String(r.asset_id), { eventId: String(r.event_id), direction: Number(r.direction), status, lateChase: Boolean(r.late_chase), observedAt: String(r.observed_at), fresh: true, reason: String(r.reason ?? "intrabar context") });
  }
  return map;
}

async function currentBooks(): Promise<Map<string, { mid: number; spreadBps: number }>> {
  const r = await fetch("https://api.binance.com/api/v3/ticker/bookTicker", { headers: { accept: "application/json", "user-agent": "Brian-ALPHA-v2-Shadow/1.0" }, signal: AbortSignal.timeout(8_000) });
  if (!r.ok) throw new Error(`Binance bookTicker HTTP ${r.status}`);
  const payload = await r.json(); if (!Array.isArray(payload)) throw new Error("invalid Binance bookTicker payload");
  const map = new Map<string, { mid: number; spreadBps: number }>();
  for (const x of payload) {
    if (!x || typeof x !== "object") continue; const row = x as Record<string, unknown>; const symbol = String(row.symbol ?? "").toUpperCase();
    const bid = finite(row.bidPrice), ask = finite(row.askPrice); if (!symbol || !(bid > 0) || !(ask >= bid)) continue;
    const mid = (bid + ask) / 2; map.set(`crypto:${symbol}`, { mid, spreadBps: 10000 * (ask - bid) / Math.max(mid, 1e-12) });
  }
  return map;
}

async function recordRun(startedAt: string, status: string, observed: number, stored: number, error?: unknown) {
  const finishedAt = new Date().toISOString(); const runId = await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  await supabase.from("brian_collector_runs").insert({ run_id: runId, collector_id: COLLECTOR_ID, started_at: startedAt, finished_at: finishedAt, status, observed_records: observed, stored_records: stored, degraded_sources: [], error_class: error ? "ALPHA_COMPILER_ERROR" : null, error_message: error ? errorText(error).slice(0, 1500) : null, evidence_class: EVIDENCE, shadow_only: true, live_execution: false, metadata: { compiler_version: ALPHA_COMPILER_VERSION } });
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return json({ error: "POST required" }, 405);
  const startedAt = new Date().toISOString();
  try {
    const last = await supabase.from("brian_collector_runs").select("started_at").eq("collector_id", COLLECTOR_ID).in("status", ["SUCCESS","DEGRADED"]).order("started_at", { ascending: false }).limit(1).maybeSingle();
    if (last.error) throw last.error;
    if (last.data?.started_at) { const age = (Date.now() - Date.parse(last.data.started_at)) / 1000; if (Number.isFinite(age) && age < MIN_INTERVAL_SECONDS) return json({ status: "SKIPPED_RATE_GUARD", age_seconds: age, shadow_only: true, live_execution: false }); }

    const lease = await withCollectorLease(supabase, COLLECTOR_ID, LEASE_SECONDS, async () => {
      const nowMs = Date.now(); const observedAt = new Date(nowMs).toISOString();
      const assets = await latestRadarAssets(); const evidence = await loadSensorEvidence(assets, nowMs);
      await addDipEvidence(evidence, assets, nowMs); await addFrozenPhase37Evidence(evidence, assets, nowMs);
      const intrabar = await loadIntrabarContexts(assets, nowMs); const books = await currentBooks();

      const decisionRows: Record<string, unknown>[] = []; const costRows: Record<string, unknown>[] = [];
      for (const asset of assets) {
        const rows = evidence.get(asset) ?? [];
        // First pass: eligibility without cost. Missing cost intentionally VETOs an otherwise actionable candidate.
        const preliminary = compileAlphaDecision({ evidenceRows: rows, costQuote: null, intrabarContext: intrabar.get(asset) ?? null });
        let costQuote: DynamicCostQuote | null = null; let costId: string | null = null;
        if (preliminary.evidenceScore >= 0.18 && preliminary.supportGroups.length >= 2 && preliminary.direction !== 0) {
          const book = books.get(asset); if (book) {
            const notional = ticketForScore(preliminary.evidenceScore);
            costQuote = compileDegradedTopOfBookCost({ side: preliminary.direction === 1 ? "BUY" : "SELL", notionalUsd: notional, feeBps: FALLBACK_FEE_BPS, spreadBps: book.spreadBps, assumedSlippageBps: FALLBACK_SLIPPAGE_BPS, midPrice: book.mid });
            costId = await sha(`${ALPHA_COMPILER_VERSION}|cost|${asset}|${observedAt}|${preliminary.direction}|${notional}`);
            costRows.push({ quote_id: costId, compiler_version: ALPHA_COMPILER_VERSION, asset_id: asset, observed_at: observedAt, side: costQuote.side, requested_notional_usd: costQuote.requestedNotionalUsd, filled_notional_usd: costQuote.filledNotionalUsd, fill_ratio: costQuote.fillRatio, fillable: costQuote.fillable, fee_bps: costQuote.feeBps, spread_bps: costQuote.spreadBps, depth_slippage_bps: costQuote.depthSlippageBps, one_way_cost_bps: costQuote.oneWayCostBps, estimated_round_trip_cost_bps: costQuote.estimatedRoundTripCostBps, quality: costQuote.quality, source_ids: [], reason: costQuote.reason, metadata: { mid_price: book.mid, fallback_slippage_bps: FALLBACK_SLIPPAGE_BPS }, evidence_class: EVIDENCE, shadow_only: true, live_execution: false });
          }
        }
        const decision = compileAlphaDecision({ evidenceRows: rows, costQuote, intrabarContext: intrabar.get(asset) ?? null });
        const decisionId = await sha(`${ALPHA_COMPILER_VERSION}|decision|${asset}|${observedAt}|${decision.action}|${decision.direction}|${decision.evidenceScore.toFixed(12)}`);
        decisionRows.push({ decision_id: decisionId, compiler_version: ALPHA_COMPILER_VERSION, observed_at: observedAt, asset_id: asset, action: decision.action, direction: decision.direction, evidence_score: decision.evidenceScore, independent_group_count: decision.independentGroupCount, support_groups: decision.supportGroups, conflict_groups: decision.conflictGroups, source_observation_ids: decision.sourceObservationIds.filter((id) => !id.startsWith("dip:") && !id.startsWith("phase37:")), source_intrabar_event_ids: decision.lateChaseVetoEventId ? [decision.lateChaseVetoEventId] : [], source_cost_quote_id: costId, requested_virtual_notional_usd: costQuote?.requestedNotionalUsd ?? 0, gross_edge_bps: null, estimated_round_trip_cost_bps: decision.estimatedRoundTripCostBps, net_edge_bps: null, veto_reason: decision.vetoReason, reason: decision.reason, metadata: { ignored_observation_ids: decision.ignoredObservationIds, source_evidence_ids_all: decision.sourceObservationIds, emergent_mover_role: "attention_only", score_is_not_expected_return_bps: true, cost_quality: decision.costQuality }, evidence_class: EVIDENCE, shadow_only: true, live_execution: false });
      }

      if (costRows.length) { const ins = await supabase.from("brian_dynamic_cost_quotes").insert(costRows); if (ins.error) throw ins.error; }
      if (decisionRows.length) { const ins = await supabase.from("brian_alpha_decisions").insert(decisionRows); if (ins.error) throw ins.error; }
      await recordRun(startedAt, "SUCCESS", assets.length, costRows.length + decisionRows.length);
      return json({ status: "CAPTURED", compiler_version: ALPHA_COMPILER_VERSION, observed_at: observedAt, assets: assets.length, decisions: decisionRows.length, actionable: decisionRows.filter((x) => x.action === "OPEN_LONG" || x.action === "OPEN_SHORT").length, vetoed: decisionRows.filter((x) => x.action === "VETO").length, wait: decisionRows.filter((x) => x.action === "WAIT").length, cost_quality: "DEGRADED_TOP_OF_BOOK", shadow_only: true, live_execution: false });
    });
    if (lease.contended) return json({ status: "SKIPPED_LEASE_CONTENDED", shadow_only: true, live_execution: false });
    return lease.value!;
  } catch (error) {
    console.error("brian-alpha-decision-compiler-v2 failed", errorText(error));
    try { await recordRun(startedAt, "FAILED", 0, 0, error); } catch { /* logging must not mask primary error */ }
    return json({ status: "FAILED_CLOSED", error: errorText(error), shadow_only: true, live_execution: false }, 500);
  }
});

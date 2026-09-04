import { createClient } from "npm:@supabase/supabase-js@2";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import {
  ALPHA_COMPILER_VERSION,
  compileAlphaDecision,
  type AlphaDecisionResult,
  type AlphaEvidenceRow,
  type IntrabarVetoContext,
} from "../_shared/alpha_decision.ts";
import {
  compileL2Cost,
  type DynamicCostQuote,
} from "../_shared/dynamic_cost.ts";
import {
  selectAlphaRuntimeCost,
  type RuntimeL2Status,
} from "../_shared/alpha_runtime_policy.ts";
import { parseBinanceDepthSnapshotRaw } from "../_shared/binance_l2_wire.ts";

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
const L2_DEPTH_LIMIT = 100;
const FROZEN_PHASE37_EXPERIMENT_ID = "phase37-prospective-live-20260903";
const MACRO_CONTEXT_WINDOW_MS = 60 * 60_000;
const MAX_MACRO_CONTEXT_EVENTS = 24;

type OfficialMacroContextEvent = {
  observation_id: string;
  observed_at: string;
  independent_group: string;
  source_ids: string[];
  reason: string;
  organization: string | null;
  title: string | null;
  published_at: string | null;
  provenance_uri: string | null;
};

type OfficialMacroContext = {
  role: "context_only_no_direction_vote";
  direction_vote: 0;
  window_minutes: 60;
  as_of: string;
  event_count: number;
  latest_observed_at: string | null;
  events: OfficialMacroContextEvent[];
  degraded_reason?: string;
};

type ReferenceBook = { mid: number; spreadBps: number };
type ObservedL2Cost = {
  quote: DynamicCostQuote;
  sourceId: string;
  lastUpdateId: string;
  fetchedAt: string;
};

class ObservedL2InvalidError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ObservedL2InvalidError";
  }
}

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
function emptyMacroContext(asOf: string, reason?: string): OfficialMacroContext {
  return {
    role: "context_only_no_direction_vote",
    direction_vote: 0,
    window_minutes: 60,
    as_of: asOf,
    event_count: 0,
    latest_observed_at: null,
    events: [],
    ...(reason ? { degraded_reason: reason } : {}),
  };
}
function failClosedAssetDecision(reason: string, vetoReason = "EVIDENCE_INVALID"): AlphaDecisionResult {
  return {
    compilerVersion: ALPHA_COMPILER_VERSION,
    action: "VETO",
    direction: 0,
    evidenceScore: 0,
    independentGroupCount: 0,
    supportGroups: [],
    conflictGroups: [],
    sourceObservationIds: [],
    ignoredObservationIds: [],
    lateChaseVetoEventId: null,
    costQuality: null,
    estimatedRoundTripCostBps: null,
    fillable: null,
    reason,
    vetoReason,
  };
}
function vetoFromPreliminary(preliminary: AlphaDecisionResult, reason: string, vetoReason: string): AlphaDecisionResult {
  return {
    ...preliminary,
    action: "VETO",
    lateChaseVetoEventId: null,
    costQuality: null,
    estimatedRoundTripCostBps: null,
    fillable: null,
    reason,
    vetoReason,
  };
}
function preliminaryIsActionable(preliminary: AlphaDecisionResult | undefined): preliminary is AlphaDecisionResult {
  return Boolean(
    preliminary && preliminary.direction !== 0 &&
    preliminary.supportGroups.length >= 2 && preliminary.evidenceScore >= 0.18
  );
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
    .neq("independent_group", "news_gdelt")
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
    .select("tick_id,experiment_id,policy_kind,observed_at,target_weights")
    .eq("experiment_id", FROZEN_PHASE37_EXPERIMENT_ID)
    .in("policy_kind", ["NATIVE", "PROFIT"])
    .lte("observed_at", new Date(nowMs).toISOString())
    .order("observed_at", { ascending: false }).limit(20);
  if (latest.error) return;
  const allowed = new Set(assets);
  const seenPolicies = new Set<string>();
  for (const r of latest.data ?? []) {
    const policy = String(r.policy_kind);
    if (seenPolicies.has(policy)) continue;
    seenPolicies.add(policy);
    const observedAt = String(r.observed_at); if (nowMs - Date.parse(observedAt) > 10 * 60_000) continue;
    const weights = (r.target_weights ?? {}) as Record<string, unknown>;
    for (const [symbol, rawWeight] of Object.entries(weights)) {
      const asset = `crypto:${symbol.toUpperCase()}`; if (!allowed.has(asset)) continue;
      const weight = finite(rawWeight); if (Math.abs(weight) <= 1e-12) continue;
      const rows = map.get(asset) ?? [];
      // NATIVE and PROFIT share one independent group because they descend from the same frozen model.
      rows.push({ observationId: `phase37:${r.tick_id}`, sourceKind: "phase37_frozen_control", independentGroup: "phase37_frozen_model", direction: weight > 0 ? 1 : -1, strength: clip(Math.abs(weight) / 0.35), confidence: 1, reliability: 0.5, observedAt, horizon: "FAST_5_30M", fresh: true, reason: `frozen Phase 3.7 ${policy} target weight` });
      map.set(asset, rows);
    }
  }
}

async function loadOfficialMacroContext(asOfMs: number, asOf: string): Promise<OfficialMacroContext> {
  const since = new Date(asOfMs - MACRO_CONTEXT_WINDOW_MS).toISOString();
  const resp = await supabase.from("brian_sensor_observations")
    .select("observation_id,observed_at,independent_group,source_ids,reason,metadata")
    .eq("asset_id", "global:MACRO")
    .eq("sensor_family", "official_macro_event")
    .eq("available", true)
    .eq("direction", 0)
    .gte("observed_at", since)
    .lte("observed_at", asOf)
    .order("observed_at", { ascending: false })
    .limit(MAX_MACRO_CONTEXT_EVENTS);
  if (resp.error) throw resp.error;
  const events: OfficialMacroContextEvent[] = (resp.data ?? []).map((row) => {
    const md = (row.metadata ?? {}) as Record<string, unknown>;
    return {
      observation_id: String(row.observation_id),
      observed_at: String(row.observed_at),
      independent_group: String(row.independent_group),
      source_ids: Array.isArray(row.source_ids) ? row.source_ids.map(String) : [],
      reason: String(row.reason ?? "official macro event"),
      organization: md.organization == null ? null : String(md.organization),
      title: md.title == null ? null : String(md.title),
      published_at: md.published_at == null ? null : String(md.published_at),
      provenance_uri: md.provenance_uri == null ? null : String(md.provenance_uri),
    };
  });
  return {
    role: "context_only_no_direction_vote",
    direction_vote: 0,
    window_minutes: 60,
    as_of: asOf,
    event_count: events.length,
    latest_observed_at: events[0]?.observed_at ?? null,
    events,
  };
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

async function currentBooks(): Promise<Map<string, ReferenceBook>> {
  const r = await fetch("https://api.binance.com/api/v3/ticker/bookTicker", { headers: { accept: "application/json", "user-agent": "Brian-ALPHA-v2-Shadow/1.0" }, signal: AbortSignal.timeout(8_000) });
  if (!r.ok) throw new Error(`Binance bookTicker HTTP ${r.status}`);
  const payload = await r.json(); if (!Array.isArray(payload)) throw new Error("invalid Binance bookTicker payload");
  const map = new Map<string, ReferenceBook>();
  for (const x of payload) {
    if (!x || typeof x !== "object") continue; const row = x as Record<string, unknown>; const symbol = String(row.symbol ?? "").toUpperCase();
    const bid = finite(row.bidPrice), ask = finite(row.askPrice); if (!symbol || !(bid > 0) || !(ask >= bid)) continue;
    const mid = (bid + ask) / 2; map.set(`crypto:${symbol}`, { mid, spreadBps: 10000 * (ask - bid) / Math.max(mid, 1e-12) });
  }
  return map;
}

async function fetchObservedL2Cost(asset: string, direction: -1 | 1, notional: number): Promise<ObservedL2Cost> {
  const symbol = asset.includes(":") ? asset.split(":", 2)[1] : asset;
  if (!/^[A-Z0-9]+USDT$/.test(symbol)) throw new Error(`unsupported Binance L2 asset ${asset}`);
  const url = `https://api.binance.com/api/v3/depth?symbol=${encodeURIComponent(symbol)}&limit=${L2_DEPTH_LIMIT}`;
  const response = await fetch(url, { headers: { accept: "application/json", "user-agent": "Brian-ALPHA-v2-Shadow-L2/1.0" }, signal: AbortSignal.timeout(8_000) });
  if (!response.ok) throw new Error(`Binance depth HTTP ${response.status} for ${symbol}`);
  const raw = await response.text();
  try {
    const snapshot = parseBinanceDepthSnapshotRaw(raw);
    const quote = compileL2Cost({
      side: direction === 1 ? "BUY" : "SELL",
      notionalUsd: notional,
      feeBps: FALLBACK_FEE_BPS,
      bids: snapshot.bids.map(([price, size]) => ({ price, size })),
      asks: snapshot.asks.map(([price, size]) => ({ price, size })),
    });
    const fetchedAt = new Date().toISOString();
    return {
      quote,
      sourceId: `binance_public_rest_depth:${symbol}:${snapshot.lastUpdateId}`,
      lastUpdateId: String(snapshot.lastUpdateId),
      fetchedAt,
    };
  } catch (error) {
    throw new ObservedL2InvalidError(`observed depth snapshot rejected for ${symbol}: ${errorText(error)}`);
  }
}

async function recordRun(startedAt: string, status: string, observed: number, stored: number, degradedSources: string[], error?: unknown) {
  const finishedAt = new Date().toISOString(); const runId = await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  await supabase.from("brian_collector_runs").insert({ run_id: runId, collector_id: COLLECTOR_ID, started_at: startedAt, finished_at: finishedAt, status, observed_records: observed, stored_records: stored, degraded_sources: degradedSources, error_class: error ? "ALPHA_COMPILER_ERROR" : null, error_message: error ? errorText(error).slice(0, 1500) : null, evidence_class: EVIDENCE, shadow_only: true, live_execution: false, metadata: { compiler_version: ALPHA_COMPILER_VERSION, frozen_phase37_experiment_id: FROZEN_PHASE37_EXPERIMENT_ID } });
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return json({ error: "POST required" }, 405);
  try {
    await requireCronAuth(req, supabase);
  } catch (error) {
    const message = errorText(error);
    const unauthorized = message.includes("UNAUTHORIZED_CRON");
    return json({ status: unauthorized ? "UNAUTHORIZED" : "FAILED_CLOSED", error: message, shadow_only: true, live_execution: false }, unauthorized ? 401 : 503);
  }

  const startedAt = new Date().toISOString();
  try {
    const last = await supabase.from("brian_collector_runs").select("started_at").eq("collector_id", COLLECTOR_ID).in("status", ["SUCCESS","DEGRADED"]).order("started_at", { ascending: false }).limit(1).maybeSingle();
    if (last.error) throw last.error;
    if (last.data?.started_at) { const age = (Date.now() - Date.parse(last.data.started_at)) / 1000; if (Number.isFinite(age) && age < MIN_INTERVAL_SECONDS) return json({ status: "SKIPPED_RATE_GUARD", age_seconds: age, shadow_only: true, live_execution: false }); }

    const lease = await withCollectorLease(supabase, COLLECTOR_ID, LEASE_SECONDS, async () => {
      const evidenceNowMs = Date.now();
      const degradedSources: string[] = [];
      const assets = await latestRadarAssets();

      let evidence = new Map<string, AlphaEvidenceRow[]>();
      try { evidence = await loadSensorEvidence(assets, evidenceNowMs); }
      catch (error) { degradedSources.push(`sensor_evidence:${errorText(error)}`); }
      await addDipEvidence(evidence, assets, evidenceNowMs);
      await addFrozenPhase37Evidence(evidence, assets, evidenceNowMs);

      let intrabar = new Map<string, IntrabarVetoContext>();
      let intrabarContextAvailable = true;
      try { intrabar = await loadIntrabarContexts(assets, evidenceNowMs); }
      catch (error) {
        intrabarContextAvailable = false;
        degradedSources.push(`intrabar_context:${errorText(error)}`);
      }

      let books = new Map<string, ReferenceBook>();
      try { books = await currentBooks(); }
      catch (error) { degradedSources.push(`book_ticker:${errorText(error)}`); }

      const preliminaryByAsset = new Map<string, AlphaDecisionResult>();
      const poisonByAsset = new Map<string, string>();
      for (const asset of assets) {
        try {
          preliminaryByAsset.set(asset, compileAlphaDecision({ evidenceRows: evidence.get(asset) ?? [], costQuote: null, intrabarContext: intrabar.get(asset) ?? null }));
        } catch (error) {
          poisonByAsset.set(asset, errorText(error));
          degradedSources.push(`evidence_invalid:${asset}:${errorText(error)}`);
        }
      }

      const l2ByAsset = new Map<string, ObservedL2Cost>();
      const l2StatusByAsset = new Map<string, RuntimeL2Status>();
      await Promise.all(assets.map(async (asset) => {
        const preliminary = preliminaryByAsset.get(asset);
        if (!preliminaryIsActionable(preliminary)) {
          l2StatusByAsset.set(asset, "NOT_REQUESTED");
          return;
        }
        const notional = ticketForScore(preliminary.evidenceScore);
        try {
          l2ByAsset.set(asset, await fetchObservedL2Cost(asset, preliminary.direction as -1 | 1, notional));
          l2StatusByAsset.set(asset, "OBSERVED");
        } catch (error) {
          const status: RuntimeL2Status = error instanceof ObservedL2InvalidError ? "INVALID" : "UNAVAILABLE";
          l2StatusByAsset.set(asset, status);
          degradedSources.push(`l2_${status.toLowerCase()}:${asset}:${errorText(error)}`);
        }
      }));

      const macroAsOf = new Date().toISOString();
      let macroContext = emptyMacroContext(macroAsOf);
      try { macroContext = await loadOfficialMacroContext(Date.parse(macroAsOf), macroAsOf); }
      catch (error) {
        const reason = errorText(error);
        degradedSources.push(`official_macro_context:${reason}`);
        macroContext = emptyMacroContext(macroAsOf, reason);
      }

      // Decision time is after all evidence/cost/context reads, so every attached input is causal.
      const observedAt = new Date().toISOString();
      const decisionRows: Record<string, unknown>[] = []; const costRows: Record<string, unknown>[] = [];
      let observedL2Count = 0; let degradedCostCount = 0;

      for (const asset of assets) {
        const rows = evidence.get(asset) ?? [];
        const referenceBook = books.get(asset);
        const observedL2 = l2ByAsset.get(asset);
        const l2Status = l2StatusByAsset.get(asset) ?? "NOT_REQUESTED";
        const poison = poisonByAsset.get(asset);
        const preliminary = preliminaryByAsset.get(asset);
        let costQuote: DynamicCostQuote | null = null; let costId: string | null = null;
        let costSourceIds: string[] = [];
        let costMetadata: Record<string, unknown> = {};

        if (!poison && preliminaryIsActionable(preliminary)) {
          const notional = ticketForScore(preliminary.evidenceScore);
          costQuote = selectAlphaRuntimeCost({
            l2Status,
            observedL2Quote: observedL2?.quote ?? null,
            referenceBook: referenceBook ?? null,
            side: preliminary.direction === 1 ? "BUY" : "SELL",
            notionalUsd: notional,
            feeBps: FALLBACK_FEE_BPS,
            fallbackSlippageBps: FALLBACK_SLIPPAGE_BPS,
          });

          if (costQuote?.quality === "L2_OBSERVED" && observedL2) {
            observedL2Count++;
            costSourceIds = [observedL2.sourceId];
            costMetadata = {
              source: "binance_public_rest_depth_snapshot",
              last_update_id: observedL2.lastUpdateId,
              fetched_at: observedL2.fetchedAt,
              depth_limit: L2_DEPTH_LIMIT,
              l2_runtime_status: l2Status,
            };
          } else if (costQuote?.quality === "DEGRADED_TOP_OF_BOOK" && referenceBook) {
            degradedCostCount++;
            costSourceIds = ["binance_public_book_ticker"];
            costMetadata = {
              source: "binance_public_book_ticker",
              mid_price: referenceBook.mid,
              fallback_slippage_bps: FALLBACK_SLIPPAGE_BPS,
              l2_runtime_status: l2Status,
            };
          }
          if (costQuote) {
            costId = await sha(`${ALPHA_COMPILER_VERSION}|cost|${asset}|${observedAt}|${preliminary.direction}|${notional}|${costQuote.quality}`);
            costRows.push({ quote_id: costId, compiler_version: ALPHA_COMPILER_VERSION, asset_id: asset, observed_at: observedAt, side: costQuote.side, requested_notional_usd: costQuote.requestedNotionalUsd, filled_notional_usd: costQuote.filledNotionalUsd, fill_ratio: costQuote.fillRatio, fillable: costQuote.fillable, fee_bps: costQuote.feeBps, spread_bps: costQuote.spreadBps, depth_slippage_bps: costQuote.depthSlippageBps, one_way_cost_bps: costQuote.oneWayCostBps, estimated_round_trip_cost_bps: costQuote.estimatedRoundTripCostBps, quality: costQuote.quality, source_ids: costSourceIds, reason: costQuote.reason, metadata: costMetadata, evidence_class: EVIDENCE, shadow_only: true, live_execution: false });
          }
        }

        let decision: AlphaDecisionResult;
        if (poison) {
          decision = failClosedAssetDecision(`asset evidence rejected without aborting unrelated assets: ${poison}`);
        } else if (!intrabarContextAvailable && preliminaryIsActionable(preliminary)) {
          decision = vetoFromPreliminary(
            preliminary,
            "actionable evidence cannot open because intrabar late-chase context is unavailable",
            "CONTEXT_UNAVAILABLE",
          );
        } else {
          try { decision = compileAlphaDecision({ evidenceRows: rows, costQuote, intrabarContext: intrabar.get(asset) ?? null }); }
          catch (error) { decision = failClosedAssetDecision(`asset compile failed closed: ${errorText(error)}`); degradedSources.push(`asset_compile:${asset}:${errorText(error)}`); }
        }

        const referencePrice = observedL2?.quote.referenceMid ?? referenceBook?.mid ?? null;
        const decisionId = await sha(`${ALPHA_COMPILER_VERSION}|decision|${asset}|${observedAt}|${decision.action}|${decision.direction}|${decision.evidenceScore.toFixed(12)}`);
        decisionRows.push({
          decision_id: decisionId,
          compiler_version: ALPHA_COMPILER_VERSION,
          observed_at: observedAt,
          asset_id: asset,
          observed_reference_price: referencePrice,
          action: decision.action,
          direction: decision.direction,
          evidence_score: decision.evidenceScore,
          independent_group_count: decision.independentGroupCount,
          support_groups: decision.supportGroups,
          conflict_groups: decision.conflictGroups,
          source_observation_ids: decision.sourceObservationIds.filter((id) => !id.startsWith("dip:") && !id.startsWith("phase37:")),
          source_intrabar_event_ids: decision.lateChaseVetoEventId ? [decision.lateChaseVetoEventId] : [],
          source_cost_quote_id: costId,
          requested_virtual_notional_usd: costQuote?.requestedNotionalUsd ?? 0,
          gross_edge_bps: null,
          estimated_round_trip_cost_bps: decision.estimatedRoundTripCostBps,
          net_edge_bps: null,
          veto_reason: decision.vetoReason,
          reason: decision.reason,
          metadata: {
            ignored_observation_ids: decision.ignoredObservationIds,
            source_evidence_ids_all: decision.sourceObservationIds,
            emergent_mover_role: "attention_only",
            gdelt_role: "discovery_only_no_direction_vote",
            frozen_phase37_experiment_id: FROZEN_PHASE37_EXPERIMENT_ID,
            score_is_not_expected_return_bps: true,
            cost_quality: decision.costQuality,
            l2_runtime_status: l2Status,
            intrabar_context_available: intrabarContextAvailable,
            reference_price: {
              value: referencePrice,
              source: observedL2 ? "binance_public_rest_depth_mid" : referenceBook ? "binance_public_book_ticker_mid" : "unavailable",
              captured_at: observedL2?.fetchedAt ?? observedAt,
            },
            official_macro_context: macroContext,
          },
          evidence_class: EVIDENCE,
          shadow_only: true,
          live_execution: false,
        });
      }

      if (costRows.length) { const ins = await supabase.from("brian_dynamic_cost_quotes").insert(costRows); if (ins.error) throw ins.error; }
      if (decisionRows.length) { const ins = await supabase.from("brian_alpha_decisions").insert(decisionRows); if (ins.error) throw ins.error; }
      const status = degradedSources.length ? "DEGRADED" : "SUCCESS";
      await recordRun(startedAt, status, assets.length, costRows.length + decisionRows.length, degradedSources);
      return json({ status: "CAPTURED", run_quality: status, compiler_version: ALPHA_COMPILER_VERSION, observed_at: observedAt, assets: assets.length, decisions: decisionRows.length, actionable: decisionRows.filter((x) => x.action === "OPEN_LONG" || x.action === "OPEN_SHORT").length, vetoed: decisionRows.filter((x) => x.action === "VETO").length, wait: decisionRows.filter((x) => x.action === "WAIT").length, macro_context_events: macroContext.event_count, l2_observed_costs: observedL2Count, degraded_top_of_book_costs: degradedCostCount, degraded_sources: degradedSources, shadow_only: true, live_execution: false });
    });
    if (lease.contended) return json({ status: "SKIPPED_LEASE_CONTENDED", shadow_only: true, live_execution: false });
    return lease.value!;
  } catch (error) {
    console.error("brian-alpha-decision-compiler-v2 failed", errorText(error));
    try { await recordRun(startedAt, "FAILED", 0, 0, [], error); } catch { /* logging must not mask primary error */ }
    return json({ status: "FAILED_CLOSED", error: errorText(error), shadow_only: true, live_execution: false }, 500);
  }
});

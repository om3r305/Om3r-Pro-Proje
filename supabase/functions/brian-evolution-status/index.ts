import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import {
  assessWorldSource,
  buildEvolutionDashboardModel,
  type CapabilityGap,
} from "../_shared/evolution_core.ts";
import type { CapabilitySnapshot, WorldSourceCandidate } from "../_shared/evolution_contract.ts";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const AUTH_ID = "control-v3";
const ALLOWED_ORIGIN = /^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const ALLOWED_EXACT = new Set([
  "https://monster-coins-pro-seven.vercel.app",
  "https://monster-coins-pro-oemer-yildirim.vercel.app",
  "https://monster-coins-pro-git-brian-2026-oemer-yildirim.vercel.app",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);

function cors(origin?: string | null): Record<string, string> {
  const allowed = origin && (ALLOWED_EXACT.has(origin) || ALLOWED_ORIGIN.test(origin))
    ? origin
    : "https://monster-coins-pro-oemer-yildirim.vercel.app";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-headers": "content-type,x-brian-dashboard-key",
    "access-control-allow-methods": "POST,OPTIONS",
    "vary": "Origin",
  };
}

function out(body: unknown, status = 200, origin?: string | null): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...cors(origin) },
  });
}

async function sha256Hex(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function constantTimeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let i = 0; i < left.length; i++) diff |= left.charCodeAt(i) ^ right.charCodeAt(i);
  return diff === 0;
}

async function requireDashboardAuth(req: Request): Promise<void> {
  const supplied = (req.headers.get("x-brian-dashboard-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_DASHBOARD");
  const result = await db.from("brian_dashboard_auth")
    .select("dashboard_key_sha256")
    .eq("auth_id", AUTH_ID)
    .single();
  if (result.error || !result.data) throw new Error("DASHBOARD_AUTH_UNAVAILABLE");
  if (!constantTimeEqual(await sha256Hex(supplied), String(result.data.dashboard_key_sha256 ?? ""))) {
    throw new Error("UNAUTHORIZED_DASHBOARD");
  }
}

function toCapability(row: Record<string, unknown>): CapabilitySnapshot {
  return {
    capabilityId: String(row.capability_id),
    observedAt: String(row.observed_at),
    domain: row.domain as CapabilitySnapshot["domain"],
    name: String(row.name),
    version: row.version == null ? null : String(row.version),
    stage: row.stage as CapabilitySnapshot["stage"],
    health: row.health as CapabilitySnapshot["health"],
    description: String(row.description),
    sourceIds: Array.isArray(row.source_ids) ? row.source_ids.map(String) : [],
    dependencies: Array.isArray(row.dependencies) ? row.dependencies.map(String) : [],
    limitations: Array.isArray(row.limitations) ? row.limitations.map(String) : [],
    evidenceRefs: Array.isArray(row.evidence_refs) ? row.evidence_refs.map(String) : [],
    metadata: (row.metadata ?? {}) as Record<string, unknown>,
    evidenceClass: "PROSPECTIVE_EVOLUTION_SHADOW",
    shadowOnly: true,
    liveExecution: false,
  };
}

function toGap(row: Record<string, unknown>): CapabilityGap {
  return {
    gapId: String(row.gap_id),
    capabilityId: String(row.capability_id),
    domain: row.domain as CapabilityGap["domain"],
    severity: row.severity as CapabilityGap["severity"],
    reason: String(row.reason),
    suggestedAction: String(row.suggested_action),
    evidenceRefs: Array.isArray(row.evidence_refs) ? row.evidence_refs.map(String) : [],
  };
}

function toSource(row: Record<string, unknown>): WorldSourceCandidate {
  return {
    sourceId: String(row.source_id),
    discoveredAt: String(row.discovered_at),
    canonicalUri: String(row.canonical_uri),
    provider: String(row.provider),
    sourceKind: String(row.source_kind),
    authorityClass: row.authority_class as WorldSourceCandidate["authorityClass"],
    accessMode: row.access_mode as WorldSourceCandidate["accessMode"],
    stage: row.stage as WorldSourceCandidate["stage"],
    freshnessSeconds: row.freshness_seconds == null ? null : Number(row.freshness_seconds),
    corroborationRequired: row.corroboration_required !== false,
    manipulationRisk: Number(row.manipulation_risk ?? 0.5),
    rationale: String(row.rationale),
    metadata: (row.metadata ?? {}) as Record<string, unknown>,
  };
}

function dedupeLatest<T extends Record<string, unknown>>(rows: T[], key: keyof T): T[] {
  const seen = new Set<string>();
  const out: T[] = [];
  for (const row of rows) {
    const id = String(row[key]);
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push(row);
  }
  return out;
}

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
  if (req.method !== "POST") return out({ error: "POST required" }, 405, origin);
  try {
    await requireDashboardAuth(req);
  } catch (error) {
    return out({ error: String(error) }, 401, origin);
  }

  const [capsQ, gapsQ, sourcesQ, eventsQ, runsQ] = await Promise.all([
    db.from("brian_evolution_latest_capabilities")
      .select("capability_id,observed_at,domain,name,version,stage,health,description,source_ids,dependencies,limitations,evidence_refs,metadata")
      .order("capability_id", { ascending: true }),
    db.from("brian_evolution_latest_gaps")
      .select("gap_id,observed_at,capability_id,domain,severity,reason,suggested_action,evidence_refs,metadata")
      .order("observed_at", { ascending: false }).limit(100),
    db.from("brian_world_source_candidates")
      .select("source_id,discovered_at,canonical_uri,provider,source_kind,authority_class,access_mode,stage,freshness_seconds,corroboration_required,manipulation_risk,rationale,metadata,created_at")
      .order("created_at", { ascending: false }).limit(500),
    db.from("brian_evolution_events")
      .select("event_id,entity_type,entity_id,event_type,occurred_at,stage,title,summary,evidence_refs,payload")
      .order("occurred_at", { ascending: false }).limit(80),
    db.from("brian_evolution_orchestrator_runs")
      .select("run_id,started_at,finished_at,status,capability_snapshots,gap_snapshots,source_candidates,source_assessments,journal_events,error_class,error_message,metadata")
      .order("started_at", { ascending: false }).limit(30),
  ]);

  const failures = [capsQ, gapsQ, sourcesQ, eventsQ, runsQ].filter((q) => q.error).map((q) => q.error!.message);
  if (failures.length) return out({ status: "DEGRADED", errors: failures, shadow_only: true, live_execution: false }, 500, origin);

  const capabilities = (capsQ.data ?? []).map((row) => toCapability(row as Record<string, unknown>));
  const gaps = (gapsQ.data ?? []).map((row) => toGap(row as Record<string, unknown>));
  const latestSources = dedupeLatest((sourcesQ.data ?? []) as Record<string, unknown>[], "source_id")
    .map((row) => toSource(row));
  const latestJournalAt = eventsQ.data?.[0]?.occurred_at ? String(eventsQ.data[0].occurred_at) : null;
  const now = new Date().toISOString();
  const dashboard = buildEvolutionDashboardModel(capabilities, gaps, latestSources, latestJournalAt, now);
  const sources = latestSources.slice(0, 60).map((source) => ({
    ...source,
    assessment: assessWorldSource(source, now),
  }));

  return out({
    status: "ONLINE",
    observed_at: now,
    dashboard,
    capabilities,
    gaps: gaps.slice(0, 40),
    sources,
    journal: eventsQ.data ?? [],
    runs: runsQ.data ?? [],
    cloud_independent: true,
    canonical_mutation: false,
    direct_alpha_influence: false,
    shadow_only: true,
    live_execution: false,
  }, 200, origin);
});

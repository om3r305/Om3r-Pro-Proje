import { createClient } from "npm:@supabase/supabase-js@2.116.0";

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
    vary: "Origin",
  };
}
function out(body: unknown, status = 200, origin?: string | null): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...cors(origin) } });
}
async function sha256Hex(value: string): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function constantTimeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let i = 0; i < left.length; i++) diff |= left.charCodeAt(i) ^ right.charCodeAt(i);
  return diff === 0;
}
async function auth(req: Request): Promise<void> {
  const supplied = (req.headers.get("x-brian-dashboard-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_DASHBOARD");
  const q = await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id", AUTH_ID).single();
  if (q.error || !q.data) throw new Error("DASHBOARD_AUTH_UNAVAILABLE");
  if (!constantTimeEqual(await sha256Hex(supplied), String(q.data.dashboard_key_sha256 ?? ""))) throw new Error("UNAUTHORIZED_DASHBOARD");
}

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
  if (req.method !== "POST") return out({ error: "POST required" }, 405, origin);
  try { await auth(req); } catch (error) { return out({ error: String(error) }, 401, origin); }

  const [narrativesQ, futureQ, mechanismsQ, scenariosQ, impactsQ, entitiesQ, relationsQ, runsQ, discoveryQ] = await Promise.all([
    db.from("brian_world_latest_narratives")
      .select("narrative_id,label,observed_at,event_ids,entity_ids,strength,breadth,direction_balance,stage,evidence_refs")
      .order("strength", { ascending: false }).limit(40),
    db.from("brian_world_upcoming_events")
      .select("future_event_id,event_kind,scheduled_at,first_observed_at,title,entity_ids,asset_ids,source_event_id,confidence,stage,evidence_refs")
      .limit(60),
    db.from("brian_world_causal_mechanisms")
      .select("mechanism_id,narrative_id,observed_at,cause,transmission,affected_assets,confidence,stage,evidence_refs,counter_evidence_required")
      .order("observed_at", { ascending: false }).limit(60),
    db.from("brian_world_scenario_snapshots")
      .select("scenario_id,mechanism_id,observed_at,branch,assumptions,invalidators,asset_impacts,confidence,stage,evidence_refs")
      .order("observed_at", { ascending: false }).limit(80),
    db.from("brian_world_asset_impact_candidates")
      .select("impact_id,asset_id,observed_at,mechanism_id,scenario_id,conditional_direction,confidence,rationale,stage,evidence_refs,direct_alpha_influence")
      .order("observed_at", { ascending: false }).limit(100),
    db.from("brian_world_entity_observations")
      .select("entity_id,canonical_name,entity_type,observed_at,event_id,match_kind,confidence,evidence_refs")
      .order("observed_at", { ascending: false }).limit(120),
    db.from("brian_world_relation_assertions")
      .select("src_entity_id,dst_entity_id,relation,observed_at,event_id,confidence,mechanism,stage,evidence_refs")
      .order("observed_at", { ascending: false }).limit(100),
    db.from("brian_world_brain_runs")
      .select("run_id,started_at,finished_at,status,input_events,event_frames,entity_observations,relation_assertions,narrative_snapshots,future_events,causal_mechanisms,scenario_snapshots,asset_impacts,error_class,error_message,metadata")
      .order("started_at", { ascending: false }).limit(24),
    db.from("brian_collector_runs")
      .select("collector_id,status,started_at,finished_at,observed_records,stored_records,degraded_sources,error_class,error_message,metadata")
      .in("collector_id", ["brian-world-discovery-eye-v1", "brian-world-brain-v1"])
      .order("started_at", { ascending: false }).limit(30),
  ]);
  const named = { narrativesQ, futureQ, mechanismsQ, scenariosQ, impactsQ, entitiesQ, relationsQ, runsQ, discoveryQ };
  const errors = Object.entries(named).flatMap(([name, q]) => q.error ? [`${name}:${q.error.message}`] : []);
  if (errors.length) return out({ status: "DEGRADED", errors, shadow_only: true, live_execution: false }, 500, origin);

  const entityRows = entitiesQ.data ?? [];
  const uniqueEntities = new Map<string, unknown>();
  for (const row of entityRows) if (!uniqueEntities.has(String(row.entity_id))) uniqueEntities.set(String(row.entity_id), row);
  const mechanisms = mechanismsQ.data ?? [];
  const latestMechanismByNarrative = new Map<string, unknown>();
  for (const row of mechanisms) if (!latestMechanismByNarrative.has(String(row.narrative_id))) latestMechanismByNarrative.set(String(row.narrative_id), row);
  const runs = discoveryQ.data ?? [];
  const latestDiscovery = runs.find((row) => row.collector_id === "brian-world-discovery-eye-v1") ?? null;
  const latestBrain = runs.find((row) => row.collector_id === "brian-world-brain-v1") ?? null;

  return out({
    status: errors.length ? "DEGRADED" : "ONLINE",
    observed_at: new Date().toISOString(),
    summary: {
      narratives: (narrativesQ.data ?? []).length,
      upcoming_events: (futureQ.data ?? []).length,
      unique_entities: uniqueEntities.size,
      recent_relations: (relationsQ.data ?? []).length,
      causal_mechanisms: latestMechanismByNarrative.size,
      scenario_branches: (scenariosQ.data ?? []).length,
      asset_impact_candidates: (impactsQ.data ?? []).length,
      discovery_status: latestDiscovery?.status ?? "NO_DATA",
      world_brain_status: latestBrain?.status ?? "NO_DATA",
    },
    narratives: narrativesQ.data ?? [],
    upcoming_events: futureQ.data ?? [],
    entities: [...uniqueEntities.values()].slice(0, 80),
    relations: relationsQ.data ?? [],
    mechanisms: [...latestMechanismByNarrative.values()],
    scenarios: scenariosQ.data ?? [],
    impacts: impactsQ.data ?? [],
    runs: runsQ.data ?? [],
    collectors: { discovery: latestDiscovery, world_brain: latestBrain },
    semantics: {
      discovery_is_truth: false,
      mechanisms_are_research_hypotheses: true,
      scenarios_are_conditional: true,
      direct_alpha_influence: false,
      canonical_mutation: false,
      browser_required: false,
    },
    shadow_only: true,
    live_execution: false,
  }, 200, origin);
});

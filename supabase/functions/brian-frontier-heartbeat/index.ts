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
const TRACKED = [
  "brian-world-brain-v1",
  "brian-world-discovery-eye-v1",
  "brian-alpha-decision-compiler-v2",
  "brian-intrabar-eye",
  "brian-evolution-orchestrator-v1",
  "brian-evolution-researcher-v1",
  "brian-evolution-sandbox-v1",
  "brian-evolution-ocean-worker-v1",
  "brian-evolution-treasury-v1",
  "brian-evolution-alpha-edge-challenger-v1",
];

function cors(origin: string | null) {
  const allowed = origin && (ALLOWED_EXACT.has(origin) || ALLOWED_ORIGIN.test(origin))
    ? origin
    : "https://monster-coins-pro-oemer-yildirim.vercel.app";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-headers": "content-type,x-brian-dashboard-key",
    "access-control-allow-methods": "POST,OPTIONS",
    "cache-control": "no-store",
    "vary": "Origin",
  };
}
function out(body: unknown, status = 200, origin: string | null = null) {
  return new Response(JSON.stringify(body), { status, headers: { ...cors(origin), "content-type": "application/json; charset=utf-8" } });
}
async function sha256Hex(value: string) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function equal(a: string, b: string) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
async function requireDashboard(req: Request) {
  const supplied = (req.headers.get("x-brian-dashboard-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_DASHBOARD");
  const q = await db.from("brian_dashboard_auth").select("dashboard_key_sha256").eq("auth_id", AUTH_ID).single();
  if (q.error || !q.data) throw new Error("DASHBOARD_AUTH_UNAVAILABLE");
  if (!equal(await sha256Hex(supplied), String(q.data.dashboard_key_sha256 ?? ""))) throw new Error("UNAUTHORIZED_DASHBOARD");
}
async function rpc(name: string) {
  const q = await db.rpc(name);
  if (q.error) throw new Error(`${name}:${q.error.message}`);
  return q.data;
}

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
  if (req.method !== "POST") return out({ error: "POST required" }, 405, origin);
  try { await requireDashboard(req); } catch (error) { return out({ error: String(error) }, 401, origin); }

  try {
    const [control, alpha, runs, worldRun] = await Promise.all([
      rpc("brian_system_control_status"),
      db.from("brian_alpha_decisions")
        .select("observed_at,decision_id,asset_id,action,direction,evidence_score,estimated_round_trip_cost_bps")
        .order("observed_at", { ascending: false }).limit(1).maybeSingle(),
      db.from("brian_collector_runs")
        .select("collector_id,status,started_at,finished_at,error_class,error_message,observed_records,stored_records")
        .in("collector_id", TRACKED)
        .order("started_at", { ascending: false }).limit(120),
      db.from("brian_world_brain_runs")
        .select("status,started_at,finished_at,input_events,event_frames,entity_observations,narrative_snapshots,asset_impacts")
        .order("started_at", { ascending: false }).limit(1).maybeSingle(),
    ]);

    const latest: Record<string, unknown> = {};
    if (!runs.error && Array.isArray(runs.data)) {
      for (const row of runs.data) {
        const id = String(row.collector_id ?? "");
        if (id && latest[id] == null) latest[id] = row;
      }
    }

    return out({
      status: "OK",
      observed_at: new Date().toISOString(),
      control,
      alpha: alpha.error ? null : alpha.data,
      world_run: worldRun.error ? null : worldRun.data,
      collectors: latest,
      dip_touched: false,
      shadow_only: true,
      live_execution: false,
    }, 200, origin);
  } catch (error) {
    return out({
      status: "DEGRADED",
      error: error instanceof Error ? error.message : String(error),
      dip_touched: false,
      shadow_only: true,
      live_execution: false,
    }, 503, origin);
  }
});

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
    "vary": "Origin",
  };
}
function out(body: unknown, status = 200, origin?: string | null) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...cors(origin) },
  });
}
async function sha256Hex(value: string) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}
function constantTimeEqual(left: string, right: string) {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let index = 0; index < left.length; index++) diff |= left.charCodeAt(index) ^ right.charCodeAt(index);
  return diff === 0;
}
async function requireDashboardAuth(req: Request) {
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
  try {
    await requireDashboardAuth(req);
  } catch (error) {
    return out({ error: String(error) }, 401, origin);
  }

  const [snapshotQ, actionsQ, runsQ] = await Promise.all([
    db.from("brian_treasury_shadow_latest")
      .select("snapshot_id,cycle_id,observed_at,starting_equity_usd,cash_usd,equity_usd,realized_pnl_usd,cumulative_costs_usd,deployment_usd,deployment_pct,cash_reserve_pct,positions,action_count,promotion_gate_open,promotion_gate_ref,promotion_gate_reason,treasury_version,gate_version,blocked_reasons,metadata")
      .limit(1).maybeSingle(),
    db.from("brian_treasury_shadow_actions")
      .select("action_id,cycle_id,observed_at,kind,asset_id,direction,capital_usd,reference_price,cost_usd,expected_net_edge_bps,source_decision_id,reason,position_id")
      .order("observed_at", { ascending: false }).limit(80),
    db.from("brian_collector_runs")
      .select("collector_id,status,started_at,finished_at,observed_records,stored_records,error_class,error_message,metadata")
      .eq("collector_id", "brian-evolution-treasury-v1").order("started_at", { ascending: false }).limit(30),
  ]);

  const errors = [snapshotQ.error, actionsQ.error, runsQ.error].filter(Boolean).map((error) => error!.message);
  if (errors.length) return out({ status: "DEGRADED", errors, shadow_only: true, live_execution: false }, 500, origin);
  const snapshot = snapshotQ.data ?? null;
  const positions = snapshot && Array.isArray(snapshot.positions) ? snapshot.positions : [];
  const starting = Number(snapshot?.starting_equity_usd ?? 10_000);
  const equity = Number(snapshot?.equity_usd ?? starting);
  const pnl = Number.isFinite(equity) && Number.isFinite(starting) ? equity - starting : null;

  return out({
    status: snapshot ? "ONLINE" : "WAITING_FOR_FIRST_CYCLE",
    observed_at: new Date().toISOString(),
    summary: {
      starting_equity_usd: starting,
      equity_usd: equity,
      total_pnl_usd: pnl,
      cash_usd: Number(snapshot?.cash_usd ?? starting),
      deployment_usd: Number(snapshot?.deployment_usd ?? 0),
      deployment_pct: Number(snapshot?.deployment_pct ?? 0),
      cash_reserve_pct: Number(snapshot?.cash_reserve_pct ?? 1),
      realized_pnl_usd: Number(snapshot?.realized_pnl_usd ?? 0),
      cumulative_costs_usd: Number(snapshot?.cumulative_costs_usd ?? 0),
      open_positions: positions.length,
      promotion_gate_open: snapshot?.promotion_gate_open === true,
      promotion_gate_ref: snapshot?.promotion_gate_ref ?? null,
      promotion_gate_reason: snapshot?.promotion_gate_reason ?? "treasury has not run yet",
    },
    snapshot,
    positions,
    actions: actionsQ.data ?? [],
    runs: runsQ.data ?? [],
    cloud_independent: true,
    canonical_alpha_mutation: false,
    shadow_only: true,
    live_execution: false,
  }, 200, origin);
});

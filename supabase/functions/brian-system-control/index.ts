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

function cors(origin: string | null) {
  const allowed = origin && (ALLOWED_EXACT.has(origin) || ALLOWED_ORIGIN.test(origin))
    ? origin
    : "https://monster-coins-pro-seven.vercel.app";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-headers": "content-type,x-brian-dashboard-key",
    "access-control-allow-methods": "POST,OPTIONS",
    "cache-control": "no-store",
    "vary": "Origin",
  };
}
function out(body: unknown, status = 200, origin: string | null = null) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...cors(origin), "content-type": "application/json; charset=utf-8" },
  });
}
async function sha(value: string) {
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
  if (!equal(await sha(supplied), String(q.data.dashboard_key_sha256))) throw new Error("UNAUTHORIZED_DASHBOARD");
}
function amount(value: unknown): number | null {
  if (value == null || value === "") return null;
  const n = Number(value);
  if (!Number.isFinite(n) || n < 100 || n > 1_000_000 || Math.round(n * 100) / 100 !== n) {
    throw new Error("TREASURY_TARGET_OUT_OF_RANGE");
  }
  return n;
}
async function rpc(name: string, args: Record<string, unknown> = {}) {
  const q = await db.rpc(name, args);
  if (q.error) throw new Error(`${name}:${q.error.message}`);
  return q.data;
}
async function status() {
  const [control, alpha, world, runs] = await Promise.all([
    rpc("brian_system_control_status"),
    db.from("brian_alpha_decisions").select("observed_at,decision_id,asset_id,action").order("observed_at", { ascending: false }).limit(1).maybeSingle(),
    db.from("brian_world_event_snapshots").select("observed_at,event_id,event_kind").order("observed_at", { ascending: false }).limit(1).maybeSingle(),
    db.from("brian_collector_runs").select("collector_id,status,started_at,finished_at,error_class,error_message")
      .not("collector_id", "ilike", "%dip%")
      .order("started_at", { ascending: false }).limit(30),
  ]);
  return {
    status: "OK",
    control,
    last_data: {
      alpha: alpha.error ? null : alpha.data,
      world: world.error ? null : world.data,
      collectors: runs.error ? [] : runs.data,
    },
    dip_touched: false,
    shadow_only: true,
    live_execution: false,
  };
}

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response("ok", { headers: cors(origin) });
  if (req.method !== "POST") return out({ status: "METHOD_NOT_ALLOWED" }, 405, origin);
  try {
    await requireDashboard(req);
    const body = await req.json().catch(() => ({})) as Record<string, unknown>;
    const action = String(body.action ?? "status").toLowerCase();
    const selected = amount(body.starting_equity ?? body.treasury_amount);

    if (action === "status") return out(await status(), 200, origin);

    if (action === "stop") {
      const changed = await rpc("brian_set_system_enabled", { p_enabled: false });
      return out({ status: "STOPPED", result: changed, ...(await status()) }, 200, origin);
    }

    if (action === "set_treasury") {
      if (selected == null) throw new Error("TREASURY_TARGET_REQUIRED");
      await rpc("brian_set_treasury_target", { p_amount: selected });
      const before = await rpc("brian_system_control_status") as Record<string, unknown>;
      const treasury = (before?.treasury ?? null) as Record<string, unknown> | null;
      const openPositions = Number(treasury?.open_positions ?? 0);
      let applied = false;
      if (before?.system_enabled === false && openPositions === 0) {
        await rpc("brian_rebase_treasury_if_safe", { p_amount: selected });
        applied = true;
      }
      return out({ status: applied ? "TREASURY_APPLIED" : "TREASURY_QUEUED", selected, applied, ...(await status()) }, 200, origin);
    }

    if (action === "start" || action === "restart") {
      const before = await rpc("brian_system_control_status") as Record<string, unknown>;
      const treasury = (before?.treasury ?? null) as Record<string, unknown> | null;
      const openPositions = Number(treasury?.open_positions ?? 0);
      const currentStart = Number(treasury?.starting_equity_usd ?? 0);
      const target = selected ?? Number(before?.treasury_target_equity_usd ?? (currentStart || 10000));
      const needsRebase = Number.isFinite(target) && target > 0 && Math.abs(currentStart - target) > 0.005;
      if (needsRebase && openPositions > 0) {
        return out({
          status: "TREASURY_CHANGE_BLOCKED",
          error: "Kasayı değiştirmek için açık SHADOW pozisyon kalmamalı.",
          current_starting_equity_usd: currentStart,
          requested_starting_equity_usd: target,
          open_positions: openPositions,
          dip_touched: false,
          shadow_only: true,
          live_execution: false,
        }, 409, origin);
      }

      if (action === "restart" || needsRebase) await rpc("brian_set_system_enabled", { p_enabled: false });
      if (selected != null) await rpc("brian_set_treasury_target", { p_amount: selected });
      if (needsRebase) await rpc("brian_rebase_treasury_if_safe", { p_amount: target });
      const changed = await rpc("brian_set_system_enabled", { p_enabled: true });
      return out({ status: action === "restart" ? "RESTARTED" : "RUNNING", result: changed, ...(await status()) }, 200, origin);
    }

    return out({ status: "UNKNOWN_ACTION", action }, 400, origin);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const unauthorized = message.includes("UNAUTHORIZED_DASHBOARD");
    console.error("brian-system-control", { message, unauthorized });
    return out({
      status: unauthorized ? "UNAUTHORIZED" : "FAILED_CLOSED",
      error: message,
      dip_touched: false,
      shadow_only: true,
      live_execution: false,
    }, unauthorized ? 401 : 500, origin);
  }
});

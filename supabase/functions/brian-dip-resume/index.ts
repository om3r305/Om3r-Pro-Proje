import { createClient } from "npm:@supabase/supabase-js@2";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE_ROLE, { auth: { persistSession: false, autoRefreshToken: false } });
const EVIDENCE = "AGGRESSIVE_DIP_SHADOW";
const ORIGIN = /^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
const EXACT = new Set([
  "https://monster-coins-pro-seven.vercel.app",
  "https://monster-coins-pro-oemer-yildirim.vercel.app",
  "https://monster-coins-pro-git-brian-2026-oemer-yildirim.vercel.app",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);

function cors(origin: string | null) {
  const allowed = origin && (EXACT.has(origin) || ORIGIN.test(origin))
    ? origin
    : "https://monster-coins-pro-oemer-yildirim.vercel.app";
  return {
    "access-control-allow-origin": allowed,
    "access-control-allow-headers": "content-type,x-brian-dashboard-key",
    "access-control-allow-methods": "POST,OPTIONS",
    "vary": "Origin",
  };
}
function json(body: unknown, status = 200, origin: string | null = null) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...cors(origin) },
  });
}
async function sha256(value: string) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((x) => x.toString(16).padStart(2, "0")).join("");
}
function same(a: string, b: string) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
async function requireDashboard(req: Request) {
  const supplied = (req.headers.get("x-brian-dashboard-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_DASHBOARD");
  const q = await db.from("brian_dashboard_auth")
    .select("dashboard_key_sha256,created_at")
    .order("created_at", { ascending: false }).limit(1).maybeSingle();
  if (q.error || !q.data) throw new Error("UNAUTHORIZED_DASHBOARD");
  if (!same(await sha256(supplied), String(q.data.dashboard_key_sha256))) throw new Error("UNAUTHORIZED_DASHBOARD");
}

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response("ok", { headers: cors(origin) });
  if (req.method !== "POST") return json({ status: "METHOD_NOT_ALLOWED" }, 405, origin);
  try {
    await requireDashboard(req);
    const body = await req.json().catch(() => ({})) as Record<string, unknown>;
    const sessionId = String(body.session_id ?? "").trim();
    const engineToken = String(body.engine_token ?? "").trim();
    if (!/^dip-[A-Za-z0-9-]{10,160}$/.test(sessionId)) throw new Error("INVALID_DIP_SESSION_ID");
    if (engineToken.length < 20 || engineToken.length > 200) throw new Error("INVALID_ENGINE_TOKEN");

    const result = await db.rpc("brian_dip_resume_session", {
      p_event_id: `dip-evt-${crypto.randomUUID()}`,
      p_session_id: sessionId,
      p_engine_token_sha256: await sha256(engineToken),
    });
    if (result.error) throw result.error;
    const row = Array.isArray(result.data) ? result.data[0] : result.data;
    if (!row) throw new Error("DIP_RESUME_NO_ROW");

    return json({
      status: "RESUMED",
      session_id: String(row.session_id),
      started_at: row.requested_at ?? null,
      starting_equity: Number(row.starting_equity),
      trade_notional: Number(row.trade_notional),
      config: row.config ?? {},
      evidence_class: EVIDENCE,
      same_session: true,
      history_preserved: true,
      shadow_only: true,
      live_execution: false,
    }, 200, origin);
  } catch (error) {
    const message = String(error instanceof Error ? error.message : error);
    const unauthorized = message.includes("UNAUTHORIZED_");
    console.error("brian-dip-resume", { message, unauthorized });
    return json({
      status: unauthorized ? "UNAUTHORIZED" : "FAILED_CLOSED",
      error: message,
      evidence_class: EVIDENCE,
      shadow_only: true,
      live_execution: false,
    }, unauthorized ? 401 : 500, origin);
  }
});

import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const VERSION = "brian.realtime-alpha-bridge.v2";
const INTERNAL_KEY_SHA256 = "b0549b2b41a5b832b37455389583e1d166d210490a8c6fe43cda2748aca7c38a";

type Json = Record<string, unknown>;

function out(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}
async function sha256Hex(value: string) {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)),
  );
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function constantTimeEqual(left: string, right: string) {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let i = 0; i < left.length; i++) diff |= left.charCodeAt(i) ^ right.charCodeAt(i);
  return diff === 0;
}
async function auth(req: Request) {
  const supplied = (req.headers.get("x-brian-internal-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_INTERNAL");
  if (!constantTimeEqual(await sha256Hex(supplied), INTERNAL_KEY_SHA256)) {
    throw new Error("UNAUTHORIZED_INTERNAL");
  }
}
function isDuplicate(error: unknown) {
  if (!error || typeof error !== "object") return false;
  const row = error as Record<string, unknown>;
  return String(row.code ?? "") === "23505" ||
    String(row.message ?? "").toLowerCase().includes("duplicate");
}
async function insertIdempotent(table: string, rows: Json[]) {
  let inserted = 0;
  let duplicates = 0;
  for (const row of rows) {
    const result = await db.from(table).insert(row);
    if (!result.error) { inserted++; continue; }
    if (isDuplicate(result.error)) { duplicates++; continue; }
    throw new Error(table + ":" + JSON.stringify(result.error).slice(0, 900));
  }
  return { inserted, duplicates };
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);
  try { await auth(req); } catch { return out({ status: "UNAUTHORIZED" }, 401); }

  try {
    const body = await req.json().catch(() => ({})) as Json;
    if (body.shadow_only !== true || body.live_execution !== false) {
      return out({ status: "REJECTED_SAFETY_CONTRACT" }, 400);
    }
    const costs = Array.isArray(body.costs) ? body.costs.slice(0, 20) as Json[] : [];
    const decisions = Array.isArray(body.decisions) ? body.decisions.slice(0, 20) as Json[] : [];

    const costResult = await insertIdempotent("brian_dynamic_cost_quotes", costs);
    const decisionResult = await insertIdempotent("brian_alpha_decisions", decisions);

    return out({
      status: "CAPTURED_REALTIME_ALPHA_BRIDGE",
      version: VERSION,
      costs_received: costs.length,
      decisions_received: decisions.length,
      costs_inserted: costResult.inserted,
      costs_duplicate: costResult.duplicates,
      decisions_inserted: decisionResult.inserted,
      decisions_duplicate: decisionResult.duplicates,
      shadow_only: true,
      live_execution: false,
    });
  } catch (error) {
    return out({
      status: "FAILED_CLOSED",
      version: VERSION,
      error: String(error instanceof Error ? error.message : error).slice(0, 1200),
      shadow_only: true,
      live_execution: false,
    }, 500);
  }
});

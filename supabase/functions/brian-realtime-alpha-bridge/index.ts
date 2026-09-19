import postgres from "npm:postgres@3.4.7";

const DB_URL = Deno.env.get("SUPABASE_DB_URL") ?? "";
const sql = DB_URL ? postgres(DB_URL, {
  prepare: false,
  max: 1,
  connect_timeout: 3,
  idle_timeout: 5,
  max_lifetime: 30,
}) : null;

const VERSION = "brian.realtime-alpha-bridge.v5";
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
async function writeDirect(costs: Json[], decisions: Json[]) {
  if (!sql) throw new Error("SUPABASE_DB_URL_MISSING");
  const costById = new Map<string, Json>();
  for (const cost of costs) {
    const id = String(cost.quote_id ?? "");
    if (id) costById.set(id, cost);
  }

  let written = 0;
  for (const decision of decisions) {
    const costId = String(decision.source_cost_quote_id ?? "");
    const cost = costId ? (costById.get(costId) ?? null) : null;
    await sql`
      select public.brian_cloudflare_alpha_shadow_write_v1(
        ${sql.json(cost)}::jsonb,
        ${sql.json(decision)}::jsonb
      ) as result
    `;
    written++;
  }
  return { written };
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

    const directResult = await writeDirect(costs, decisions);

    return out({
      status: "CAPTURED_REALTIME_ALPHA_BRIDGE",
      version: VERSION,
      costs_received: costs.length,
      decisions_received: decisions.length,
      decisions_written: directResult.written,
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

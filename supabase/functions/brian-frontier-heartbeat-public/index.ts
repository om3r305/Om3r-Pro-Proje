import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, {
  auth: { persistSession: false, autoRefreshToken: false },
});

function cors(origin: string | null) {
  return {
    "access-control-allow-origin": origin || "*",
    "access-control-allow-headers": "content-type",
    "access-control-allow-methods": "GET,POST,OPTIONS",
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

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
  if (req.method !== "GET" && req.method !== "POST") return out({ error: "GET or POST required" }, 405, origin);

  try {
    const cached = await db.rpc("brian_frontier_heartbeat_cached");
    if (cached.error) throw new Error(cached.error.message);
    const payload = cached.data && typeof cached.data === "object"
      ? cached.data as Record<string, unknown>
      : {};

    return out({
      ...payload,
      status: payload.status || "OK",
      source: "heartbeat_cache",
      dip_touched: false,
      shadow_only: true,
      live_execution: false,
      read_only: true,
    }, 200, origin);
  } catch (error) {
    return out({
      status: "DEGRADED",
      error: error instanceof Error ? error.message : String(error),
      source: "heartbeat_cache",
      dip_touched: false,
      shadow_only: true,
      live_execution: false,
      read_only: true,
    }, 503, origin);
  }
});

import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import postgres from "npm:postgres@3.4.7";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, {
  auth: { persistSession: false, autoRefreshToken: false },
});
const DB_URL = Deno.env.get("SUPABASE_DB_URL") ?? "";
const sql = DB_URL ? postgres(DB_URL, {
  prepare: false,
  max: 1,
  connect_timeout: 3,
  idle_timeout: 3,
  max_lifetime: 30,
}) : null;

const BOOTSTRAP: Record<string, unknown> = {"status":"OK","observed_at":"2026-09-18T20:49:02.123138+00:00","alpha":{"action":"WAIT","asset_id":"crypto:XRPUSDT","direction":0,"observed_at":"2026-09-18T20:26:27.059+00:00","evidence_score":0,"estimated_round_trip_cost_bps":null},"control":{"treasury":{"cash_usd":4996.864157830412,"equity_usd":4996.864157830412,"observed_at":"2026-09-18T20:29:16.826+00:00","snapshot_id":"8964b92a6cf068e88c5db89a3425b66501e13e0f99fd7a12edacddcccb461bde","deployment_pct":0,"open_positions":0,"starting_equity_usd":5000},"system_enabled":true,"managed_jobs_total":30,"managed_jobs_active":18,"core_dispatcher_active":true},"world_run":{"status":"SUCCESS","started_at":"2026-09-18T20:38:54.93+00:00","finished_at":"2026-09-18T20:41:01.572+00:00","event_frames":190,"input_events":489,"asset_impacts":11,"entity_observations":195,"narrative_snapshots":7},"collectors":{"brian-world-brain-v1":{"status":"SUCCESS","started_at":"2026-09-18T20:38:54.93+00:00","error_class":null,"finished_at":"2026-09-18T20:41:01.572+00:00"},"brian-evolution-treasury-v1":{"status":"SUCCESS","started_at":"2026-09-18T20:27:54.723+00:00","error_class":null,"finished_at":"2026-09-18T20:29:37.447+00:00"},"brian-world-discovery-eye-v1":{"status":"SUCCESS","started_at":"2026-09-18T20:40:56.572+00:00","error_class":null,"finished_at":"2026-09-18T20:41:12.456+00:00"},"brian-alpha-decision-compiler-v2":{"status":"SUCCESS","started_at":"2026-09-18T20:26:15.447+00:00","error_class":null,"finished_at":"2026-09-18T20:26:45.773+00:00"},"brian-evolution-ocean-worker-v1":{"status":"SUCCESS","started_at":"2026-09-18T20:07:09.329+00:00","error_class":null,"finished_at":"2026-09-18T20:07:25.741+00:00"}},"source":"heartbeat_bootstrap","read_only":true,"dip_touched":false,"shadow_only":true,"live_execution":false};

let lastGood: Record<string, unknown> = BOOTSTRAP;

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

async function withTimeout<T>(promise: Promise<T>, timeoutMs: number, label: string): Promise<T> {
  let timer: number | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new Error(label)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

function normalizePayload(payload: unknown, source: string): Record<string, unknown> {
  const row = payload && typeof payload === "object" ? payload as Record<string, unknown> : {};
  lastGood = {
    ...row,
    status: row.status || "OK",
    source,
    transport_degraded: false,
    dip_touched: false,
    shadow_only: true,
    live_execution: false,
    read_only: true,
  };
  return lastGood;
}

async function readDirectDb(timeoutMs = 2800): Promise<Record<string, unknown>> {
  if (!sql) throw new Error("SUPABASE_DB_URL_UNAVAILABLE");
  const rows = await withTimeout(
    sql<{ payload: Record<string, unknown> }[]>\`select public.brian_frontier_heartbeat_cached() as payload\`,
    timeoutMs,
    "heartbeat-direct-db-timeout",
  );
  const payload = rows?.[0]?.payload;
  if (!payload) throw new Error("heartbeat-direct-db-empty");
  return normalizePayload(payload, "heartbeat_cache_direct_db");
}

async function readRestCache(timeoutMs = 3500): Promise<Record<string, unknown>> {
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= 3; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort("heartbeat-cache-timeout"), timeoutMs);
    try {
      const cached = await db.rpc("brian_frontier_heartbeat_cached").abortSignal(controller.signal);
      if (cached.error) throw new Error(cached.error.message);
      return normalizePayload(cached.data, "heartbeat_cache_rest");
    } catch (error) {
      lastError = error;
      if (attempt < 3) await new Promise((resolve) => setTimeout(resolve, 150 * attempt));
    } finally {
      clearTimeout(timer);
    }
  }
  throw lastError ?? new Error("heartbeat-cache-unavailable");
}

async function readCache(): Promise<Record<string, unknown>> {
  let directError: unknown = null;
  try {
    return await readDirectDb();
  } catch (error) {
    directError = error;
  }
  try {
    return await readRestCache();
  } catch (restError) {
    throw new Error(`direct=${directError instanceof Error ? directError.message : String(directError)}; rest=${restError instanceof Error ? restError.message : String(restError)}`);
  }
}

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
  if (req.method !== "GET" && req.method !== "POST") return out({ error: "GET or POST required" }, 405, origin);

  try {
    return out(await readCache(), 200, origin);
  } catch (error) {
    return out({
      ...lastGood,
      status: lastGood.status || "OK",
      source: "heartbeat_last_good",
      transport_degraded: true,
      transport_error: error instanceof Error ? error.message : String(error),
      served_at: new Date().toISOString(),
      dip_touched: false,
      shadow_only: true,
      live_execution: false,
      read_only: true,
    }, 200, origin);
  }
});

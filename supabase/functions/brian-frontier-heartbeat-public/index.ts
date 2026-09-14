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

function str(value: unknown): string | null {
  return value == null ? null : String(value);
}

function num(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function sanitizeCollector(value: unknown) {
  if (!value || typeof value !== "object") return null;
  const row = value as Record<string, unknown>;
  return {
    status: str(row.status),
    started_at: str(row.started_at),
    finished_at: str(row.finished_at),
    error_class: str(row.error_class),
  };
}

Deno.serve(async (req: Request) => {
  const origin = req.headers.get("origin");
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
  if (req.method !== "GET" && req.method !== "POST") return out({ error: "GET or POST required" }, 405, origin);

  try {
    const snapshot = await db.rpc("brian_frontier_heartbeat_snapshot");
    if (snapshot.error) throw new Error(snapshot.error.message);
    const raw = snapshot.data && typeof snapshot.data === "object"
      ? snapshot.data as Record<string, unknown>
      : {};
    const control = raw.control && typeof raw.control === "object"
      ? raw.control as Record<string, unknown>
      : {};
    const treasury = control.treasury && typeof control.treasury === "object"
      ? control.treasury as Record<string, unknown>
      : {};
    const alpha = raw.alpha && typeof raw.alpha === "object"
      ? raw.alpha as Record<string, unknown>
      : {};
    const worldRun = raw.world_run && typeof raw.world_run === "object"
      ? raw.world_run as Record<string, unknown>
      : {};
    const collectors = raw.collectors && typeof raw.collectors === "object"
      ? raw.collectors as Record<string, unknown>
      : {};

    const safeCollectors: Record<string, unknown> = {};
    for (const id of [
      "brian-world-brain-v1",
      "brian-world-discovery-eye-v1",
      "brian-alpha-decision-compiler-v2",
      "brian-evolution-treasury-v1",
      "brian-evolution-orchestrator-v1",
      "brian-evolution-researcher-v1",
      "brian-evolution-sandbox-v1",
      "brian-evolution-ocean-worker-v1",
    ]) {
      safeCollectors[id] = sanitizeCollector(collectors[id]);
    }

    return out({
      status: "OK",
      observed_at: new Date().toISOString(),
      alpha: Object.keys(alpha).length ? {
        action: str(alpha.action),
        asset_id: str(alpha.asset_id),
        direction: num(alpha.direction),
        observed_at: str(alpha.observed_at),
      } : null,
      control: {
        system_enabled: control.system_enabled !== false,
        managed_jobs_total: num(control.managed_jobs_total),
        managed_jobs_active: num(control.managed_jobs_active),
        treasury: Object.keys(treasury).length ? {
          observed_at: str(treasury.observed_at),
          equity_usd: num(treasury.equity_usd),
          cash_usd: num(treasury.cash_usd),
          open_positions: num(treasury.open_positions),
          starting_equity_usd: num(treasury.starting_equity_usd),
        } : null,
      },
      world_run: Object.keys(worldRun).length ? {
        status: str(worldRun.status),
        started_at: str(worldRun.started_at),
        finished_at: str(worldRun.finished_at),
        event_frames: num(worldRun.event_frames),
        input_events: num(worldRun.input_events),
        asset_impacts: num(worldRun.asset_impacts),
        entity_observations: num(worldRun.entity_observations),
        narrative_snapshots: num(worldRun.narrative_snapshots),
      } : null,
      collectors: safeCollectors,
      dip_touched: false,
      shadow_only: true,
      live_execution: false,
      read_only: true,
    }, 200, origin);
  } catch (error) {
    return out({
      status: "DEGRADED",
      error: error instanceof Error ? error.message : String(error),
      dip_touched: false,
      shadow_only: true,
      live_execution: false,
      read_only: true,
    }, 503, origin);
  }
});

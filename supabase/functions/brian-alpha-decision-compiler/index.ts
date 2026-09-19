import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });

const VERSION = "brian.core-alpha-realtime-sync.v1";
const COLLECTOR_ID = "brian-alpha-decision-compiler-v2";
const EVIDENCE = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const EXPECTED_CRON_SHA256 = "814a5df4f8d6e3b15f1b9ac19a4ea823ad69eedc52caa6ad7573fde7aa96eaab";
const EXPORT_URL = "https://dliediwlldojkfjzlznm.supabase.co/functions/v1/brian-realtime-alpha-export";

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
function errText(e: unknown) {
  if (e instanceof Error) return `${e.name}: ${e.message}`;
  try { return JSON.stringify(e); } catch { return String(e); }
}
async function sha256Hex(value: string) {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function constantTimeEqual(a: string, b: string) {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
async function requireCron(req: Request) {
  const key = (req.headers.get("x-brian-cron-key") ?? "").trim();
  if (!key) throw new Error("UNAUTHORIZED_CRON");
  const digest = await sha256Hex(key);
  if (!constantTimeEqual(digest, EXPECTED_CRON_SHA256)) throw new Error("UNAUTHORIZED_CRON");
  return key;
}
async function recordRun(startedAt: string, status: "SUCCESS" | "FAILED", observed: number, stored: number, upstreamVersion: string | null, error?: unknown) {
  const finishedAt = new Date().toISOString();
  const runId = await sha256Hex(`${COLLECTOR_ID}|sync|${startedAt}|${finishedAt}|${status}`);
  const q = await db.from("brian_collector_runs").insert({
    run_id: runId,
    collector_id: COLLECTOR_ID,
    started_at: startedAt,
    finished_at: finishedAt,
    status,
    observed_records: observed,
    stored_records: stored,
    degraded_sources: [],
    error_class: error ? "ALPHA_REALTIME_SYNC_ERROR" : null,
    error_message: error ? errText(error).slice(0, 1500) : null,
    evidence_class: EVIDENCE,
    shadow_only: true,
    live_execution: false,
    metadata: {
      mode: "REALTIME_ALPHA_SYNC",
      sync_version: VERSION,
      upstream_version: upstreamVersion,
      source_project: "brian-realtime",
      shadow_only: true,
      live_execution: false,
    },
  });
  if (q.error) console.error("sync run receipt", q.error.message);
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);
  const startedAt = new Date().toISOString();

  let bridgeKey = "";
  try {
    bridgeKey = await requireCron(req);
  } catch (e) {
    return out({ status: "UNAUTHORIZED", version: VERSION, error: errText(e) }, 401);
  }

  try {
    const upstream = await fetch(EXPORT_URL, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-brian-internal-key": bridgeKey,
      },
      body: JSON.stringify({ since_minutes: 20, limit: 500 }),
      signal: AbortSignal.timeout(15_000),
    });
    const raw = await upstream.text();
    let payload: Json = {};
    try { payload = JSON.parse(raw) as Json; } catch {
      throw new Error(`UPSTREAM_JSON_INVALID:${raw.slice(0,300)}`);
    }
    if (!upstream.ok || String(payload.status ?? "") !== "SUCCESS") {
      throw new Error(`UPSTREAM_FAILED:${upstream.status}:${String(payload.status ?? "")}`);
    }

    const rows = Array.isArray(payload.rows) ? payload.rows.filter((v): v is Json => Boolean(v && typeof v === "object")) : [];
    const syncedAt = new Date().toISOString();
    let stored = 0;

    for (let i = 0; i < rows.length; i += 50) {
      const chunk = rows.slice(i, i + 50).map((row) => ({
        ...row,
        source_cost_quote_id: null,
        metadata: {
          ...((row.metadata && typeof row.metadata === "object" && !Array.isArray(row.metadata)) ? row.metadata as Json : {}),
          core_sync_source: "brian-realtime",
          core_sync_version: VERSION,
          core_synced_at: syncedAt,
        },
        shadow_only: true,
        live_execution: false,
      }));

      const q = await db.from("brian_alpha_decisions")
        .upsert(chunk, { onConflict: "decision_id", ignoreDuplicates: true });
      if (q.error) throw new Error(`CORE_ALPHA_UPSERT:${q.error.message}`);
      stored += chunk.length;
    }

    await recordRun(startedAt, "SUCCESS", rows.length, stored, String(payload.version ?? ""));
    return out({
      status: "SUCCESS",
      version: VERSION,
      upstream_version: payload.version ?? null,
      received: rows.length,
      stored_attempted: stored,
      source_project: "brian-realtime",
      shadow_only: true,
      live_execution: false,
    });
  } catch (e) {
    await recordRun(startedAt, "FAILED", 0, 0, null, e);
    return out({
      status: "FAILED_CLOSED",
      version: VERSION,
      error: errText(e).slice(0, 1200),
      shadow_only: true,
      live_execution: false,
    }, 500);
  }
});

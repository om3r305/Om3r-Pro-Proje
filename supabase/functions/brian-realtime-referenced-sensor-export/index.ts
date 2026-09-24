import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const VERSION = "brian.realtime-referenced-sensor-export.v1";
const MAX_IDS = 500;
const ALLOWED_SHA256 = new Set([
  "b0549b2b41a5b832b37455389583e1d166d210490a8c6fe43cda2748aca7c38a",
  "814a5df4f8d6e3b15f1b9ac19a4ea823ad69eedc52caa6ad7573fde7aa96eaab",
]);

function out(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

async function sha256Hex(value: string): Promise<string> {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)),
  );
  return [...digest]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function requireBridgeAuth(req: Request): Promise<void> {
  const supplied = (req.headers.get("x-brian-internal-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_INTERNAL");
  const digest = await sha256Hex(supplied);
  if (!ALLOWED_SHA256.has(digest)) {
    throw new Error("UNAUTHORIZED_INTERNAL");
  }
}

function normalizeIds(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  const ids = value
    .map((item) => String(item ?? "").trim())
    .filter((item) => /^[a-f0-9]{64}$/i.test(item));
  return [...new Set(ids)].slice(0, MAX_IDS);
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);

  try {
    await requireBridgeAuth(req);
  } catch {
    return out({ status: "UNAUTHORIZED" }, 401);
  }

  const body = await req.json().catch(() => ({})) as Record<string, unknown>;
  const ids = normalizeIds(body.observation_ids);
  if (!ids.length) {
    return out({
      status: "SUCCESS",
      version: VERSION,
      requested: 0,
      rows: [],
      count: 0,
      shadow_only: true,
      live_execution: false,
    });
  }

  const q = await db
    .from("brian_sensor_observations")
    .select(
      "observation_id,eye_id,template_id,asset_id,market_domain,sensor_family,horizon,independent_group,observed_at,direction,strength,confidence,reliability,available,source_ids,reason,evidence_class,shadow_only,live_execution,metadata,created_at",
    )
    .in("observation_id", ids)
    .eq("evidence_class", "PROSPECTIVE_DEVELOPMENT_SHADOW")
    .eq("shadow_only", true)
    .eq("live_execution", false)
    .order("observed_at", { ascending: true })
    .limit(MAX_IDS);

  if (q.error) {
    return out({
      status: "FAILED_CLOSED",
      version: VERSION,
      error: q.error.message,
      rows: [],
      shadow_only: true,
      live_execution: false,
    }, 500);
  }

  const rows = (q.data ?? []).filter((row) => ids.includes(String(row.observation_id)));
  return out({
    status: "SUCCESS",
    version: VERSION,
    requested: ids.length,
    rows,
    count: rows.length,
    missing: ids.length - rows.length,
    shadow_only: true,
    live_execution: false,
  });
});

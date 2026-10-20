// brian-niulai4-pilot-ingest: the ONLY write path from the local NIULAI4 pilot into CORE.
// POST only, x-brian-pilot-key. Refuses everything (503 PILOT_INGEST_NOT_CONFIGURED) until the
// operator stores a key hash and enables the single brian_niulai4_pilot_auth row. It can write
// only the pilot's own runs/heartbeats/reports through brian_record_niulai4_pilot_v1, which
// re-validates shadow-only, fingerprint, clock range and rate limits. No CORS: not a browser API.

import postgres from "npm:postgres@3.4.5";
import { checkPilotKey, parseIngestBody } from "../_shared/prediction_status.ts";

const headers = { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" };
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers });

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return response({ error: "METHOD_NOT_ALLOWED" }, 405);
  const key = (req.headers.get("x-brian-pilot-key") || "").trim();
  const text = await req.text();
  const sql = postgres(Deno.env.get("SUPABASE_DB_URL")!, { prepare: false, max: 1, idle_timeout: 1, connect_timeout: 8, max_lifetime: 30 });
  try {
    const [row] = await sql`select enabled, ingest_key_sha256 from public.brian_niulai4_pilot_auth where auth_id = 'pilot-ingest-v1'`;
    const auth = await checkPilotKey(key, row ? { enabled: Boolean(row.enabled), ingest_key_sha256: row.ingest_key_sha256 ?? null } : null);
    if (!auth.ok) return response({ error: auth.error }, auth.status);
    const body = parseIngestBody(text);
    if (!body.ok) return response({ status: "REJECTED", reason: body.error }, 400);
    const [r] = await sql`select public.brian_record_niulai4_pilot_v1(${sql.json(body.value as never)}) as v`;
    const result = r.v as { status: string };
    const code = result.status === "STORED" || result.status === "DUPLICATE" ? 200 : result.status === "RATE_LIMITED" ? 429 : 400;
    return response(result, code);
  } catch (e) {
    return response({ error: "INGEST_UNAVAILABLE", detail: (e as Error)?.name ?? "Error" }, 503);
  } finally {
    await sql.end({ timeout: 1 }).catch(() => {});
  }
});

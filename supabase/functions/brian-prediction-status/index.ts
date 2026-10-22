// brian-prediction-status: read-only UI status for Prediction Intelligence and the NIULAI4 pilot.
// Contract: docs/BRIAN_PREDICTION_UI_CONTRACT.md. POST only, x-brian-dashboard-key (same auth row as
// the other dashboard endpoints). One read-only transaction; each section fails independently and
// is then reported UNAVAILABLE instead of being hidden or defaulted.

import postgres from "npm:postgres@3.4.5";
import { buildPredictionStatus, constantTimeEqual, sha256Hex } from "../_shared/prediction_status.ts";

const headers = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
  "access-control-allow-origin": "*",
  "access-control-allow-headers": "content-type,x-brian-dashboard-key,x-brian-cron-key",
  "access-control-allow-methods": "POST,OPTIONS",
};
const response = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers });
let cache: { body: unknown; at: number } | null = null;
const CACHE_MS = 30_000;

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers });
  if (req.method !== "POST") return response({ error: "METHOD_NOT_ALLOWED" }, 405);
  const key = (req.headers.get("x-brian-dashboard-key") || req.headers.get("x-brian-cron-key") || "").trim();
  if (!key) return response({ error: "UNAUTHORIZED_DASHBOARD" }, 401);
  const sql = postgres(Deno.env.get("SUPABASE_DB_URL")!, { prepare: false, max: 1, idle_timeout: 1, connect_timeout: 8, max_lifetime: 30 });
  try {
    const [auth] = await sql`select dashboard_key_sha256, cron_key_sha256 from public.brian_dashboard_auth where auth_id = 'control-v3'`;
    const hash = await sha256Hex(key);
    if (!auth || ![auth.dashboard_key_sha256, auth.cron_key_sha256].some((v) => v && constantTimeEqual(hash, String(v)))) {
      return response({ error: "UNAUTHORIZED_DASHBOARD" }, 401);
    }
    if (cache && Date.now() - cache.at < CACHE_MS) {
      return response({ ...(cache.body as object), cache_age_seconds: (Date.now() - cache.at) / 1000 });
    }
    const parts: { prediction: unknown; calibrationInputs: unknown; pilot: unknown; research: unknown } = { prediction: null, calibrationInputs: null, pilot: null, research: null };
    let integration: unknown = null;
    const errors: { prediction?: string; calibration?: string; pilot?: string; research?: string; integration?: string } = {};
    let observedAt = new Date();
    await sql.begin(async (tx) => {
      await tx`set local statement_timeout = '8s'`;
      await tx`set transaction isolation level repeatable read, read only`;
      const [clock] = await tx`select transaction_timestamp() as at`;
      observedAt = new Date(clock.at);
      const read = async (name: keyof typeof errors, q: () => Promise<unknown>) => {
        try {
          return await tx.savepoint(async () => await q());
        } catch (e) {
          errors[name] = `READ_FAILED:${(e as Error)?.name ?? "Error"}`;
          return null;
        }
      };
      parts.prediction = await read("prediction", async () => (await tx`select public.brian_prediction_status_v1(transaction_timestamp()) as v`)[0].v);
      parts.calibrationInputs = await read("calibration", async () => (await tx`select public.brian_prediction_calibration_inputs_v1(transaction_timestamp(), interval '24 hours') as v`)[0].v);
      parts.pilot = await read("pilot", async () => (await tx`select public.brian_niulai4_pilot_status_v1(transaction_timestamp()) as v`)[0].v);
      parts.research = await read("research", async () => (await tx`select public.brian_research_status_v1(transaction_timestamp()) as v`)[0].v);
      integration = await read("integration", async () => (await tx`select public.brian_treasury_integration_status_v1(transaction_timestamp()) as v`)[0].v);
    });
    // Whole-system wiring (treasury, evidence ledger, sensors, studies, local collectors): see the UI contract.
    const body = {
      ...buildPredictionStatus(parts, errors, observedAt),
      integration: integration ?? { status: "UNAVAILABLE", error: errors.integration ?? "READ_FAILED", integrated: false },
    };
    cache = { body, at: Date.now() };
    return response({ ...body, cache_age_seconds: 0 });
  } catch (e) {
    return response({ error: "STATUS_UNAVAILABLE", detail: (e as Error)?.name ?? "Error", shadow_only: true, trade_signal: false }, 503);
  } finally {
    await sql.end({ timeout: 1 }).catch(() => {});
  }
});

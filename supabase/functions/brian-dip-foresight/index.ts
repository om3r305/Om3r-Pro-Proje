import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import {
  authorizeReaderOrCron,
  withCollectorLease,
} from "../_shared/dip_v8_auth.ts";
import { POLICY_VERSION } from "../_shared/dip_v8.ts";
import { readForesight, resolvePending } from "./resolver.ts";
const db = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
  { auth: { persistSession: false, autoRefreshToken: false } },
);
const EXACT = new Set([
  "https://monster-coins-pro-seven.vercel.app",
  "https://monster-coins-pro-oemer-yildirim.vercel.app",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
]);
const ORIGIN =
  /^https:\/\/monster-coins(?:-pro)?-[a-z0-9-]*oemer-yildirim\.vercel\.app$/i;
function headers(origin: string | null) {
  return {
    "content-type": "application/json",
    "cache-control": "no-store",
    "access-control-allow-origin":
      origin && (EXACT.has(origin) || ORIGIN.test(origin))
        ? origin
        : "https://monster-coins-pro-oemer-yildirim.vercel.app",
    "access-control-allow-headers":
      "content-type,x-brian-dashboard-key,x-brian-cron-key,authorization,apikey",
    "access-control-allow-methods": "POST,OPTIONS",
    "vary": "Origin",
  };
}
Deno.serve(async (req: Request) => {
  const h = headers(req.headers.get("origin"));
  if (req.method === "OPTIONS") return new Response("ok", { headers: h });
  if (req.method !== "POST") {
    return new Response("method", { status: 405, headers: h });
  }
  try {
    const mode = await authorizeReaderOrCron(db, req);
    // Dashboard requests never persist decisions or resolve outcomes.
    if (mode === "cron") {
      const result = await withCollectorLease(
        db,
        "brian-dip-foresight-v8",
        (_owner, guard) => resolvePending(db, guard),
      );
      return Response.json(
        result.contended ? { status: "WAIT_LEASE" } : result.value,
        { headers: h },
      );
    }
    const body = await req.json().catch(() => ({}));
    return Response.json(
      await readForesight(
        db,
        typeof body.session_id === "string" ? body.session_id : undefined,
      ),
      { headers: h },
    );
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return Response.json({
      status: "FAILED_CLOSED",
      error: message,
      policy_version: POLICY_VERSION,
      shadow_only: true,
      live_execution: false,
    }, { status: message.includes("UNAUTHORIZED") ? 401 : 500, headers: h });
  }
});

import { createClient } from "npm:@supabase/supabase-js@2";
import { POLICY_VERSION } from "../_shared/dip_v8.ts";
import { requireCronAuth, withCollectorLease } from "../_shared/dip_v8_auth.ts";
import { runWorker } from "./worker.ts";
const db = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
  { auth: { persistSession: false, autoRefreshToken: false } },
);
Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return new Response("method", { status: 405 });
  try {
    await requireCronAuth(db, req);
    const lease = await withCollectorLease(
      db,
      "brian-dip-shadow-worker-v8",
      (owner, assertOwned) => runWorker(db, owner, assertOwned),
    );
    return Response.json(
      lease.contended
        ? { status: "WAIT_LEASE", worker_version: POLICY_VERSION }
        : lease.value,
      { headers: { "cache-control": "no-store" } },
    );
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    console.error("dip-v8", message);
    return Response.json({
      status: "FAILED_CLOSED",
      error: message,
      worker_version: POLICY_VERSION,
      shadow_only: true,
      live_execution: false,
    }, {
      status: message.includes("UNAUTHORIZED") ? 401 : 500,
      headers: { "cache-control": "no-store" },
    });
  }
});

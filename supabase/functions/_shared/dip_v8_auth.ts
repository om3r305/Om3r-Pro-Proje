import type { SupabaseClient } from "npm:@supabase/supabase-js@2.116.0";
import { hash, same } from "./dip_v8.ts";
export async function requireCronAuth(
  db: SupabaseClient,
  req: Request,
): Promise<void> {
  const supplied = (req.headers.get("x-brian-cron-key") || "").trim();
  if (!supplied) throw Error("UNAUTHORIZED_CRON");
  const q = await db.from("brian_dashboard_auth").select("cron_key_sha256").eq(
    "auth_id",
    "control-v3",
  ).single();
  if (q.error || !q.data) throw Error("CRON_AUTH_UNAVAILABLE");
  if (!same(await hash(supplied), String(q.data.cron_key_sha256 || ""))) {
    throw Error("UNAUTHORIZED_CRON");
  }
}
export async function authorizeReaderOrCron(
  db: SupabaseClient,
  req: Request,
): Promise<"reader" | "cron"> {
  const key = (req.headers.get("x-brian-dashboard-key") || "").trim();
  if (!key) {
    await requireCronAuth(db, req);
    return "cron";
  }
  const q = await db.from("brian_dashboard_auth").select("dashboard_key_sha256")
    .order("created_at", { ascending: false }).limit(1).maybeSingle();
  if (
    q.error || !q.data ||
    !same(await hash(key), String(q.data.dashboard_key_sha256 || ""))
  ) throw Error("UNAUTHORIZED_DASHBOARD");
  return "reader";
}
export async function withCollectorLease<T>(
  db: SupabaseClient,
  collector: string,
  work: (owner: string, assertOwned: () => void) => Promise<T>,
): Promise<{ contended: boolean; value?: T }> {
  const owner = crypto.randomUUID(), seconds = 55;
  const a = await db.rpc("brian_acquire_collector_lease", {
    p_collector_id: collector,
    p_owner_token: owner,
    p_lease_seconds: seconds,
  });
  if (a.error) throw Error("LEASE_READ_FAILED:" + a.error.message);
  if (a.data !== true) return { contended: true };
  let lost = false, finished = false;
  const heartbeat = setInterval(async () => {
    try {
      const r = await db.rpc("brian_renew_collector_lease", {
        p_collector_id: collector,
        p_owner_token: owner,
        p_lease_seconds: seconds,
      });
      if (r.error || r.data !== true) lost = true;
    } catch {
      lost = true;
    }
    if (lost || finished) clearInterval(heartbeat);
  }, 18_000);
  try {
    return {
      contended: false,
      value: await work(owner, () => {
        if (lost) throw Error("V8_LEASE_LOST");
      }),
    };
  } finally {
    finished = true;
    clearInterval(heartbeat);
    await db.rpc("brian_release_collector_lease", {
      p_collector_id: collector,
      p_owner_token: owner,
    });
  }
}

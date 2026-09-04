import type { SupabaseClient } from "npm:@supabase/supabase-js@2";

const DEFAULT_AUTH_ID = "control-v2";

function constantTimeEqual(left: string, right: string): boolean {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let i = 0; i < left.length; i++) diff |= left.charCodeAt(i) ^ right.charCodeAt(i);
  return diff === 0;
}

async function sha256Hex(value: string): Promise<string> {
  const bytes = new TextEncoder().encode(value);
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Require the same hashed cron key used by Brian Control Center before a service-role Edge
 * Function can perform any write. This is intentionally independent of Supabase gateway JWT
 * verification: production should keep verify_jwt enabled as an outer gate, while this secret
 * remains the application-level cron authorization boundary.
 */
export async function requireCronAuth(
  req: Request,
  supabase: SupabaseClient,
  authId = DEFAULT_AUTH_ID,
): Promise<void> {
  const supplied = (req.headers.get("x-brian-cron-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_CRON");

  const result = await supabase.from("brian_dashboard_auth")
    .select("cron_key_sha256")
    .eq("auth_id", authId)
    .single();
  if (result.error || !result.data) {
    throw new Error(`CRON_AUTH_UNAVAILABLE:${result.error?.message ?? "missing auth row"}`);
  }

  const expected = String(result.data.cron_key_sha256 ?? "");
  if (!expected || !constantTimeEqual(await sha256Hex(supplied), expected)) {
    throw new Error("UNAUTHORIZED_CRON");
  }
}

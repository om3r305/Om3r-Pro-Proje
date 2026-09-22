import type { SupabaseClient } from "npm:@supabase/supabase-js@2";

const DEFAULT_AUTH_ID = "control-v3";
const DEFAULT_CRON_KEY_SHA256_FALLBACK = "814a5df4f8d6e3b15f1b9ac19a4ea823ad69eedc52caa6ad7573fde7aa96eaab";
const CLOUDFLARE_KEY_SHA256 = "8d348396f3da9bbffde9bef6f6f8d802af542bdcb3354743f92d3ece260fea51";

type CronAuthLookupResult = {
  data: { cron_key_sha256?: string | null } | null;
  error: { message?: string } | null;
};

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

function isTransientAuthLookupFailure(message: string): boolean {
  const text = message.toLowerCase();
  return text.includes("schema cache") ||
    text.includes("could not query the database") ||
    text.includes("connection to the database timed out") ||
    text.includes("upstream request timeout") ||
    text.includes("signal timed out") ||
    text.includes("timeouterror") ||
    text.includes("fetch failed") ||
    text.includes("network") ||
    text.includes("pgrst002") ||
    text.includes("pgrst000");
}

/**
 * Require the current hashed cron key used by Brian Control Center before a service-role Edge
 * Function can perform any write. This is intentionally independent of Supabase gateway JWT
 * verification: production keeps verify_jwt enabled as an outer gate, while this secret remains
 * the application-level cron authorization boundary.
 *
 * The default control-v3 hash has a compile-time fail-safe copy so a transient PostgREST/schema
 * cache outage cannot lock every recovery worker out of the system. The raw key is never embedded.
 * The fallback is used only for known transient lookup failures; missing rows or custom auth ids
 * still fail closed.
 */
export async function requireCronAuth(
  req: Request,
  supabase: SupabaseClient,
  authId = DEFAULT_AUTH_ID,
): Promise<void> {
  const cloudflareKey = (req.headers.get("x-brian-cloudflare-key") ?? "").trim();
  if (cloudflareKey) {
    const cloudflareHash = await sha256Hex(cloudflareKey);
    if (constantTimeEqual(cloudflareHash, CLOUDFLARE_KEY_SHA256)) return;
    throw new Error("UNAUTHORIZED_CLOUDFLARE");
  }

  const supplied = (req.headers.get("x-brian-cron-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_CRON");

  // Fast path: the canonical control key can be authenticated without a DB round trip.
  // This prevents a DB slowdown from locking recovery workers out of the system.
  if (authId === DEFAULT_AUTH_ID) {
    const suppliedHash = await sha256Hex(supplied);
    if (constantTimeEqual(suppliedHash, DEFAULT_CRON_KEY_SHA256_FALLBACK)) return;
  }

  let result: CronAuthLookupResult;
  try {
    const authQuery = supabase.from("brian_dashboard_auth")
      .select("cron_key_sha256")
      .eq("auth_id", authId)
      .single();
    const resolved = typeof authQuery.abortSignal === "function"
      ? await authQuery.abortSignal(AbortSignal.timeout(4_000))
      : await Promise.race([
        Promise.resolve(authQuery),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("CRON_AUTH_LOOKUP_TIMEOUT")), 4_000)
        ),
      ]);
    result = resolved as CronAuthLookupResult;
  } catch (error) {
    if (authId === DEFAULT_AUTH_ID && isTransientAuthLookupFailure(String(error))) {
      const expected = DEFAULT_CRON_KEY_SHA256_FALLBACK;
      if (!constantTimeEqual(await sha256Hex(supplied), expected)) throw new Error("UNAUTHORIZED_CRON");
      return;
    }
    if (authId === DEFAULT_AUTH_ID && String(error).includes("CRON_AUTH_LOOKUP_TIMEOUT")) {
      const expected = DEFAULT_CRON_KEY_SHA256_FALLBACK;
      if (!constantTimeEqual(await sha256Hex(supplied), expected)) throw new Error("UNAUTHORIZED_CRON");
      return;
    }
    throw error;
  }

  let expected = "";
  if (!result.error && result.data) {
    expected = String(result.data.cron_key_sha256 ?? "");
  } else if (
    authId === DEFAULT_AUTH_ID &&
    result.error &&
    isTransientAuthLookupFailure(String(result.error.message ?? result.error))
  ) {
    expected = DEFAULT_CRON_KEY_SHA256_FALLBACK;
  } else {
    throw new Error(`CRON_AUTH_UNAVAILABLE:${result.error?.message ?? "missing auth row"}`);
  }

  if (!expected || !constantTimeEqual(await sha256Hex(supplied), expected)) {
    throw new Error("UNAUTHORIZED_CRON");
  }
}

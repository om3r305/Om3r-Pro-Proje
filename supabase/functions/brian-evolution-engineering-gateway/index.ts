import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const ISSUER = "https://token.actions.githubusercontent.com";
const AUDIENCE = "brian-evolution-engineering-v1";
const REPOSITORY = "om3r305/Om3r-Pro-Proje";
const ALLOWED_WORKFLOWS = new Set([
  ".github/workflows/brian-engineer.yml",
  ".github/workflows/brian-engineer-release.yml",
]);

type Claims = Record<string, unknown>;
let jwksCache: { expires: number; keys: JsonWebKey[] } | null = null;

function out(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}
function b64url(value: string): Uint8Array {
  const normalized = value.replaceAll("-", "+").replaceAll("_", "/") + "=".repeat((4 - value.length % 4) % 4);
  return Uint8Array.from(atob(normalized), (c) => c.charCodeAt(0));
}
function jsonPart(value: string): Record<string, unknown> {
  return JSON.parse(new TextDecoder().decode(b64url(value))) as Record<string, unknown>;
}
function audienceOk(value: unknown): boolean {
  return typeof value === "string" ? value === AUDIENCE : Array.isArray(value) && value.map(String).includes(AUDIENCE);
}
async function jwks(): Promise<JsonWebKey[]> {
  if (jwksCache && jwksCache.expires > Date.now()) return jwksCache.keys;
  const response = await fetch(`${ISSUER}/.well-known/jwks`, { headers: { accept: "application/json" } });
  if (!response.ok) throw new Error(`OIDC_JWKS_HTTP_${response.status}`);
  const body = await response.json() as { keys?: JsonWebKey[] };
  if (!Array.isArray(body.keys) || !body.keys.length) throw new Error("OIDC_JWKS_EMPTY");
  jwksCache = { expires: Date.now() + 60 * 60_000, keys: body.keys };
  return body.keys;
}
async function verifyGithubOidc(req: Request): Promise<Claims> {
  const auth = req.headers.get("authorization") ?? "";
  if (!auth.startsWith("Bearer ")) throw new Error("OIDC_BEARER_REQUIRED");
  const token = auth.slice(7).trim();
  const parts = token.split(".");
  if (parts.length !== 3) throw new Error("OIDC_FORMAT_INVALID");
  const header = jsonPart(parts[0]);
  const claims = jsonPart(parts[1]);
  if (header.alg !== "RS256" || typeof header.kid !== "string") throw new Error("OIDC_HEADER_INVALID");
  const key = (await jwks()).find((candidate) => candidate.kid === header.kid);
  if (!key) throw new Error("OIDC_KID_UNKNOWN");
  const cryptoKey = await crypto.subtle.importKey("jwk", key, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]);
  const signed = new TextEncoder().encode(`${parts[0]}.${parts[1]}`);
  const verified = await crypto.subtle.verify("RSASSA-PKCS1-v1_5", cryptoKey, b64url(parts[2]), signed);
  if (!verified) throw new Error("OIDC_SIGNATURE_INVALID");

  const now = Math.floor(Date.now() / 1000);
  if (claims.iss !== ISSUER || !audienceOk(claims.aud)) throw new Error("OIDC_ISSUER_OR_AUDIENCE_INVALID");
  if (Number(claims.exp ?? 0) < now - 15 || Number(claims.nbf ?? 0) > now + 15) throw new Error("OIDC_TIME_INVALID");
  if (claims.repository !== REPOSITORY) throw new Error("OIDC_REPOSITORY_DENIED");
  const workflowRef = String(claims.job_workflow_ref ?? "");
  const workflowPath = workflowRef.startsWith(`${REPOSITORY}/`) ? workflowRef.slice(REPOSITORY.length + 1).split("@")[0] : "";
  if (!ALLOWED_WORKFLOWS.has(workflowPath)) throw new Error("OIDC_WORKFLOW_DENIED");
  return claims;
}
function cleanPayload(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const text = JSON.stringify(value);
  if (text.length > 60_000) throw new Error("PAYLOAD_TOO_LARGE");
  return value as Record<string, unknown>;
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ status: "METHOD_NOT_ALLOWED" }, 405);
  try {
    const claims = await verifyGithubOidc(req);
    const body = cleanPayload(await req.json().catch(() => ({})));
    const action = String(body.action ?? "");
    const eventName = String(claims.event_name ?? "");

    if (action === "claim") {
      if (!new Set(["schedule", "workflow_dispatch"]).has(eventName)) throw new Error("CLAIM_EVENT_DENIED");
      if (claims.ref !== "refs/heads/brian-2026") throw new Error("CLAIM_REF_DENIED");
      const baseSha = String(body.base_sha ?? "");
      const requestId = body.request_id == null || String(body.request_id).trim() === "" ? null : String(body.request_id);
      const workerId = `github-actions:${String(claims.run_id ?? "unknown")}:${String(claims.run_attempt ?? "1")}`;
      const q = await db.rpc("claim_engineering_task", { p_worker_id: workerId, p_base_sha: baseSha, p_request_id: requestId });
      if (q.error) throw new Error(`CLAIM:${q.error.message}`);
      return out({ status: q.data ? "CLAIMED" : "NO_TASK", claim: q.data, identity: { run_id: claims.run_id, event_name: eventName } });
    }

    if (action === "event") {
      const runId = String(body.run_id ?? "");
      const phase = String(body.phase ?? "");
      const eventKind = String(body.event_kind ?? "");
      const commitSha = body.commit_sha == null ? null : String(body.commit_sha);
      const passed = body.passed == null ? null : Boolean(body.passed);
      const payload = cleanPayload(body.payload);
      if (!/^[0-9a-f-]{36}$/i.test(runId)) throw new Error("RUN_ID_INVALID");
      if (!eventKind || eventKind.length > 120) throw new Error("EVENT_KIND_INVALID");
      const q = await db.rpc("record_engineering_event", {
        p_run_id: runId,
        p_event_kind: eventKind,
        p_phase: phase,
        p_passed: passed,
        p_commit_sha: commitSha,
        p_payload: payload,
      });
      if (q.error) throw new Error(`EVENT:${q.error.message}`);
      return out({ status: "RECORDED", run_id: runId, phase, event_kind: eventKind });
    }

    return out({ status: "UNKNOWN_ACTION" }, 400);
  } catch (error) {
    console.error("brian-evolution-engineering-gateway", String(error));
    return out({ status: "FAILED_CLOSED", error: String(error) }, 401);
  }
});

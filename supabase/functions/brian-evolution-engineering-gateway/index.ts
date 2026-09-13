import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(SUPABASE_URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const ISSUER = "https://token.actions.githubusercontent.com";
const AUDIENCE = "brian-evolution-engineering-v1";
const REPOSITORY = "om3r305/Om3r-Pro-Proje";
const REPOSITORY_OWNER = "om3r305";
const BASE_REF = "refs/heads/brian-2026";
const ENGINEER_WORKFLOW = ".github/workflows/brian-engineer.yml";
const RELEASE_WORKFLOW = ".github/workflows/brian-engineer-release.yml";
const ALLOWED_WORKFLOWS = new Set([ENGINEER_WORKFLOW, RELEASE_WORKFLOW]);

const ENGINEER_EVENTS = new Map<string, string>([
  ["UNDERSTAND", "UNDERSTAND"],
  ["PLAN", "PLAN"],
  ["CODE", "CODE"],
  ["COMPILE", "COMPILE"],
  ["UNIT_REGRESSION", "TEST"],
  ["REPLAY", "REPLAY"],
  ["STRESS", "REPLAY"],
  ["INDEPENDENT_REVIEW", "REVIEW"],
  ["PR_CREATED", "PR"],
  ["VERCEL_PREVIEW", "PREVIEW"],
  ["PREVIEW_EQUIVALENT", "PREVIEW"],
  ["BLOCKED", "BLOCKED"],
]);
const RELEASE_EVENTS = new Map<string, string>([
  ["DEPLOYED", "DEPLOY"],
  ["MONITOR_HEALTHY", "MONITOR"],
  ["COMPLETE", "COMPLETE"],
  ["ROLLBACK", "ROLLBACK"],
  ["BLOCKED", "BLOCKED"],
]);

type Claims = Record<string, unknown>;
type JwkWithKid = JsonWebKey & { kid?: string };
let jwksCache: { expires: number; keys: JwkWithKid[] } | null = null;

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
async function jwks(): Promise<JwkWithKid[]> {
  if (jwksCache && jwksCache.expires > Date.now()) return jwksCache.keys;
  const response = await fetch(`${ISSUER}/.well-known/jwks`, { headers: { accept: "application/json" } });
  if (!response.ok) throw new Error(`OIDC_JWKS_HTTP_${response.status}`);
  const body = await response.json() as { keys?: JwkWithKid[] };
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
  if (claims.repository !== REPOSITORY || claims.repository_owner !== REPOSITORY_OWNER) throw new Error("OIDC_REPOSITORY_DENIED");
  const workflowRef = String(claims.job_workflow_ref ?? "");
  const workflowPath = workflowRef.startsWith(`${REPOSITORY}/`) ? workflowRef.slice(REPOSITORY.length + 1).split("@")[0] : "";
  if (!ALLOWED_WORKFLOWS.has(workflowPath)) throw new Error("OIDC_WORKFLOW_DENIED");
  return { ...claims, _workflow_path: workflowPath };
}
function cleanPayload(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const text = JSON.stringify(value);
  if (text.length > 60_000) throw new Error("PAYLOAD_TOO_LARGE");
  return value as Record<string, unknown>;
}
function requireEngineerControl(claims: Claims) {
  if (claims._workflow_path !== ENGINEER_WORKFLOW) throw new Error("ENGINEER_WORKFLOW_REQUIRED");
  if (!new Set(["schedule", "workflow_dispatch"]).has(String(claims.event_name ?? ""))) throw new Error("ENGINEER_EVENT_DENIED");
  if (claims.ref !== BASE_REF) throw new Error("ENGINEER_REF_DENIED");
}
function requireReleaseControl(claims: Claims) {
  if (claims._workflow_path !== RELEASE_WORKFLOW) throw new Error("RELEASE_WORKFLOW_REQUIRED");
  if (claims.event_name !== "pull_request_review") throw new Error("RELEASE_EVENT_DENIED");
  if (String(claims.actor ?? "") !== REPOSITORY_OWNER) throw new Error("RELEASE_ACTOR_MUST_BE_REPOSITORY_OWNER");
}
async function record(runId: string, eventKind: string, phase: string, passed: boolean, commitSha: string | null, payload: Record<string, unknown>) {
  const q = await db.rpc("record_engineering_event", {
    p_run_id: runId,
    p_event_kind: eventKind,
    p_phase: phase,
    p_passed: passed,
    p_commit_sha: commitSha,
    p_payload: payload,
  });
  if (q.error) throw new Error(`EVENT:${q.error.message}`);
}
async function runByBranch(branchName: string) {
  if (!branchName.startsWith("brian-engineer/")) throw new Error("ENGINEERING_BRANCH_INVALID");
  const q = await db.from("brian_evolution_engineering_runs").select("*").eq("branch_name", branchName).maybeSingle();
  if (q.error || !q.data) throw new Error(`ENGINEERING_RUN_NOT_FOUND:${q.error?.message ?? branchName}`);
  return q.data as Record<string, unknown>;
}
async function runById(runId: string) {
  if (!/^[0-9a-f-]{36}$/i.test(runId)) throw new Error("RUN_ID_INVALID");
  const q = await db.from("brian_evolution_engineering_runs").select("*").eq("run_id", runId).maybeSingle();
  if (q.error || !q.data) throw new Error(`ENGINEERING_RUN_NOT_FOUND:${q.error?.message ?? runId}`);
  return q.data as Record<string, unknown>;
}
function releaseReady(run: Record<string, unknown>) {
  return run.phase === "HUMAN_APPROVAL" && run.status === "WAITING" &&
    run.compile_passed === true && run.tests_passed === true && run.replay_passed === true && run.stress_passed === true &&
    run.review_passed === true && run.preview_passed === true && run.measurement_passed === true;
}
function provenance(claims: Claims) {
  return {
    github_run_id: String(claims.run_id ?? ""),
    github_run_attempt: String(claims.run_attempt ?? ""),
    github_actor: String(claims.actor ?? ""),
    source_workflow: String(claims._workflow_path ?? ""),
  };
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ status: "METHOD_NOT_ALLOWED" }, 405);
  try {
    const claims = await verifyGithubOidc(req);
    const body = cleanPayload(await req.json().catch(() => ({})));
    const action = String(body.action ?? "");

    if (action === "claim") {
      requireEngineerControl(claims);
      const baseSha = String(body.base_sha ?? "");
      if (!/^[0-9a-f]{40}$/i.test(baseSha)) throw new Error("BASE_SHA_INVALID");
      const oidcSha = String(claims.sha ?? "");
      if (oidcSha && oidcSha !== baseSha) throw new Error("BASE_SHA_DOES_NOT_MATCH_OIDC_SHA");
      const requestId = body.request_id == null || String(body.request_id).trim() === "" ? null : String(body.request_id);
      const workerId = `github-actions:${String(claims.run_id ?? "unknown")}:${String(claims.run_attempt ?? "1")}`;
      const q = await db.rpc("claim_engineering_task", { p_worker_id: workerId, p_base_sha: baseSha, p_request_id: requestId });
      if (q.error) throw new Error(`CLAIM:${q.error.message}`);
      return out({ status: q.data ? "CLAIMED" : "NO_TASK", claim: q.data, identity: provenance(claims) });
    }

    if (action === "event") {
      const runId = String(body.run_id ?? "");
      const eventKind = String(body.event_kind ?? "").toUpperCase();
      const requestedPhase = String(body.phase ?? "").toUpperCase();
      const commitSha = body.commit_sha == null || String(body.commit_sha).trim() === "" ? null : String(body.commit_sha);
      const payload = { ...cleanPayload(body.payload), ...provenance(claims) };
      if (!/^[0-9a-f-]{36}$/i.test(runId)) throw new Error("RUN_ID_INVALID");

      let expectedPhase: string | undefined;
      if (claims._workflow_path === ENGINEER_WORKFLOW) {
        requireEngineerControl(claims);
        expectedPhase = ENGINEER_EVENTS.get(eventKind);
      } else {
        requireReleaseControl(claims);
        expectedPhase = RELEASE_EVENTS.get(eventKind);
      }
      if (!expectedPhase) throw new Error("EVENT_KIND_NOT_ALLOWED_FOR_WORKFLOW");
      if (requestedPhase !== expectedPhase) throw new Error("EVENT_PHASE_MISMATCH");
      const passed = eventKind === "BLOCKED" ? false : true;
      await record(runId, eventKind, expectedPhase, passed, commitSha, payload);
      return out({ status: "RECORDED", run_id: runId, phase: expectedPhase, event_kind: eventKind });
    }

    if (action === "measure") {
      requireEngineerControl(claims);
      const runId = String(body.run_id ?? "");
      const commitSha = String(body.commit_sha ?? "");
      const measurement = { ...cleanPayload(body.measurement), ...provenance(claims), exact_commit_sha: commitSha };
      const run = await runById(runId);
      if (run.commit_sha !== commitSha) throw new Error("MEASUREMENT_COMMIT_MISMATCH");
      const q = await db.rpc("measure_engineering_run", { p_run_id: runId, p_commit_sha: commitSha, p_payload: measurement });
      if (q.error) throw new Error(`MEASURE:${q.error.message}`);
      return out({ status: "MEASURED", result: q.data });
    }

    if (action === "human_approve") {
      requireReleaseControl(claims);
      const branchName = String(body.branch_name ?? "");
      const headSha = String(body.head_sha ?? "");
      const run = await runByBranch(branchName);
      if (!releaseReady(run)) return out({ status: "BLOCKED", reason: "engineering evidence gates are incomplete", run_id: run.run_id }, 409);
      if (run.commit_sha !== headSha) return out({ status: "BLOCKED", reason: "approved head SHA does not match measured commit", run_id: run.run_id }, 409);
      const actor = String(claims.actor ?? "");
      const q = await db.rpc("approve_engineering_run", { p_run_id: run.run_id, p_commit_sha: headSha, p_actor: actor });
      if (q.error) throw new Error(`HUMAN_APPROVAL:${q.error.message}`);
      return out({ status: "APPROVED", result: q.data, actor });
    }

    if (action === "release_gate") {
      requireReleaseControl(claims);
      const run = await runByBranch(String(body.branch_name ?? ""));
      const ready = releaseReady(run) && run.human_approval_status === "APPROVED" && run.human_approved_by === REPOSITORY_OWNER;
      return out({ status: ready ? "READY" : "BLOCKED", ready, run });
    }

    return out({ status: "UNKNOWN_ACTION" }, 400);
  } catch (error) {
    console.error("brian-evolution-engineering-gateway", String(error));
    return out({ status: "FAILED_CLOSED", error: String(error) }, 401);
  }
});

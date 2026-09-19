import { createClient } from "npm:@supabase/supabase-js@2.116.0";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(SUPABASE_URL, SERVICE, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const VERSION = "brian.cloudflare-shadow-ingest.v2";
const CLOUDFLARE_KEY_SHA256 = "8d348396f3da9bbffde9bef6f6f8d802af542bdcb3354743f92d3ece260fea51";
const MAX_CAPTURES = 20;
const MAX_EVENTS = 100;

type Json = Record<string, unknown>;

function out(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function errText(error: unknown) {
  return error instanceof Error ? error.name + ": " + error.message : String(error);
}

async function sha256Hex(value: string) {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)),
  );
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function constantTimeEqual(left: string, right: string) {
  if (left.length !== right.length) return false;
  let diff = 0;
  for (let i = 0; i < left.length; i++) diff |= left.charCodeAt(i) ^ right.charCodeAt(i);
  return diff === 0;
}

async function requireCloudflareAuth(req: Request) {
  const supplied = (req.headers.get("x-brian-cloudflare-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED_CLOUDFLARE");
  const hash = await sha256Hex(supplied);
  if (!constantTimeEqual(hash, CLOUDFLARE_KEY_SHA256)) {
    throw new Error("UNAUTHORIZED_CLOUDFLARE");
  }
}

function cleanString(value: unknown, max = 1500) {
  return String(value ?? "").trim().slice(0, max);
}

function cleanIso(value: unknown) {
  const ms = Date.parse(String(value ?? ""));
  if (!Number.isFinite(ms)) throw new Error("INVALID_TIMESTAMP");
  return new Date(ms).toISOString();
}

function validateEventId(value: unknown) {
  const id = cleanString(value, 128);
  if (!id || !/^[A-Za-z0-9:_-]+$/.test(id)) throw new Error("INVALID_EVENT_ID");
  return id;
}

function sanitizeCapture(row: Json) {
  const captureId = cleanString(row.capture_id, 128);
  if (!captureId) throw new Error("INVALID_CAPTURE_ID");
  return {
    capture_id: captureId,
    provider: cleanString(row.provider, 200) || "cloudflare_eye",
    record_type: cleanString(row.record_type, 200) || "cloudflare_source_arch_v2_feed",
    observed_at: cleanIso(row.observed_at),
    captured_at: cleanIso(row.captured_at ?? row.observed_at),
    provenance_uri: cleanString(row.provenance_uri, 3000),
    payload_hash: cleanString(row.payload_hash, 128),
    payload: {
      ...((row.payload && typeof row.payload === "object") ? row.payload as Json : {}),
      cloudflare_shadow: true,
      decision_evidence_locked: true,
      shadow_only: true,
      live_execution: false,
    },
  };
}

function sanitizeEvent(row: Json) {
  const eventId = validateEventId(row.event_id);
  const rawCaptureId = cleanString(row.raw_capture_id, 128);
  if (!rawCaptureId) throw new Error("RAW_CAPTURE_ID_REQUIRED");
  const metadata =
    row.metadata && typeof row.metadata === "object" ? row.metadata as Json : {};

  return {
    event_id: eventId,
    asset: cleanString(row.asset, 120) || "GLOBAL",
    event_kind: cleanString(row.event_kind, 160) || "OFFICIAL_SOURCE_ITEM",
    source_kind: cleanString(row.source_kind, 160),
    source_id: cleanString(row.source_id, 300),
    published_at: row.published_at ? cleanIso(row.published_at) : null,
    first_observed_at: cleanIso(row.first_observed_at),
    captured_at: cleanIso(row.captured_at ?? row.first_observed_at),
    claim: cleanString(row.claim, 1500),
    direction: 0,
    magnitude: Math.max(0, Math.min(1, Number(row.magnitude ?? 1))),
    trust_class: cleanString(row.trust_class, 120) || "INDEPENDENT_PROFESSIONAL",
    entity_confidence: Math.max(0, Math.min(1, Number(row.entity_confidence ?? 0))),
    content_fingerprint: cleanString(row.content_fingerprint, 128),
    corroboration_key: row.corroboration_key ? cleanString(row.corroboration_key, 128) : null,
    provenance_uri: cleanString(row.provenance_uri, 3000),
    pit_verified: Boolean(row.pit_verified),
    raw_capture_id: rawCaptureId,
    metadata: {
      ...metadata,
      cloudflare_shadow: true,
      direction_not_inferred: true,
      external_content_used_as_instruction: false,
      eligible_for_decision_evidence: false,
      decision_evidence_locked: true,
      shadow_only: true,
      live_execution: false,
    },
  };
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);

  try {
    await requireCloudflareAuth(req);
  } catch (error) {
    return out({
      status: "UNAUTHORIZED",
      error: errText(error),
      shadow_only: true,
      live_execution: false,
    }, 401);
  }

  try {
    const body = await req.json().catch(() => ({})) as Json;
    if (body.shadow_only !== true || body.live_execution !== false) {
      return out({ status: "REJECTED_SAFETY_CONTRACT" }, 400);
    }

    const recheckRows = Array.isArray(body.rechecks) ? body.rechecks : [];
    if (recheckRows.length) {
      return out({
        status: "REJECTED_RECHECK_DISABLED",
        reason: "ALPHA event recheck is disabled during Cloudflare shadow phase",
        shadow_only: true,
        live_execution: false,
      }, 409);
    }

    const captureRows = Array.isArray(body.captures)
      ? body.captures.slice(0, MAX_CAPTURES)
      : [];
    const eventRows = Array.isArray(body.events)
      ? body.events.slice(0, MAX_EVENTS)
      : [];

    const captures = captureRows.map((row) => sanitizeCapture(row as Json));
    const events = eventRows.map((row) => sanitizeEvent(row as Json));

    if (captures.length) {
      const captureWrite = await db
        .from("brian_raw_captures")
        .upsert(captures, {
          onConflict: "capture_id",
          ignoreDuplicates: true,
        });
      if (captureWrite.error) throw captureWrite.error;
    }

    if (events.length) {
      const eventWrite = await db
        .from("brian_intel_events")
        .upsert(events, {
          onConflict: "event_id",
          ignoreDuplicates: true,
        });
      if (eventWrite.error) throw eventWrite.error;
    }

    return out({
      status: "CAPTURED_SHADOW",
      version: VERSION,
      captures_received: captures.length,
      events_received: events.length,
      alpha_rechecks_queued: 0,
      alpha_recheck_enabled: false,
      auth_boundary: "cloudflare_dedicated_key",
      idempotent_event_key: "event_id",
      decision_evidence_locked: true,
      shadow_only: true,
      live_execution: false,
    });
  } catch (error) {
    return out({
      status: "FAILED_CLOSED",
      version: VERSION,
      error: errText(error).slice(0, 1200),
      shadow_only: true,
      live_execution: false,
    }, 500);
  }
});

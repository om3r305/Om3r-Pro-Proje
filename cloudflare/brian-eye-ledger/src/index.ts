import { DurableObject } from "cloudflare:workers";
import { XMLParser } from "fast-xml-parser";

const VERSION = "brian.cf-eye-ledger.v1";
const MAX_SOURCES = 20;
const MAX_ITEMS_PER_FEED = 80;
const MAX_ITEM_AGE_MS = 48 * 60 * 60 * 1000;

type Json = Record<string, unknown>;

type SourceEndpoint = {
  endpoint_id: string;
  source_id: string;
  organization?: string;
  canonical_domain: string;
  endpoint_url: string;
  endpoint_kind: "RSS" | "ATOM" | "STATUSPAGE_ATOM";
  tier: string;
  category: string;
  region?: string;
  priority?: number;
};

type FeedItem = {
  title: string;
  link: string;
  guid: string;
  publishedAt: string | null;
  categories: string[];
};

type CaptureEnvelope = {
  capture_id: string;
  provider: string;
  record_type: string;
  observed_at: string;
  captured_at: string;
  provenance_uri: string;
  payload_hash: string;
  payload: Json;
};

type EventEnvelope = {
  event_id: string;
  asset: string;
  event_kind: string;
  source_kind: string;
  source_id: string;
  published_at: string | null;
  first_observed_at: string;
  captured_at: string;
  claim: string;
  direction: 0;
  magnitude: number;
  trust_class: string;
  entity_confidence: number;
  content_fingerprint: string;
  corroboration_key: string | null;
  provenance_uri: string;
  pit_verified: true;
  raw_capture_id: string;
  metadata: Json;
};

type LedgerInput = {
  event_id: string;
  source_id: string;
  published_at: string | null;
  payload_hash: string;
  event: EventEnvelope;
  capture: CaptureEnvelope;
};

type LedgerRow = {
  event_id: string;
  first_seen_at: string;
  last_seen_at: string;
  seen_count: number;
  forwarded_at: string | null;
  event_json: string;
  capture_json: string;
};

type LedgerResult = {
  event_id: string;
  first_seen: boolean;
  first_seen_at: string;
  seen_count: number;
  needs_forward: boolean;
  event: EventEnvelope;
  capture: CaptureEnvelope;
};

export interface Env {
  FIRST_SEEN_LEDGER: DurableObjectNamespace<FirstSeenLedger>;
  SHADOW_ONLY: string;
  LIVE_EXECUTION: string;
  ENABLE_SCHEDULED_EYE: string;
  R2_ENABLED: string;
  SUPABASE_INGEST_URL: string;
  BRIAN_CLOUDFLARE_KEY?: string;
  ALPHA_RECHECK_ENABLED: string;
  SOURCE_MANIFEST_JSON?: string;
  RAW_BUCKET?: R2Bucket;
}

export class FirstSeenLedger extends DurableObject<Env> {
  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.ctx.storage.sql.exec(
      "CREATE TABLE IF NOT EXISTS first_seen_events (" +
        "event_id TEXT PRIMARY KEY," +
        "source_id TEXT NOT NULL," +
        "published_at TEXT," +
        "payload_hash TEXT NOT NULL," +
        "first_seen_at TEXT NOT NULL," +
        "last_seen_at TEXT NOT NULL," +
        "seen_count INTEGER NOT NULL DEFAULT 1," +
        "forwarded_at TEXT," +
        "last_forward_error TEXT," +
        "event_json TEXT NOT NULL," +
        "capture_json TEXT NOT NULL" +
      ");" +
      "CREATE INDEX IF NOT EXISTS idx_first_seen_pending ON first_seen_events(forwarded_at, first_seen_at);"
    );
  }

  async record(input: LedgerInput): Promise<LedgerResult> {
    const now = new Date().toISOString();
    const prior = this.ctx.storage.sql
      .exec<LedgerRow>(
        "SELECT event_id, first_seen_at, last_seen_at, seen_count, forwarded_at, event_json, capture_json " +
        "FROM first_seen_events WHERE event_id = ? LIMIT 1",
        input.event_id
      )
      .toArray()[0];

    if (!prior) {
      const event = { ...input.event, first_observed_at: now };
      this.ctx.storage.sql.exec(
        "INSERT INTO first_seen_events " +
        "(event_id, source_id, published_at, payload_hash, first_seen_at, last_seen_at, seen_count, forwarded_at, last_forward_error, event_json, capture_json) " +
        "VALUES (?, ?, ?, ?, ?, ?, 1, NULL, NULL, ?, ?)",
        input.event_id,
        input.source_id,
        input.published_at,
        input.payload_hash,
        now,
        now,
        JSON.stringify(event),
        JSON.stringify(input.capture)
      );
      return {
        event_id: input.event_id,
        first_seen: true,
        first_seen_at: now,
        seen_count: 1,
        needs_forward: true,
        event,
        capture: input.capture
      };
    }

    const updated = this.ctx.storage.sql
      .exec<{ seen_count: number; first_seen_at: string; forwarded_at: string | null }>(
        "UPDATE first_seen_events SET last_seen_at = ?, seen_count = seen_count + 1, payload_hash = ? " +
        "WHERE event_id = ? RETURNING seen_count, first_seen_at, forwarded_at",
        now,
        input.payload_hash,
        input.event_id
      )
      .one();

    return {
      event_id: input.event_id,
      first_seen: false,
      first_seen_at: updated.first_seen_at,
      seen_count: Number(updated.seen_count),
      needs_forward: !updated.forwarded_at,
      event: JSON.parse(prior.event_json) as EventEnvelope,
      capture: JSON.parse(prior.capture_json) as CaptureEnvelope
    };
  }

  async markForwarded(eventId: string): Promise<void> {
    this.ctx.storage.sql.exec(
      "UPDATE first_seen_events SET forwarded_at = ?, last_forward_error = NULL WHERE event_id = ?",
      new Date().toISOString(),
      eventId
    );
  }

  async markForwardError(eventId: string, error: string): Promise<void> {
    this.ctx.storage.sql.exec(
      "UPDATE first_seen_events SET last_forward_error = ? WHERE event_id = ?",
      error.slice(0, 1000),
      eventId
    );
  }

  async lookup(eventId: string): Promise<Json | null> {
    const row = this.ctx.storage.sql
      .exec<Json>("SELECT * FROM first_seen_events WHERE event_id = ? LIMIT 1", eventId)
      .toArray()[0];
    return row ?? null;
  }
}

function out(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store"
    }
  });
}

function errText(error: unknown) {
  return error instanceof Error ? error.name + ": " + error.message : String(error);
}

function arr<T>(value: T | T[] | null | undefined): T[] {
  return value == null ? [] : Array.isArray(value) ? value : [value];
}

function txt(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (value && typeof value === "object") {
    const row = value as Json;
    return String(row["#text"] ?? row["@_term"] ?? row["@_label"] ?? "").trim();
  }
  return value == null ? "" : String(value).trim();
}

function link(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (value && typeof value === "object") {
    const row = value as Json;
    return String(row["@_href"] ?? row.href ?? row["#text"] ?? "").trim();
  }
  return "";
}

function iso(value: unknown): string | null {
  const raw = txt(value);
  if (!raw) return null;
  const ms = Date.parse(raw);
  return Number.isFinite(ms) ? new Date(ms).toISOString() : null;
}

function hostMatches(host: string, domain: string) {
  const h = host.toLowerCase().replace(/^www\./, "");
  const d = domain.toLowerCase().replace(/^www\./, "");
  return h === d || h.endsWith("." + d);
}

function normalizeClaim(value: string) {
  return value
    .toLowerCase()
    .normalize("NFKD")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .slice(0, 500);
}

async function sha(value: string | Uint8Array) {
  const bytes = typeof value === "string" ? new TextEncoder().encode(value) : value;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function parseFeed(xml: string): FeedItem[] {
  const parser = new XMLParser({
    ignoreAttributes: false,
    trimValues: true,
    processEntities: true
  });
  const doc = parser.parse(xml) as Json;
  const rss = (doc.rss as Json | undefined)?.channel as Json | undefined;
  const atom = doc.feed as Json | undefined;
  const raw = arr((rss?.item ?? atom?.entry) as Json | Json[] | undefined);
  const items: FeedItem[] = [];

  for (const row of raw.slice(0, MAX_ITEMS_PER_FEED)) {
    if (!row || typeof row !== "object") continue;
    const title = txt(row.title).replace(/\s+/g, " ").trim();
    const links = arr(row.link as unknown);
    let href = "";
    for (const candidate of links) {
      href = link(candidate);
      if (href) break;
    }
    const guid = txt(row.guid ?? row.id) || href || title;
    const publishedAt = iso(row.pubDate ?? row.published ?? row.updated);
    const categories = arr(row.category as unknown)
      .map((x) => txt(x))
      .filter(Boolean);
    if (!title || !guid) continue;
    items.push({
      title: title.slice(0, 1500),
      link: href,
      guid,
      publishedAt,
      categories
    });
  }
  return items;
}

function freshEnough(item: FeedItem, nowMs: number) {
  if (!item.publishedAt) return true;
  const ms = Date.parse(item.publishedAt);
  return Number.isFinite(ms) && ms <= nowMs + 5 * 60_000 && nowMs - ms <= MAX_ITEM_AGE_MS;
}

function validateSource(endpoint: SourceEndpoint) {
  if (!endpoint.endpoint_id || !endpoint.source_id || !endpoint.canonical_domain) {
    throw new Error("INVALID_SOURCE_MANIFEST_ROW");
  }
  const url = new URL(endpoint.endpoint_url);
  if (url.protocol !== "https:") throw new Error("SOURCE_URL_MUST_BE_HTTPS");
  if (!hostMatches(url.hostname, endpoint.canonical_domain)) {
    throw new Error("SOURCE_ORIGIN_MISMATCH:" + url.hostname);
  }
}

function loadManifest(env: Env): SourceEndpoint[] {
  const raw = env.SOURCE_MANIFEST_JSON?.trim();
  if (!raw) return [];
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed)) throw new Error("SOURCE_MANIFEST_JSON_MUST_BE_ARRAY");
  const rows = parsed.slice(0, MAX_SOURCES) as SourceEndpoint[];
  for (const row of rows) validateSource(row);
  return rows;
}

function sameSecret(a: string, b: string) {
  if (!a || !b || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

function requireWorkerAuth(req: Request, env: Env) {
  const expected = env.BRIAN_CLOUDFLARE_KEY?.trim() ?? "";
  const supplied = req.headers.get("x-brian-cloudflare-key")?.trim() ?? "";
  if (!sameSecret(expected, supplied)) throw new Error("UNAUTHORIZED");
}

async function fetchFeed(endpoint: SourceEndpoint) {
  const timeoutMs = endpoint.endpoint_kind === "STATUSPAGE_ATOM" ? 20000 : 9000;
  const response = await fetch(endpoint.endpoint_url, {
    redirect: "follow",
    headers: {
      accept: "application/rss+xml,application/atom+xml,application/xml,text/xml;q=0.9,*/*;q=0.1"
    },
    signal: AbortSignal.timeout(timeoutMs)
  });
  if (!response.ok) throw new Error("HTTP_" + response.status);
  const finalHost = new URL(response.url || endpoint.endpoint_url).hostname;
  if (!hostMatches(finalHost, endpoint.canonical_domain)) {
    throw new Error("ORIGIN_MISMATCH:" + finalHost);
  }
  const xml = await response.text();
  if (!xml.trim()) throw new Error("EMPTY_FEED");
  return xml;
}

async function persistRaw(
  env: Env,
  endpoint: SourceEndpoint,
  observedAt: string,
  payloadHash: string,
  xml: string
) {
  const path =
    "source-arch-v2/" +
    endpoint.endpoint_id +
    "/" +
    observedAt.slice(0, 10) +
    "/" +
    payloadHash +
    ".xml";

  if (env.R2_ENABLED !== "true" || !env.RAW_BUCKET) {
    return { stored: false, path, reason: "R2_DISABLED" };
  }

  await env.RAW_BUCKET.put(path, xml, {
    httpMetadata: { contentType: "application/xml; charset=utf-8" },
    customMetadata: {
      endpoint_id: endpoint.endpoint_id,
      source_id: endpoint.source_id,
      observed_at: observedAt,
      payload_hash: payloadHash,
      shadow_only: "true"
    }
  });

  return { stored: true, path, reason: null };
}

function ledgerStub(env: Env, eventId: string) {
  const shard = eventId.slice(0, 2) || "00";
  const id = env.FIRST_SEEN_LEDGER.idFromName("events:" + shard);
  return { shard, stub: env.FIRST_SEEN_LEDGER.get(id) };
}

async function forwardBatch(env: Env, pending: LedgerResult[]) {
  if (!pending.length) return { forwarded: 0, status: "NO_NEW_EVENTS" };
  if (!env.SUPABASE_INGEST_URL || !env.BRIAN_CLOUDFLARE_KEY) {
    throw new Error("SUPABASE_INGEST_NOT_CONFIGURED");
  }

  const captures = new Map<string, CaptureEnvelope>();
  const events = new Map<string, EventEnvelope>();
  for (const row of pending) {
    captures.set(row.capture.capture_id, row.capture);
    events.set(row.event.event_id, row.event);
  }

  const response = await fetch(env.SUPABASE_INGEST_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-brian-cloudflare-key": env.BRIAN_CLOUDFLARE_KEY
    },
    body: JSON.stringify({
      version: VERSION,
      captures: [...captures.values()],
      events: [...events.values()],
      rechecks: [],
      shadow_only: true,
      live_execution: false
    }),
    signal: AbortSignal.timeout(12000)
  });

  const body = await response.json().catch(() => ({})) as Json;
  if (!response.ok) {
    const message =
      "SUPABASE_INGEST_HTTP_" +
      response.status +
      ":" +
      JSON.stringify(body).slice(0, 700);

    for (const row of pending) {
      const target = ledgerStub(env, row.event_id);
      await target.stub.markForwardError(row.event_id, message);
    }
    throw new Error(message);
  }

  for (const row of pending) {
    const target = ledgerStub(env, row.event_id);
    await target.stub.markForwarded(row.event_id);
  }

  return {
    forwarded: pending.length,
    status: String(body.status ?? "CAPTURED_SHADOW")
  };
}

async function pollSource(env: Env, endpoint: SourceEndpoint) {
  const observedAt = new Date().toISOString();
  const nowMs = Date.parse(observedAt);
  const xml = await fetchFeed(endpoint);
  const raw = new TextEncoder().encode(xml);
  const payloadHash = await sha(raw);
  const captureId = await sha(endpoint.endpoint_id + "|" + observedAt + "|" + payloadHash);
  const rawPersist = await persistRaw(env, endpoint, observedAt, payloadHash, xml);

  const capture: CaptureEnvelope = {
    capture_id: captureId,
    provider: "cloudflare_eye:" + endpoint.endpoint_id,
    record_type: "cloudflare_source_arch_v2_feed",
    observed_at: observedAt,
    captured_at: new Date().toISOString(),
    provenance_uri: endpoint.endpoint_url,
    payload_hash: payloadHash,
    payload: {
      runtime: VERSION,
      endpoint_id: endpoint.endpoint_id,
      source_id: endpoint.source_id,
      organization: endpoint.organization ?? null,
      tier: endpoint.tier,
      category: endpoint.category,
      region: endpoint.region ?? null,
      r2_enabled: env.R2_ENABLED === "true",
      r2_object_committed: rawPersist.stored,
      r2_path: rawPersist.path,
      decision_evidence_locked: true,
      external_content_used_as_instruction: false,
      shadow_only: true,
      live_execution: false
    }
  };

  const parsed = parseFeed(xml);
  const selected = parsed.filter((item) => freshEnough(item, nowMs));
  const pending: LedgerResult[] = [];

  for (const item of selected) {
    const eventId = await sha(
      "source-arch-v2|" + endpoint.endpoint_id + "|" + item.guid
    );
    const fingerprint = await sha(
      endpoint.endpoint_id + "|" + item.title + "|" + item.link
    );
    const normalized = normalizeClaim(item.title);
    const corroboration = normalized
      ? await sha(endpoint.category + "|" + normalized)
      : null;
    const trustClass =
      endpoint.tier === "T1_OFFICIAL_PRIMARY" || endpoint.tier === "T0_RAW_TELEMETRY"
        ? "OFFICIAL_PRIMARY"
        : "INDEPENDENT_PROFESSIONAL";

    const event: EventEnvelope = {
      event_id: eventId,
      asset: "GLOBAL",
      event_kind:
        endpoint.category === "CRYPTO_EXCHANGE_STATUS"
          ? "OFFICIAL_EXCHANGE_STATUS"
          : "OFFICIAL_SOURCE_ITEM",
      source_kind: endpoint.tier,
      source_id: endpoint.source_id,
      published_at: item.publishedAt,
      first_observed_at: observedAt,
      captured_at: new Date().toISOString(),
      claim: item.title,
      direction: 0,
      magnitude: Math.max(0.1, Math.min(1, Number(endpoint.priority ?? 50) / 100)),
      trust_class: trustClass,
      entity_confidence: endpoint.tier === "T1_OFFICIAL_PRIMARY" ? 1 : 0.92,
      content_fingerprint: fingerprint,
      corroboration_key: corroboration,
      provenance_uri: item.link || endpoint.endpoint_url,
      pit_verified: true,
      raw_capture_id: captureId,
      metadata: {
        source_arch_version: "V2",
        runtime: VERSION,
        endpoint_id: endpoint.endpoint_id,
        organization: endpoint.organization ?? null,
        tier: endpoint.tier,
        category: endpoint.category,
        region: endpoint.region ?? null,
        categories: item.categories,
        direction_not_inferred: true,
        external_content_used_as_instruction: false,
        eligible_for_decision_evidence: false,
        decision_evidence_locked: true,
        cloudflare_shadow: true,
        shadow_only: true,
        live_execution: false
      }
    };

    const target = ledgerStub(env, eventId);
    const ledger = await target.stub.record({
      event_id: eventId,
      source_id: endpoint.source_id,
      published_at: item.publishedAt,
      payload_hash: payloadHash,
      event: {
        ...event,
        metadata: {
          ...event.metadata,
          cf_ledger_shard: target.shard
        }
      },
      capture
    });

    if (ledger.needs_forward) pending.push(ledger);
  }

  const forwarded = await forwardBatch(env, pending);
  return {
    endpoint_id: endpoint.endpoint_id,
    parsed: parsed.length,
    selected: selected.length,
    pending: pending.length,
    forwarded: forwarded.forwarded,
    forward_status: forwarded.status,
    raw_r2_committed: rawPersist.stored
  };
}

async function runSources(env: Env) {
  if (!env.SUPABASE_INGEST_URL || !env.BRIAN_CLOUDFLARE_KEY) {
    return {
      status: "NOT_CONFIGURED",
      reason: "SUPABASE_INGEST_URL_OR_BRIAN_CLOUDFLARE_KEY_MISSING",
      shadow_only: true,
      live_execution: false
    };
  }

  const sources = loadManifest(env);
  if (!sources.length) {
    return {
      status: "NO_SOURCES_CONFIGURED",
      source_count: 0,
      shadow_only: true,
      live_execution: false
    };
  }

  const results: Json[] = [];
  for (const endpoint of sources) {
    try {
      results.push(await pollSource(env, endpoint));
    } catch (error) {
      results.push({
        endpoint_id: endpoint.endpoint_id,
        error: errText(error).slice(0, 1200)
      });
    }
  }

  const failures = results.filter((row) => row.error);
  return {
    status: failures.length ? "DEGRADED" : "SUCCESS",
    version: VERSION,
    source_count: sources.length,
    failures: failures.length,
    results,
    shadow_only: true,
    live_execution: false
  };
}

async function routeAlphaRecheck(req: Request, env: Env) {
  requireWorkerAuth(req, env);
  if (env.ALPHA_RECHECK_ENABLED !== "true") {
    return out({ status: "DISABLED_SHADOW_PHASE", shadow_only: true, live_execution: false }, 503);
  }
  if (!env.SUPABASE_INGEST_URL || !env.BRIAN_CLOUDFLARE_KEY) {
    return out({ status: "NOT_CONFIGURED" }, 503);
  }

  const body = await req.json().catch(() => ({})) as Json;
  const eventId = String(body.event_id ?? "").trim();
  const assetId = String(body.asset_id ?? "").trim();
  const alertId = String(body.alert_id ?? "").trim();

  if (!eventId || !/^crypto:[A-Z0-9]{2,20}USDT$/.test(assetId)) {
    return out({ error: "event_id and crypto:*USDT asset_id required" }, 400);
  }

  const response = await fetch(env.SUPABASE_INGEST_URL, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-brian-cloudflare-key": env.BRIAN_CLOUDFLARE_KEY
    },
    body: JSON.stringify({
      version: VERSION,
      captures: [],
      events: [],
      rechecks: [{
        event_id: eventId,
        asset_id: assetId,
        alert_id: alertId || null
      }],
      shadow_only: true,
      live_execution: false
    }),
    signal: AbortSignal.timeout(12000)
  });

  const payload = await response.json().catch(() => ({}));
  return out(payload, response.status);
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    const url = new URL(req.url);

    if (req.method === "GET" && url.pathname === "/health") {
      let sourceCount = 0;
      try {
        sourceCount = loadManifest(env).length;
      } catch {
        sourceCount = -1;
      }
      return out({
        status: "OK",
        version: VERSION,
        source_count: sourceCount,
        scheduled_eye_enabled: env.ENABLE_SCHEDULED_EYE === "true",
        r2_enabled: env.R2_ENABLED === "true" && Boolean(env.RAW_BUCKET),
        alpha_recheck_enabled: env.ALPHA_RECHECK_ENABLED === "true",
        shadow_only: true,
        live_execution: false
      });
    }

    if (req.method === "GET" && url.pathname === "/one-shot-shadow-test") {
      let sources: SourceEndpoint[] = [];
      try {
        sources = loadManifest(env);
      } catch (error) {
        return out({
          status: "SAFE_TEST_BLOCKED",
          reason: "INVALID_SOURCE_MANIFEST",
          error: errText(error).slice(0, 300),
          shadow_only: true,
          live_execution: false
        }, 409);
      }

      const safe =
        env.SHADOW_ONLY === "true" &&
        env.LIVE_EXECUTION !== "true" &&
        env.ENABLE_SCHEDULED_EYE !== "true" &&
        env.R2_ENABLED !== "true" &&
        env.ALPHA_RECHECK_ENABLED !== "true" &&
        sources.length === 1;

      if (!safe) {
        return out({
          status: "SAFE_TEST_BLOCKED",
          source_count: sources.length,
          scheduled_eye_enabled: env.ENABLE_SCHEDULED_EYE === "true",
          r2_enabled: env.R2_ENABLED === "true",
          alpha_recheck_enabled: env.ALPHA_RECHECK_ENABLED === "true",
          shadow_only: env.SHADOW_ONLY === "true",
          live_execution: env.LIVE_EXECUTION === "true"
        }, 409);
      }

      const result = await runSources(env);
      return out({
        status: "ONE_SHOT_COMPLETE",
        result,
        safety: {
          source_count: 1,
          scheduled_eye_enabled: false,
          r2_enabled: false,
          alpha_recheck_enabled: false,
          shadow_only: true,
          live_execution: false
        }
      });
    }

    if (req.method === "POST" && url.pathname === "/run") {
      try {
        requireWorkerAuth(req, env);
      } catch {
        return out({ status: "UNAUTHORIZED" }, 401);
      }
      return out(await runSources(env));
    }

    if (req.method === "POST" && url.pathname === "/route/alpha-recheck") {
      try {
        return await routeAlphaRecheck(req, env);
      } catch {
        return out({ status: "UNAUTHORIZED" }, 401);
      }
    }

    if (req.method === "GET" && url.pathname === "/ledger") {
      try {
        requireWorkerAuth(req, env);
      } catch {
        return out({ status: "UNAUTHORIZED" }, 401);
      }
      const eventId = url.searchParams.get("event_id")?.trim() ?? "";
      if (!eventId) return out({ error: "event_id required" }, 400);
      const target = ledgerStub(env, eventId);
      return out({
        shard: target.shard,
        row: await target.stub.lookup(eventId)
      });
    }

    return out({ error: "NOT_FOUND" }, 404);
  },

  async scheduled(
    _controller: ScheduledController,
    env: Env,
    ctx: ExecutionContext
  ) {
    if (env.ENABLE_SCHEDULED_EYE !== "true") return;
    ctx.waitUntil(runSources(env));
  }
} satisfies ExportedHandler<Env>;

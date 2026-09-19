import { DurableObject } from "cloudflare:workers";
import { XMLParser } from "fast-xml-parser";

const VERSION = "brian.cf-eye-ledger.v2.6";
const MAX_SOURCES = 20;
const MAX_ITEMS_PER_FEED = 80;
const MAX_ITEM_AGE_MS = 48 * 60 * 60 * 1000;
const SENTINEL_CRON_KEY_SHA256 = "814a5df4f8d6e3b15f1b9ac19a4ea823ad69eedc52caa6ad7573fde7aa96eaab";

type Json = Record<string, unknown>;

type SourceEndpoint = {
  endpoint_id: string;
  source_id: string;
  organization?: string;
  canonical_domain: string;
  endpoint_url: string;
  endpoint_kind: "RSS" | "ATOM" | "STATUSPAGE_ATOM" | "STATUSPAGE_JSON" | "HTML_LINKS";
  tier: string;
  category: string;
  region?: string;
  priority?: number;
  html_path_prefix?: string;
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

type AlphaRecheckQueueInput = {
  request_id: string;
  event_id: string;
  asset_id: string;
  alert_id: string | null;
};

export interface Env {
  FIRST_SEEN_LEDGER: DurableObjectNamespace<FirstSeenLedger>;
  ALPHA_RECHECK_QUEUE: DurableObjectNamespace<AlphaRecheckQueue>;
  SHADOW_ONLY: string;
  LIVE_EXECUTION: string;
  ENABLE_SCHEDULED_EYE: string;
  R2_ENABLED: string;
  SUPABASE_INGEST_URL: string;
  REALTIME_INGEST_URL?: string;
  ALPHA_RECHECK_URL?: string;
  REALTIME_ALPHA_RECHECK_URL?: string;
  REALTIME_ENGINES_ENABLED?: string;
  REALTIME_UNIVERSE_URL?: string;
  REALTIME_SENSOR_URL?: string;
  REALTIME_INTRABAR_URL?: string;
  REALTIME_ALPHA_COMPILER_URL?: string;
  REALTIME_CATALYST_REACTION_URL?: string;
  BRIAN_CLOUDFLARE_KEY?: string;
  ALPHA_RECHECK_ENABLED: string;
  CORE_RECOVERY_ENABLED?: string;
  CORE_LAUNCHER_URL?: string;
  DURABLE_OBJECTS_ENABLED?: string;
  R2_OUTBOX_ENABLED?: string;
  SOURCE_MANIFEST_JSON?: string;
  RAW_BUCKET?: R2Bucket;
}

export class FirstSeenLedger extends DurableObject<Env> {
  private envRef: Env;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.envRef = env;
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

  async scheduleFlush(delayMs = 250): Promise<{ scheduled: true; at: number }> {
    const at = Date.now() + Math.max(0, delayMs);
    const current = await this.ctx.storage.getAlarm();
    if (current == null || at < current) {
      await this.ctx.storage.setAlarm(at);
    }
    return { scheduled: true, at };
  }

  async alarm(): Promise<void> {
    if (this.envRef.DURABLE_OBJECTS_ENABLED !== "true") return;
    const rows = this.ctx.storage.sql
      .exec<LedgerRow>(
        "SELECT event_id, first_seen_at, last_seen_at, seen_count, forwarded_at, event_json, capture_json " +
        "FROM first_seen_events WHERE forwarded_at IS NULL ORDER BY first_seen_at LIMIT 50"
      )
      .toArray();

    if (!rows.length) return;

    const retryLater = async (message: string, delayMs = 60000) => {
      for (const row of rows) {
        this.ctx.storage.sql.exec(
          "UPDATE first_seen_events SET last_forward_error = ? WHERE event_id = ?",
          message.slice(0, 1000),
          row.event_id
        );
      }
      await this.ctx.storage.setAlarm(Date.now() + delayMs);
    };

    if (!this.envRef.SUPABASE_INGEST_URL || !this.envRef.BRIAN_CLOUDFLARE_KEY) {
      await retryLater("SUPABASE_INGEST_NOT_CONFIGURED", 300000);
      return;
    }

    const captures = new Map<string, CaptureEnvelope>();
    const events = new Map<string, EventEnvelope>();
    for (const row of rows) {
      const capture = JSON.parse(row.capture_json) as CaptureEnvelope;
      const event = JSON.parse(row.event_json) as EventEnvelope;
      captures.set(capture.capture_id, capture);
      events.set(event.event_id, event);
    }

    try {
      const response = await fetch(this.envRef.SUPABASE_INGEST_URL, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-brian-cloudflare-key": this.envRef.BRIAN_CLOUDFLARE_KEY
        },
        body: JSON.stringify({
          version: VERSION,
          captures: [...captures.values()],
          events: [...events.values()],
          rechecks: [],
          shadow_only: true,
          live_execution: false
        }),
        signal: AbortSignal.timeout(25000)
      });

      const body = await response.json().catch(() => ({})) as Json;
      if (!response.ok) {
        await retryLater(
          "SUPABASE_INGEST_HTTP_" +
          response.status +
          ":" +
          JSON.stringify(body).slice(0, 700)
        );
        return;
      }

      const now = new Date().toISOString();
      for (const row of rows) {
        this.ctx.storage.sql.exec(
          "UPDATE first_seen_events SET forwarded_at = ?, last_forward_error = NULL WHERE event_id = ?",
          now,
          row.event_id
        );
      }

      const remaining = this.ctx.storage.sql
        .exec<{ count: number }>(
          "SELECT COUNT(*) AS count FROM first_seen_events WHERE forwarded_at IS NULL"
        )
        .one();

      if (Number(remaining.count) > 0) {
        await this.ctx.storage.setAlarm(Date.now() + 250);
      }
    } catch (error) {
      await retryLater("SUPABASE_FORWARD_EXCEPTION:" + errText(error));
    }
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
    await this.scheduleFlush(60000);
  }

  async lookup(eventId: string): Promise<Json | null> {
    const row = this.ctx.storage.sql
      .exec<Json>("SELECT * FROM first_seen_events WHERE event_id = ? LIMIT 1", eventId)
      .toArray()[0];
    return row ?? null;
  }
}

export class AlphaRecheckQueue extends DurableObject<Env> {
  private envRef: Env;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    this.envRef = env;
    this.ctx.storage.sql.exec(
      "CREATE TABLE IF NOT EXISTS alpha_rechecks (" +
        "request_id TEXT PRIMARY KEY," +
        "event_id TEXT NOT NULL," +
        "asset_id TEXT NOT NULL," +
        "alert_id TEXT," +
        "queued_at TEXT NOT NULL," +
        "updated_at TEXT NOT NULL," +
        "attempts INTEGER NOT NULL DEFAULT 0," +
        "delivered_at TEXT," +
        "last_error TEXT," +
        "last_response TEXT" +
      ");" +
      "CREATE INDEX IF NOT EXISTS idx_alpha_rechecks_pending ON alpha_rechecks(delivered_at, queued_at);"
    );
  }

  async enqueue(input: AlphaRecheckQueueInput): Promise<Json> {
    const now = new Date().toISOString();
    const prior = this.ctx.storage.sql
      .exec<Json>(
        "SELECT * FROM alpha_rechecks WHERE request_id = ? LIMIT 1",
        input.request_id
      )
      .toArray()[0];

    if (!prior) {
      this.ctx.storage.sql.exec(
        "INSERT INTO alpha_rechecks " +
        "(request_id,event_id,asset_id,alert_id,queued_at,updated_at,attempts,delivered_at,last_error,last_response) " +
        "VALUES (?,?,?,?,?,?,0,NULL,NULL,NULL)",
        input.request_id,
        input.event_id,
        input.asset_id,
        input.alert_id,
        now,
        now
      );
    }

    const current = this.ctx.storage.sql
      .exec<Json>(
        "SELECT * FROM alpha_rechecks WHERE request_id = ? LIMIT 1",
        input.request_id
      )
      .toArray()[0];

    if (!current?.delivered_at) {
      const existingAlarm = await this.ctx.storage.getAlarm();
      if (existingAlarm == null || existingAlarm > Date.now() + 500) {
        await this.ctx.storage.setAlarm(Date.now() + 250);
      }
    }

    return current ?? {};
  }

  async lookup(requestId: string): Promise<Json | null> {
    return this.ctx.storage.sql
      .exec<Json>(
        "SELECT * FROM alpha_rechecks WHERE request_id = ? LIMIT 1",
        requestId
      )
      .toArray()[0] ?? null;
  }

  async alarm(): Promise<void> {
    if (this.envRef.DURABLE_OBJECTS_ENABLED !== "true") return;
    const row = this.ctx.storage.sql
      .exec<Json>(
        "SELECT * FROM alpha_rechecks WHERE delivered_at IS NULL ORDER BY queued_at LIMIT 1"
      )
      .toArray()[0];

    if (!row) return;

    const requestId = String(row.request_id ?? "");
    const retry = async (message: string, responseText = "") => {
      this.ctx.storage.sql.exec(
        "UPDATE alpha_rechecks SET attempts = attempts + 1, updated_at = ?, last_error = ?, last_response = ? WHERE request_id = ?",
        new Date().toISOString(),
        message.slice(0, 1000),
        responseText.slice(0, 2000),
        requestId
      );
      await this.ctx.storage.setAlarm(Date.now() + 60000);
    };

    if (!this.envRef.ALPHA_RECHECK_URL || !this.envRef.BRIAN_CLOUDFLARE_KEY) {
      await retry("ALPHA_RECHECK_NOT_CONFIGURED");
      return;
    }

    try {
      const response = await fetch(this.envRef.ALPHA_RECHECK_URL, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-brian-cloudflare-key": this.envRef.BRIAN_CLOUDFLARE_KEY
        },
        body: JSON.stringify({
          request_id: requestId,
          event_id: String(row.event_id ?? ""),
          asset_id: String(row.asset_id ?? ""),
          alert_id: row.alert_id == null ? null : String(row.alert_id)
        }),
        signal: AbortSignal.timeout(55000)
      });

      const responseText = await response.text();

      if (!response.ok) {
        await retry(
          "ALPHA_HTTP_" + response.status + ":" + responseText.slice(0, 700),
          responseText
        );
        return;
      }

      this.ctx.storage.sql.exec(
        "UPDATE alpha_rechecks SET attempts = attempts + 1, updated_at = ?, delivered_at = ?, last_error = NULL, last_response = ? WHERE request_id = ?",
        new Date().toISOString(),
        new Date().toISOString(),
        responseText.slice(0, 2000),
        requestId
      );

      const remaining = this.ctx.storage.sql
        .exec<{ count: number }>(
          "SELECT COUNT(*) AS count FROM alpha_rechecks WHERE delivered_at IS NULL"
        )
        .one();
      if (Number(remaining.count) > 0) {
        await this.ctx.storage.setAlarm(Date.now() + 250);
      }
    } catch (error) {
      await retry("ALPHA_EXCEPTION:" + errText(error));
    }
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

function canonicalSourceUrl(value: string, canonicalDomain: string) {
  const raw = value.trim();
  if (!raw) return "";

  try {
    const url = new URL(raw);
    if (!hostMatches(url.hostname, canonicalDomain)) return raw;

    url.protocol = "https:";
    url.hostname = url.hostname.toLowerCase();
    url.hash = "";
    if ((url.protocol === "https:" && url.port === "443") ||
        (url.protocol === "http:" && url.port === "80")) {
      url.port = "";
    }

    // Normalize the official host identity so http/https and www/non-www
    // variants cannot create separate event identities.
    const host = url.hostname.replace(/^www\./, "");
    const path = url.pathname.replace(/\/{2,}/g, "/") || "/";
    return "https://" + host + path + url.search;
  } catch {
    return raw;
  }
}

function stableItemIdentity(endpoint: SourceEndpoint, item: FeedItem) {
  const guid = item.guid.trim();
  const linkValue = item.link.trim();

  if (/^https?:\/\//i.test(guid)) {
    return canonicalSourceUrl(guid, endpoint.canonical_domain);
  }
  if (/^https?:\/\//i.test(linkValue)) {
    return canonicalSourceUrl(linkValue, endpoint.canonical_domain);
  }
  return guid || linkValue || item.title.trim();
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
  const rdf = (doc["rdf:RDF"] ?? doc.RDF) as Json | undefined;
  const raw = arr((rss?.item ?? atom?.entry ?? rdf?.item) as Json | Json[] | undefined);
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

function decodeHtmlText(value: string) {
  return value
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/\s+/g, " ")
    .trim();
}

function parseHtmlLinks(raw: string, endpoint: SourceEndpoint): FeedItem[] {
  const prefix = endpoint.html_path_prefix?.trim() ?? "";
  if (!prefix.startsWith("/")) throw new Error("HTML_PATH_PREFIX_REQUIRED");

  const items: FeedItem[] = [];
  const seen = new Set<string>();
  const anchor = /<a\b[^>]*\bhref\s*=\s*(["'])(.*?)\1[^>]*>([\s\S]*?)<\/a>/gi;
  const datePatterns = [
    /\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b/i,
    /\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}\b/i,
    /\b\d{1,2}\/\d{1,2}\/\d{4}\b/
  ];

  const findDate = (value: string) => {
    for (const pattern of datePatterns) {
      const m = value.match(pattern);
      if (m?.[0]) return m[0];
    }
    return "";
  };

  let match: RegExpExecArray | null;
  while ((match = anchor.exec(raw)) && items.length < MAX_ITEMS_PER_FEED) {
    const hrefRaw = String(match[2] ?? "").replace(/&amp;/gi, "&").trim();
    let title = decodeHtmlText(String(match[3] ?? ""));
    if (!hrefRaw || !title) continue;

    let absolute = "";
    try {
      const u = new URL(hrefRaw, endpoint.endpoint_url);
      if (!hostMatches(u.hostname, endpoint.canonical_domain)) continue;
      if (!u.pathname.startsWith(prefix) || u.pathname === prefix.replace(/\/$/, "")) continue;
      u.hash = "";
      absolute = u.toString();
    } catch {
      continue;
    }

    if (seen.has(absolute)) continue;
    seen.add(absolute);

    const titleDate = findDate(title);
    if (titleDate) {
      title = title.replace(titleDate, "").replace(/^\s*[-|:–—]\s*/, "").trim();
    }

    const contextStart = Math.max(0, match.index - 1200);
    const context = decodeHtmlText(raw.slice(contextStart, match.index));
    let contextDate = "";
    for (const pattern of datePatterns) {
      const flags = pattern.flags.includes("g") ? pattern.flags : pattern.flags + "g";
      const all = [...context.matchAll(new RegExp(pattern.source, flags))];
      if (all.length) {
        const candidate = all[all.length - 1][0];
        if (!contextDate || Date.parse(candidate) > Date.parse(contextDate)) contextDate = candidate;
      }
    }

    const dateValue = titleDate || contextDate;
    const publishedAt = dateValue ? iso(dateValue) : null;

    if (!title) continue;
    items.push({
      title: title.slice(0, 1500),
      link: absolute,
      guid: absolute,
      publishedAt,
      categories: []
    });
  }
  return items;
}

function parseStatusPageJson(raw: string): FeedItem[] {
  const doc = JSON.parse(raw) as Json;
  const items: FeedItem[] = [];
  const page = (doc.page && typeof doc.page === "object") ? doc.page as Json : {};
  const status = (doc.status && typeof doc.status === "object") ? doc.status as Json : {};
  const incidents = Array.isArray(doc.incidents) ? doc.incidents as Json[] : [];

  const indicator = txt(status.indicator);
  const description = txt(status.description);
  if (indicator || description) {
    items.push({
      title: ("Coinbase Status: " + (description || indicator || "unknown")).slice(0, 1500),
      link: "",
      guid: "status:" + (indicator || "unknown") + ":" + (description || "unknown"),
      publishedAt: null,
      categories: [indicator || "unknown"]
    });
  }

  for (const incident of incidents.slice(0, 20)) {
    const title = txt(incident.name);
    const guid = txt(incident.id) || title;
    if (!title || !guid) continue;
    items.push({
      title: title.slice(0, 1500),
      link: txt(incident.shortlink),
      guid: "incident:" + guid,
      publishedAt: iso(incident.updated_at ?? incident.created_at),
      categories: [txt(incident.impact), txt(incident.status)].filter(Boolean)
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

async function requireAlphaRouteAuth(req: Request, env: Env) {
  const dedicated = req.headers.get("x-brian-cloudflare-key")?.trim() ?? "";
  const expectedDedicated = env.BRIAN_CLOUDFLARE_KEY?.trim() ?? "";
  if (sameSecret(expectedDedicated, dedicated)) return;

  const cronKey = req.headers.get("x-brian-cron-key")?.trim() ?? "";
  if (!cronKey) throw new Error("UNAUTHORIZED_ALPHA_ROUTE");
  const digest = await sha(cronKey);
  if (!sameSecret(SENTINEL_CRON_KEY_SHA256, digest)) {
    throw new Error("UNAUTHORIZED_ALPHA_ROUTE");
  }
}

async function fetchFeed(endpoint: SourceEndpoint) {
  const isJson = endpoint.endpoint_kind === "STATUSPAGE_JSON";
  const isHtml = endpoint.endpoint_kind === "HTML_LINKS";
  const timeoutMs =
    endpoint.endpoint_kind === "STATUSPAGE_ATOM" ? 20000 :
    isJson ? 12000 :
    isHtml ? 12000 :
    9000;
  const response = await fetch(endpoint.endpoint_url, {
    redirect: "follow",
    headers: {
      accept: isJson
        ? "application/json,*/*;q=0.1"
        : isHtml
          ? "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"
          : "application/rss+xml,application/atom+xml,application/xml,text/xml;q=0.9,*/*;q=0.1",
      "user-agent": "BrianMarketIntelligence/1.0 official-source-monitor"
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

function sourceStateKey(endpoint: SourceEndpoint) {
  return "source-state-v1/" + endpoint.endpoint_id + ".json";
}

async function sourcePayloadUnchanged(
  env: Env,
  endpoint: SourceEndpoint,
  payloadHash: string
) {
  if (env.R2_ENABLED !== "true" || !env.RAW_BUCKET) return false;
  const obj = await env.RAW_BUCKET.get(sourceStateKey(endpoint));
  if (!obj) return false;
  const state = await obj.json().catch(() => ({})) as Json;
  return String(state.payload_hash ?? "") === payloadHash;
}

async function persistSourceState(
  env: Env,
  endpoint: SourceEndpoint,
  payloadHash: string,
  observedAt: string
) {
  if (env.R2_ENABLED !== "true" || !env.RAW_BUCKET) return;
  await env.RAW_BUCKET.put(
    sourceStateKey(endpoint),
    JSON.stringify({
      endpoint_id: endpoint.endpoint_id,
      source_id: endpoint.source_id,
      payload_hash: payloadHash,
      observed_at: observedAt,
      runtime: VERSION
    }),
    {
      httpMetadata: { contentType: "application/json; charset=utf-8" },
      customMetadata: {
        endpoint_id: endpoint.endpoint_id,
        source_id: endpoint.source_id,
        payload_hash: payloadHash,
        shadow_only: "true"
      }
    }
  );
}

async function persistRaw(
  env: Env,
  endpoint: SourceEndpoint,
  observedAt: string,
  payloadHash: string,
  xml: string
) {
  const extension =
    endpoint.endpoint_kind === "STATUSPAGE_JSON" ? ".json" :
    endpoint.endpoint_kind === "HTML_LINKS" ? ".html" :
    ".xml";
  const path =
    "source-arch-v2/" +
    endpoint.endpoint_id +
    "/" +
    observedAt.slice(0, 10) +
    "/" +
    payloadHash +
    extension;

  if (env.R2_ENABLED !== "true" || !env.RAW_BUCKET) {
    return { stored: false, path, reason: "R2_DISABLED" };
  }

  const existing = await env.RAW_BUCKET.head(path);
  if (!existing) {
    await env.RAW_BUCKET.put(path, xml, {
    httpMetadata: { contentType:
      endpoint.endpoint_kind === "STATUSPAGE_JSON" ? "application/json; charset=utf-8" :
      endpoint.endpoint_kind === "HTML_LINKS" ? "text/html; charset=utf-8" :
      "application/xml; charset=utf-8"
    },
    customMetadata: {
      endpoint_id: endpoint.endpoint_id,
      source_id: endpoint.source_id,
      observed_at: observedAt,
      payload_hash: payloadHash,
      shadow_only: "true"
    }
  });
  }

  return { stored: true, path, reason: existing ? "ALREADY_STORED" : null };
}

function ledgerStub(env: Env, eventId: string) {
  const shard = eventId.slice(0, 2) || "00";
  const id = env.FIRST_SEEN_LEDGER.idFromName("events:" + shard);
  return { shard, stub: env.FIRST_SEEN_LEDGER.get(id) };
}

function alphaQueueStub(env: Env, requestId: string) {
  const shard = requestId.slice(0, 2) || "00";
  const id = env.ALPHA_RECHECK_QUEUE.idFromName("alpha:" + shard);
  return { shard, stub: env.ALPHA_RECHECK_QUEUE.get(id) };
}

async function r2RecordEvent(
  env: Env,
  event: EventEnvelope,
  capture: CaptureEnvelope,
  payloadHash: string
): Promise<LedgerResult> {
  if (!env.RAW_BUCKET || env.R2_OUTBOX_ENABLED !== "true") {
    throw new Error("R2_LEDGER_NOT_CONFIGURED");
  }

  const key = "event-ledger-v1/" + event.event_id.slice(0, 2) + "/" + event.event_id + ".json";
  const existing = await env.RAW_BUCKET.head(key);
  if (existing) {
    return {
      event_id: event.event_id,
      first_seen: false,
      first_seen_at: existing.customMetadata?.first_seen_at ?? event.first_observed_at,
      seen_count: 1,
      needs_forward: false,
      event,
      capture
    };
  }

  const now = new Date().toISOString();
  const storedEvent: EventEnvelope = {
    ...event,
    first_observed_at: now,
    metadata: {
      ...event.metadata,
      r2_ledger: true,
      durable_object_bypassed: true
    }
  };

  await env.RAW_BUCKET.put(key, JSON.stringify({
    event_id: event.event_id,
    source_id: event.source_id,
    published_at: event.published_at,
    payload_hash: payloadHash,
    first_seen_at: now,
    event: storedEvent,
    capture
  }), {
    httpMetadata: { contentType: "application/json; charset=utf-8" },
    customMetadata: {
      event_id: event.event_id,
      source_id: event.source_id,
      first_seen_at: now,
      shadow_only: "true"
    }
  });

  return {
    event_id: event.event_id,
    first_seen: true,
    first_seen_at: now,
    seen_count: 1,
    needs_forward: true,
    event: storedEvent,
    capture
  };
}

async function sendIngestBatch(env: Env, pending: LedgerResult[]) {
  const captures = new Map<string, CaptureEnvelope>();
  const events = new Map<string, EventEnvelope>();
  for (const row of pending) {
    captures.set(row.capture.capture_id, row.capture);
    events.set(row.event.event_id, row.event);
  }

  const payload = JSON.stringify({
    version: VERSION,
    captures: [...captures.values()],
    events: [...events.values()],
    rechecks: [],
    shadow_only: true,
    live_execution: false
  });

  const targets = [
    { name: "core", url: env.SUPABASE_INGEST_URL },
    ...(env.REALTIME_INGEST_URL ? [{ name: "realtime", url: env.REALTIME_INGEST_URL }] : [])
  ];

  const responses = await Promise.all(targets.map(async (target) => {
    const response = await fetch(target.url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-brian-cloudflare-key": env.BRIAN_CLOUDFLARE_KEY!
      },
      body: payload,
      signal: AbortSignal.timeout(12000)
    });
    const body = await response.text();
    if (!response.ok) {
      throw new Error(
        "INGEST_" + target.name.toUpperCase() + "_HTTP_" +
        response.status + ":" + body.slice(0, 500)
      );
    }
    return { target: target.name, http_status: response.status, body: body.slice(0, 500) };
  }));

  return { forwarded: events.size, responses };
}

async function queueR2IngestBatch(env: Env, pending: LedgerResult[], error: string) {
  if (!env.RAW_BUCKET) throw new Error("R2_BUCKET_MISSING");
  const ids = pending.map((x) => x.event_id).sort().join("|");
  const batchId = await sha("r2-ingest-outbox-v1|" + ids);
  const key = "outbox-pending/ingest/" + batchId + ".json";
  const exists = await env.RAW_BUCKET.head(key);
  if (!exists) {
    await env.RAW_BUCKET.put(key, JSON.stringify({
      batch_id: batchId,
      queued_at: new Date().toISOString(),
      last_error: error.slice(0, 1000),
      pending
    }), {
      httpMetadata: { contentType: "application/json; charset=utf-8" },
      customMetadata: { batch_id: batchId, shadow_only: "true" }
    });
  }
  return key;
}

async function forwardBatch(env: Env, pending: LedgerResult[]) {
  if (!pending.length) {
    return { forwarded: 0, queued: 0, status: "NO_NEW_EVENTS" };
  }

  if (env.DURABLE_OBJECTS_ENABLED === "true") {
    const scheduled = new Set<string>();
    for (const row of pending) {
      const target = ledgerStub(env, row.event_id);
      if (scheduled.has(target.shard)) continue;
      scheduled.add(target.shard);
      await target.stub.scheduleFlush(250);
    }
    return {
      forwarded: 0,
      queued: pending.length,
      status: "QUEUED_DURABLE_OUTBOX"
    };
  }

  try {
    const sent = await sendIngestBatch(env, pending);
    return { forwarded: sent.forwarded, queued: 0, status: "FORWARDED_DIRECT_R2_LEDGER" };
  } catch (error) {
    const key = await queueR2IngestBatch(env, pending, errText(error));
    return { forwarded: 0, queued: pending.length, status: "QUEUED_R2_OUTBOX", outbox_key: key };
  }
}

async function flushR2IngestOutbox(env: Env) {
  if (!env.RAW_BUCKET || env.R2_OUTBOX_ENABLED !== "true") return { status: "DISABLED", processed: 0 };
  const listed = await env.RAW_BUCKET.list({ prefix: "outbox-pending/ingest/", limit: 10 });
  let delivered = 0;
  for (const item of listed.objects) {
    const obj = await env.RAW_BUCKET.get(item.key);
    if (!obj) continue;
    const payload = await obj.json() as Json;
    const pending = Array.isArray(payload.pending) ? payload.pending as unknown as LedgerResult[] : [];
    if (!pending.length) {
      await env.RAW_BUCKET.delete(item.key);
      continue;
    }
    try {
      await sendIngestBatch(env, pending);
      await env.RAW_BUCKET.delete(item.key);
      delivered++;
    } catch {
      // Keep the object for the next cron pulse. No per-object alarm, no DO duration.
    }
  }
  return { status: "OK", processed: listed.objects.length, delivered };
}

async function deliverR2AlphaRequest(env: Env, requestId: string, payload: Json) {
  if (!env.RAW_BUCKET || !env.ALPHA_RECHECK_URL || !env.BRIAN_CLOUDFLARE_KEY) return false;

  const targets = [
    { name: "core", url: env.ALPHA_RECHECK_URL },
    ...(env.REALTIME_ALPHA_RECHECK_URL
      ? [{ name: "realtime", url: env.REALTIME_ALPHA_RECHECK_URL }]
      : [])
  ];

  try {
    const responses = await Promise.all(targets.map(async (target) => {
      const response = await fetch(target.url, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-brian-cloudflare-key": env.BRIAN_CLOUDFLARE_KEY!
        },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(20000)
      });
      const responseText = await response.text();
      if (!response.ok) {
        throw new Error(
          "ALPHA_" + target.name.toUpperCase() + "_HTTP_" +
          response.status + ":" + responseText.slice(0, 500)
        );
      }
      return {
        target: target.name,
        http_status: response.status,
        response: responseText.slice(0, 1500)
      };
    }));

    await env.RAW_BUCKET.put(
      "outbox-delivered/alpha/" + requestId + ".json",
      JSON.stringify({
        request_id: requestId,
        delivered_at: new Date().toISOString(),
        responses
      }),
      {
        httpMetadata: { contentType: "application/json; charset=utf-8" },
        customMetadata: { request_id: requestId, shadow_only: "true" }
      }
    );
    await env.RAW_BUCKET.delete("outbox-pending/alpha/" + requestId + ".json");
    return true;
  } catch {
    return false;
  }
}

async function flushR2AlphaOutbox(env: Env) {
  if (!env.RAW_BUCKET || env.R2_OUTBOX_ENABLED !== "true") return { status: "DISABLED", processed: 0 };
  const listed = await env.RAW_BUCKET.list({ prefix: "outbox-pending/alpha/", limit: 10 });
  let delivered = 0;
  for (const item of listed.objects) {
    const obj = await env.RAW_BUCKET.get(item.key);
    if (!obj) continue;
    const payload = await obj.json() as Json;
    const requestId = String(payload.request_id ?? "").trim();
    if (!requestId) {
      await env.RAW_BUCKET.delete(item.key);
      continue;
    }
    if (await deliverR2AlphaRequest(env, requestId, payload)) delivered++;
  }
  return { status: "OK", processed: listed.objects.length, delivered };
}

async function pollSource(env: Env, endpoint: SourceEndpoint) {
  const observedAt = new Date().toISOString();
  const nowMs = Date.parse(observedAt);
  const xml = await fetchFeed(endpoint);
  const raw = new TextEncoder().encode(xml);
  const payloadHash = await sha(raw);
  if (await sourcePayloadUnchanged(env, endpoint, payloadHash)) {
    return {
      endpoint_id: endpoint.endpoint_id,
      parsed: 0,
      selected: 0,
      pending: 0,
      forwarded: 0,
      forward_status: "UNCHANGED_SOURCE_HASH",
      raw_r2_committed: true,
      unchanged: true
    };
  }
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

  const parsed =
    endpoint.endpoint_kind === "STATUSPAGE_JSON"
      ? parseStatusPageJson(xml)
      : endpoint.endpoint_kind === "HTML_LINKS"
        ? parseHtmlLinks(xml, endpoint)
        : parseFeed(xml);
  const selected = parsed.filter((item) => freshEnough(item, nowMs));
  const pending: LedgerResult[] = [];

  for (const item of selected) {
    const stableIdentity = stableItemIdentity(endpoint, item);
    const canonicalLink = canonicalSourceUrl(item.link, endpoint.canonical_domain);
    const eventId = await sha(
      "source-arch-v2|" + endpoint.endpoint_id + "|" + stableIdentity
    );
    const fingerprint = await sha(
      endpoint.endpoint_id + "|" + item.title + "|" + canonicalLink
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
      provenance_uri: canonicalLink || endpoint.endpoint_url,
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
        stable_identity: stableIdentity,
        canonical_link: canonicalLink || null,
        direction_not_inferred: true,
        external_content_used_as_instruction: false,
        eligible_for_decision_evidence: false,
        decision_evidence_locked: true,
        cloudflare_shadow: true,
        shadow_only: true,
        live_execution: false
      }
    };

    const ledger = env.DURABLE_OBJECTS_ENABLED === "true"
      ? await (async () => {
          const target = ledgerStub(env, eventId);
          return await target.stub.record({
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
        })()
      : await r2RecordEvent(env, event, capture, payloadHash);

    if (ledger.needs_forward) pending.push(ledger);
  }

  const forwarded = await forwardBatch(env, pending);
  await persistSourceState(env, endpoint, payloadHash, observedAt);
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

  const results = await Promise.all(sources.map(async (endpoint): Promise<Json> => {
    try {
      return await pollSource(env, endpoint);
    } catch (error) {
      return {
        endpoint_id: endpoint.endpoint_id,
        error: errText(error).slice(0, 1200)
      };
    }
  }));

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

async function callRealtimeEngine(env: Env, name: string, url?: string) {
  if (!url || !env.BRIAN_CLOUDFLARE_KEY) {
    return { name, accepted: false, status: "NOT_CONFIGURED" };
  }
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-brian-cloudflare-key": env.BRIAN_CLOUDFLARE_KEY
      },
      body: "{}",
      signal: AbortSignal.timeout(50000)
    });
    const body = await response.json().catch(() => ({})) as Json;
    return {
      name,
      accepted: response.ok,
      http_status: response.status,
      target_status: String(body.status ?? ""),
      elapsed_ms: Number(body.elapsed_ms ?? 0) || null
    };
  } catch (error) {
    return {
      name,
      accepted: false,
      error: errText(error).slice(0, 700)
    };
  }
}

async function runRealtimeEngines(env: Env, scheduledMinute: number) {
  if (env.REALTIME_ENGINES_ENABLED !== "true") {
    return { status: "DISABLED", shadow_only: true, live_execution: false };
  }

  const results: Json[] = [];

  if (scheduledMinute % 5 === 0) {
    const universe = await callRealtimeEngine(
      env,
      "universe",
      env.REALTIME_UNIVERSE_URL
    );
    results.push(universe);

    if (universe.accepted) {
      results.push(await callRealtimeEngine(
        env,
        "sensor",
        env.REALTIME_SENSOR_URL
      ));
    } else {
      results.push({
        name: "sensor",
        accepted: false,
        status: "SKIPPED_UNIVERSE_FAILED"
      });
    }
  }

  results.push(await callRealtimeEngine(
    env,
    "intrabar",
    env.REALTIME_INTRABAR_URL
  ));

  if (scheduledMinute % 2 === 0) {
    results.push(await callRealtimeEngine(
      env,
      "alpha",
      env.REALTIME_ALPHA_COMPILER_URL
    ));
  }

  return {
    status: results.every((row) => row.accepted !== false) ? "SUCCESS" : "DEGRADED",
    results,
    shadow_only: true,
    live_execution: false
  };
}

async function runCoreRecovery(env: Env) {
  if (env.CORE_RECOVERY_ENABLED !== "true") {
    return { status: "DISABLED", shadow_only: true, live_execution: false };
  }
  if (!env.CORE_LAUNCHER_URL || !env.BRIAN_CLOUDFLARE_KEY) {
    return { status: "NOT_CONFIGURED", shadow_only: true, live_execution: false };
  }

  const rotating = ["discovery", "evolution", "ocean", "researcher", "sandbox"];
  const slot = Math.floor(Date.now() / 300000) % rotating.length;
  const services = ["alpha", "treasury", "world", rotating[slot]];

  const results = await Promise.all(services.map(async (service) => {
    try {
      const response = await fetch(env.CORE_LAUNCHER_URL!, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-brian-cloudflare-key": env.BRIAN_CLOUDFLARE_KEY!
        },
        body: JSON.stringify({ service }),
        signal: AbortSignal.timeout(12000)
      });
      const body = await response.json().catch(() => ({})) as Json;
      return {
        service,
        http_status: response.status,
        accepted: response.ok,
        target_status: String(body.status ?? "")
      };
    } catch (error) {
      return { service, accepted: false, error: errText(error).slice(0, 500) };
    }
  }));

  return {
    status: results.every((x) => x.accepted) ? "ACCEPTED" : "DEGRADED",
    services,
    results,
    shadow_only: true,
    live_execution: false
  };
}

async function routeAlphaRecheck(req: Request, env: Env, ctx?: ExecutionContext) {
  await requireAlphaRouteAuth(req, env);
  if (env.ALPHA_RECHECK_ENABLED !== "true") {
    return out({ status: "DISABLED_SHADOW_PHASE", shadow_only: true, live_execution: false }, 503);
  }
  if (!env.ALPHA_RECHECK_URL || !env.BRIAN_CLOUDFLARE_KEY) {
    return out({ status: "NOT_CONFIGURED" }, 503);
  }

  const body = await req.json().catch(() => ({})) as Json;
  const eventId = String(body.event_id ?? "").trim();
  const assetId = String(body.asset_id ?? "").trim();
  const alertId = String(body.alert_id ?? "").trim();

  if (!eventId || !/^crypto:[A-Z0-9]{2,20}USDT$/.test(assetId)) {
    return out({ error: "event_id and crypto:*USDT asset_id required" }, 400);
  }

  const requestId = String(body.request_id ?? "").trim() ||
    await sha("alpha-recheck-v1|" + eventId + "|" + assetId + "|" + alertId);

  if (env.DURABLE_OBJECTS_ENABLED === "true") {
    const target = alphaQueueStub(env, requestId);
    const state = await target.stub.enqueue({
      request_id: requestId,
      event_id: eventId,
      asset_id: assetId,
      alert_id: alertId || null
    });
    return out({
      status: state.delivered_at ? "ALREADY_DELIVERED" : "QUEUED_DURABLE_ALPHA_RECHECK",
      request_id: requestId,
      shard: target.shard,
      attempts: Number(state.attempts ?? 0),
      delivered_at: state.delivered_at ?? null,
      last_error: state.last_error ?? null,
      shadow_only: true,
      live_execution: false
    }, state.delivered_at ? 200 : 202);
  }

  if (!env.RAW_BUCKET || env.R2_OUTBOX_ENABLED !== "true") {
    return out({ status: "R2_OUTBOX_NOT_CONFIGURED", shadow_only: true, live_execution: false }, 503);
  }

  const deliveredKey = "outbox-delivered/alpha/" + requestId + ".json";
  if (await env.RAW_BUCKET.head(deliveredKey)) {
    return out({
      status: "ALREADY_DELIVERED",
      request_id: requestId,
      transport: "R2_OUTBOX",
      shadow_only: true,
      live_execution: false
    }, 200);
  }

  const pendingKey = "outbox-pending/alpha/" + requestId + ".json";
  if (!(await env.RAW_BUCKET.head(pendingKey))) {
    await env.RAW_BUCKET.put(pendingKey, JSON.stringify({
      request_id: requestId,
      event_id: eventId,
      asset_id: assetId,
      alert_id: alertId || null,
      queued_at: new Date().toISOString(),
      transport: "R2_OUTBOX",
      shadow_only: true,
      live_execution: false
    }), {
      httpMetadata: { contentType: "application/json; charset=utf-8" },
      customMetadata: { request_id: requestId, shadow_only: "true" }
    });
  }

  if (ctx) {
    ctx.waitUntil(deliverR2AlphaRequest(env, requestId, {
      request_id: requestId,
      event_id: eventId,
      asset_id: assetId,
      alert_id: alertId || null
    }));
  }

  return out({
    status: "QUEUED_R2_ALPHA_RECHECK",
    request_id: requestId,
    transport: "R2_OUTBOX",
    shadow_only: true,
    live_execution: false
  }, 202);
}
export default {
  async fetch(req: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const url = new URL(req.url);

    if (req.method === "GET" && (url.pathname === "/" || url.pathname === "/health" || url.pathname === "/health/")) {
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
        core_recovery_enabled: env.CORE_RECOVERY_ENABLED === "true",
        durable_objects_enabled: env.DURABLE_OBJECTS_ENABLED === "true",
        r2_outbox_enabled: env.R2_OUTBOX_ENABLED === "true",
        realtime_dual_write_enabled: Boolean(env.REALTIME_INGEST_URL),
        realtime_alpha_dual_write_enabled: Boolean(env.REALTIME_ALPHA_RECHECK_URL),
        realtime_engines_enabled: env.REALTIME_ENGINES_ENABLED === "true",
        realtime_universe_configured: Boolean(env.REALTIME_UNIVERSE_URL),
        realtime_sensor_configured: Boolean(env.REALTIME_SENSOR_URL),
        realtime_intrabar_configured: Boolean(env.REALTIME_INTRABAR_URL),
        realtime_alpha_compiler_configured: Boolean(env.REALTIME_ALPHA_COMPILER_URL),
        realtime_catalyst_reaction_configured: Boolean(env.REALTIME_CATALYST_REACTION_URL),
        shadow_only: true,
        live_execution: false
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
        return await routeAlphaRecheck(req, env, ctx);
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
      if (env.DURABLE_OBJECTS_ENABLED === "true") {
        const target = ledgerStub(env, eventId);
        return out({
          transport: "DURABLE_OBJECT",
          shard: target.shard,
          row: await target.stub.lookup(eventId)
        });
      }
      if (!env.RAW_BUCKET) return out({ status: "R2_NOT_CONFIGURED" }, 503);
      const key = "event-ledger-v1/" + eventId.slice(0, 2) + "/" + eventId + ".json";
      const obj = await env.RAW_BUCKET.get(key);
      return out({
        transport: "R2",
        key,
        row: obj ? await obj.json() : null
      });
    }

    return out({ error: "NOT_FOUND" }, 404);
  },

  async scheduled(
    controller: ScheduledController,
    env: Env,
    ctx: ExecutionContext
  ) {
    ctx.waitUntil((async () => {
      if (env.R2_OUTBOX_ENABLED === "true") {
        await flushR2AlphaOutbox(env);
        await flushR2IngestOutbox(env);
      }

      const scheduledMinute = Math.floor(controller.scheduledTime / 60000);

      const eyePromise =
        env.ENABLE_SCHEDULED_EYE === "true"
          ? runSources(env)
          : Promise.resolve({ status: "DISABLED" });

      const parallel: Promise<unknown>[] = [];

      if (env.REALTIME_ENGINES_ENABLED === "true") {
        parallel.push(runRealtimeEngines(env, scheduledMinute));
      }

      if (
        env.CORE_RECOVERY_ENABLED === "true" &&
        scheduledMinute % 5 === 0
      ) {
        parallel.push(runCoreRecovery(env));
      }

      await eyePromise;

      if (
        env.REALTIME_ENGINES_ENABLED === "true" &&
        env.REALTIME_CATALYST_REACTION_URL
      ) {
        await callRealtimeEngine(
          env,
          "catalyst_reaction",
          env.REALTIME_CATALYST_REACTION_URL
        );
      }

      if (parallel.length) await Promise.all(parallel);
    })());
  }
} satisfies ExportedHandler<Env>;

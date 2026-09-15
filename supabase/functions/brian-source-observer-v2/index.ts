import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { gzip } from "npm:pako@2.1.0";
import { XMLParser } from "npm:fast-xml-parser@4.5.0";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { withCollectorLease } from "../_shared/collector_lease.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(SUPABASE_URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const COLLECTOR_ID = "brian-source-observer-v2";
const BUCKET = "brian-intelligence-raw";
const EVIDENCE = "PROSPECTIVE_EVOLUTION_SHADOW";
const LEASE_SECONDS = 180;
const RUNTIME = "SOURCE_OBSERVER_V1";
const MAX_ITEM_AGE_MS = 48 * 3600_000;
const MAX_ITEMS_PER_FEED = 80;
const CONCURRENCY = 3;

type Endpoint = {
  endpoint_id: string;
  source_id: string;
  organization: string;
  endpoint_url: string;
  endpoint_kind: string;
  tier: string;
  category: string;
  region: string;
  polling_seconds: number;
  priority: number;
  authority_class: string;
  metadata: Record<string, unknown>;
};
type FeedItem = { title: string; link: string; guid: string; publishedAt: string | null; categories: string[] };

function out(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } });
}
function errorText(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  if (error && typeof error === "object") {
    const row = error as Record<string, unknown>;
    const fields = ["code", "message", "details", "hint", "status", "statusText"]
      .filter((key) => row[key] != null).map((key) => `${key}=${String(row[key])}`);
    if (fields.length) return fields.join(" | ");
    try { return JSON.stringify(error); } catch { /* ignore */ }
  }
  return String(error);
}
function arr<T>(value: T | T[] | null | undefined): T[] { return value == null ? [] : Array.isArray(value) ? value : [value]; }
function txt(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (value && typeof value === "object") {
    const r = value as Record<string, unknown>;
    return String(r["#text"] ?? r["@_term"] ?? r["@_label"] ?? "").trim();
  }
  return value == null ? "" : String(value).trim();
}
function link(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (value && typeof value === "object") {
    const r = value as Record<string, unknown>;
    return String(r["@_href"] ?? r.href ?? r["#text"] ?? "").trim();
  }
  return "";
}
function iso(value: unknown): string | null {
  const raw = txt(value);
  if (!raw) return null;
  const ms = Date.parse(raw);
  return Number.isFinite(ms) ? new Date(ms).toISOString() : null;
}
async function sha(value: string | Uint8Array): Promise<string> {
  const bytes = typeof value === "string" ? new TextEncoder().encode(value) : value;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function normalizedClaim(value: string): string {
  return value.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, " ").trim().slice(0, 500);
}
function categoryTexts(value: unknown): string[] {
  return arr(value as Record<string, unknown> | Record<string, unknown>[] | string | string[])
    .map((v) => typeof v === "string" ? v.trim() : txt(v)).filter(Boolean);
}
function parseFeed(xml: string): FeedItem[] {
  const parser = new XMLParser({ ignoreAttributes: false, trimValues: true, processEntities: true });
  const doc = parser.parse(xml) as Record<string, unknown>;
  const rss = (doc.rss as Record<string, unknown> | undefined)?.channel as Record<string, unknown> | undefined;
  const rssItems = arr(rss?.item as Record<string, unknown> | Record<string, unknown>[] | undefined);
  const atom = doc.feed as Record<string, unknown> | undefined;
  const atomItems = arr(atom?.entry as Record<string, unknown> | Record<string, unknown>[] | undefined);
  const raw = rssItems.length ? rssItems : atomItems;
  const items: FeedItem[] = [];
  for (const row of raw.slice(0, MAX_ITEMS_PER_FEED)) {
    if (!row || typeof row !== "object") continue;
    const title = txt(row.title).replace(/\s+/g, " ").trim();
    const href = link(row.link);
    const guid = txt(row.guid ?? row.id) || href || title;
    const publishedAt = iso(row.pubDate ?? row.published ?? row.updated);
    const categories = categoryTexts(row.category);
    if (!title || !guid) continue;
    items.push({ title: title.slice(0, 1500), link: href, guid, publishedAt, categories });
  }
  return items;
}
function secFormAllowed(endpoint: Endpoint, item: FeedItem): boolean {
  if (endpoint.endpoint_id !== "sec_edgar_current") return true;
  const configured = Array.isArray(endpoint.metadata?.forms) ? (endpoint.metadata.forms as unknown[]).map((x) => String(x).toUpperCase()) : [];
  if (!configured.length) return true;
  const hay = `${item.categories.join(" ")} ${item.title}`.toUpperCase();
  return configured.some((form) => hay.includes(form));
}
function freshEnough(item: FeedItem, nowMs: number): boolean {
  if (!item.publishedAt) return true;
  const ms = Date.parse(item.publishedAt);
  return Number.isFinite(ms) && nowMs - ms <= MAX_ITEM_AGE_MS && ms <= nowMs + 5 * 60_000;
}
async function fetchFeed(endpoint: Endpoint): Promise<string> {
  const timeout = endpoint.endpoint_id === "sec_edgar_current" ? 12_000 : 9_000;
  const r = await fetch(endpoint.endpoint_url, {
    headers: {
      accept: "application/rss+xml,application/atom+xml,application/xml,text/xml;q=0.9,*/*;q=0.1",
      "user-agent": "Brian-Market-OS/2.1 source-observer contact=owner",
    },
    signal: AbortSignal.timeout(timeout),
  });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const xml = await r.text();
  if (!xml.trim()) throw new Error("EMPTY_FEED");
  return xml;
}
async function persistRaw(endpoint: Endpoint, xml: string, observedAt: string): Promise<string> {
  const raw = new TextEncoder().encode(xml);
  const hash = await sha(raw);
  const compressed = gzip(raw, { level: 6 });
  const path = `source-arch-v2/${endpoint.endpoint_id}/${observedAt.slice(0, 10)}/${hash}.xml.gz`;
  const upload = await db.storage.from(BUCKET).upload(path, compressed, { contentType: "application/gzip", upsert: false, cacheControl: "31536000" });
  if (upload.error) {
    const msg = String(upload.error.message ?? "").toLowerCase();
    const code = String((upload.error as { statusCode?: string | number }).statusCode ?? "");
    if (code !== "409" && !msg.includes("exist") && !msg.includes("duplicate")) throw upload.error;
  }
  const captureId = await sha(`${endpoint.endpoint_id}|${observedAt}|${hash}`);
  const q = await db.from("brian_raw_captures").upsert({
    capture_id: captureId,
    provider: `source_v2:${endpoint.endpoint_id}`,
    record_type: "source_arch_v2_feed",
    observed_at: observedAt,
    captured_at: new Date().toISOString(),
    provenance_uri: endpoint.endpoint_url,
    payload_hash: hash,
    payload: {
      storage_bucket: BUCKET, storage_path: path, content_type: "application/xml", content_encoding: "gzip",
      source_arch_version: "V2", endpoint_id: endpoint.endpoint_id, source_id: endpoint.source_id,
      organization: endpoint.organization, tier: endpoint.tier, category: endpoint.category,
      external_content_used_as_instruction: false, decision_evidence_locked: true,
    },
  }, { onConflict: "capture_id", ignoreDuplicates: true });
  if (q.error) throw q.error;
  return captureId;
}
async function lastObserved(endpointId: string): Promise<number> {
  const q = await db.from("brian_raw_captures").select("observed_at")
    .eq("provider", `source_v2:${endpointId}`).order("observed_at", { ascending: false }).limit(1).maybeSingle();
  if (q.error) throw q.error;
  return q.data?.observed_at ? Date.parse(String(q.data.observed_at)) : 0;
}
async function observe(endpoint: Endpoint) {
  const observedAt = new Date().toISOString();
  const nowMs = Date.parse(observedAt);
  const xml = await fetchFeed(endpoint);
  const captureId = await persistRaw(endpoint, xml, observedAt);
  const parsed = parseFeed(xml);
  const selected = parsed.filter((item) => freshEnough(item, nowMs) && secFormAllowed(endpoint, item));
  const events: Record<string, unknown>[] = [];
  for (const item of selected) {
    const eventId = await sha(`source-arch-v2|${endpoint.endpoint_id}|${item.guid}`);
    const fingerprint = await sha(`${endpoint.endpoint_id}|${item.title}|${item.link}`);
    const normalized = normalizedClaim(item.title);
    const trustClass = endpoint.tier === "T1_OFFICIAL_PRIMARY" || endpoint.tier === "T0_RAW_TELEMETRY" ? "OFFICIAL_PRIMARY" : "INDEPENDENT_PROFESSIONAL";
    events.push({
      event_id: eventId,
      asset: "GLOBAL",
      event_kind: endpoint.category === "CRYPTO_EXCHANGE_STATUS" ? "OFFICIAL_EXCHANGE_STATUS" : "OFFICIAL_SOURCE_ITEM",
      source_kind: endpoint.tier,
      source_id: endpoint.source_id,
      published_at: item.publishedAt,
      first_observed_at: observedAt,
      captured_at: new Date().toISOString(),
      claim: item.title,
      direction: 0,
      magnitude: Math.max(0.1, Math.min(1, Number(endpoint.priority || 50) / 100)),
      trust_class: trustClass,
      entity_confidence: endpoint.tier === "T1_OFFICIAL_PRIMARY" ? 1 : .92,
      content_fingerprint: fingerprint,
      corroboration_key: normalized ? await sha(`${endpoint.category}|${normalized}`) : null,
      provenance_uri: item.link || endpoint.endpoint_url,
      pit_verified: true,
      raw_capture_id: captureId,
      metadata: {
        source_arch_version: "V2", runtime: RUNTIME, endpoint_id: endpoint.endpoint_id,
        organization: endpoint.organization, tier: endpoint.tier, category: endpoint.category, region: endpoint.region,
        categories: item.categories, direction_not_inferred: true, external_content_used_as_instruction: false,
        eligible_for_decision_evidence: false, decision_evidence_locked: true,
      },
    });
  }
  if (events.length) {
    const q = await db.from("brian_intel_events").upsert(events, { onConflict: "event_id", ignoreDuplicates: true });
    if (q.error) throw q.error;
  }
  return { endpoint_id: endpoint.endpoint_id, parsed: parsed.length, selected: selected.length, events: events.length, capture_id: captureId };
}
async function run() {
  const startedAt = new Date().toISOString();
  const q = await db.from("brian_source_registry_status_v2").select("endpoint_id,source_id,organization,endpoint_url,endpoint_kind,tier,category,region,polling_seconds,priority,authority_class,metadata,lifecycle_state,eligible_for_research")
    .eq("lifecycle_state", "ACTIVE").eq("eligible_for_research", true).in("endpoint_kind", ["RSS", "ATOM", "STATUSPAGE_ATOM"]).limit(100);
  if (q.error) throw q.error;
  const endpoints = (q.data ?? []).filter((row: any) => String(row.metadata?.adapter ?? "") === "GENERIC_FEED") as Endpoint[];
  const now = Date.now();
  const due: Endpoint[] = [];
  for (const endpoint of endpoints) {
    const last = await lastObserved(endpoint.endpoint_id);
    if (now - last >= Number(endpoint.polling_seconds || 600) * 1000) due.push(endpoint);
  }
  due.sort((a, b) => Number(b.priority) - Number(a.priority));
  const results: Record<string, unknown>[] = [];
  const degraded: string[] = [];
  for (let i = 0; i < due.length; i += CONCURRENCY) {
    const group = due.slice(i, i + CONCURRENCY);
    const settled = await Promise.all(group.map(async (endpoint) => {
      try { return await observe(endpoint); }
      catch (error) { degraded.push(`${endpoint.endpoint_id}:${errorText(error).slice(0,500)}`); return { endpoint_id: endpoint.endpoint_id, error: errorText(error) }; }
    }));
    results.push(...settled);
  }
  const stored = results.reduce((sum, row) => sum + Number(row.events ?? 0), 0);
  const runId = await sha(`${COLLECTOR_ID}|${startedAt}|${due.length}|${stored}`);
  const receipt = await db.from("brian_collector_runs").insert({
    run_id: runId, collector_id: COLLECTOR_ID, started_at: startedAt, finished_at: new Date().toISOString(),
    status: degraded.length ? "DEGRADED" : "SUCCESS", observed_records: results.reduce((s, r) => s + Number(r.parsed ?? 0), 0),
    stored_records: stored, degraded_sources: degraded.slice(0,24), error_class: null, error_message: null,
    evidence_class: EVIDENCE, shadow_only: true, live_execution: false,
    metadata: { source_arch_version: "V2", runtime: RUNTIME, endpoints_registered: endpoints.length, endpoints_due: due.length,
      external_content_used_as_instruction: false, decision_evidence_locked: true },
  });
  if (receipt.error) throw receipt.error;
  return { status: degraded.length ? "DEGRADED" : "SUCCESS", endpoints_registered: endpoints.length, endpoints_due: due.length,
    stored_events: stored, degraded_sources: degraded, results, shadow_only: true, live_execution: false, decision_evidence_locked: true, dip_touched: false };
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);
  try { await requireCronAuth(req, db); }
  catch (error) { return out({ status: "UNAUTHORIZED", error: errorText(error), shadow_only: true, live_execution: false }, 401); }
  try {
    const lease = await withCollectorLease(db, COLLECTOR_ID, LEASE_SECONDS, run);
    if (lease.contended) return out({ status: "SKIPPED_LEASE_CONTENDED", shadow_only: true, live_execution: false, dip_touched: false });
    return out(lease.value);
  } catch (error) {
    return out({ status: "FAILED_CLOSED", error: errorText(error), shadow_only: true, live_execution: false, dip_touched: false }, 500);
  }
});

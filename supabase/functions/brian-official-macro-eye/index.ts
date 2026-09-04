import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";
import { XMLParser } from "npm:fast-xml-parser@4.5.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });
const BUCKET = "brian-intelligence-raw";
const COLLECTOR_ID = "brian-official-macro-eye";
const EVIDENCE = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const LEASE_SECONDS = 90;
const RECENT_EVENT_MS = 30 * 60_000;

const FEEDS = [
  { id: "bls_employment", org: "BLS", url: "https://www.bls.gov/feed/empsit.rss", topic: "EMPLOYMENT_SITUATION" },
  { id: "bls_cpi", org: "BLS", url: "https://www.bls.gov/feed/cpi.rss", topic: "CPI" },
  { id: "bls_jolts", org: "BLS", url: "https://www.bls.gov/feed/jolts.rss", topic: "JOLTS" },
  { id: "fed_monetary", org: "FEDERAL_RESERVE", url: "https://www.federalreserve.gov/feeds/press_monetary.xml", topic: "MONETARY_POLICY" },
  { id: "ecb_press", org: "ECB", url: "https://www.ecb.europa.eu/rss/press.html", topic: "ECB_POLICY" },
] as const;

type FeedItem = { title: string; link: string; publishedAt: string | null; guid: string };

function json(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" } }); }
function utf8(s: string) { return new TextEncoder().encode(s); }
async function sha(value: string | Uint8Array) { const b = typeof value === "string" ? utf8(value) : value; const d = new Uint8Array(await crypto.subtle.digest("SHA-256", b)); return [...d].map((x) => x.toString(16).padStart(2, "0")).join(""); }
function errorText(e: unknown) { return e instanceof Error ? `${e.name}: ${e.message}` : String(e); }
function text(v: unknown): string { if (typeof v === "string") return v.trim(); if (v && typeof v === "object" && "#text" in (v as Record<string, unknown>)) return String((v as Record<string, unknown>)["#text"] ?? "").trim(); return v == null ? "" : String(v).trim(); }
function linkText(v: unknown): string { if (typeof v === "string") return v.trim(); if (v && typeof v === "object") { const r = v as Record<string, unknown>; return String(r.href ?? r["@_href"] ?? r["#text"] ?? "").trim(); } return ""; }
function dateIso(v: unknown): string | null { const raw = text(v); if (!raw) return null; const ms = Date.parse(raw); return Number.isFinite(ms) ? new Date(ms).toISOString() : null; }
function arrayify<T>(v: T | T[] | undefined | null): T[] { return v == null ? [] : Array.isArray(v) ? v : [v]; }

function parseItems(xml: string): FeedItem[] {
  const parser = new XMLParser({ ignoreAttributes: false, trimValues: true, processEntities: true });
  const doc = parser.parse(xml) as Record<string, unknown>;
  const rssChannel = (doc.rss as Record<string, unknown> | undefined)?.channel as Record<string, unknown> | undefined;
  const rssItems = arrayify(rssChannel?.item as Record<string, unknown> | Record<string, unknown>[] | undefined);
  const atomFeed = doc.feed as Record<string, unknown> | undefined;
  const atomItems = arrayify(atomFeed?.entry as Record<string, unknown> | Record<string, unknown>[] | undefined);
  const raw = rssItems.length ? rssItems : atomItems;
  const out: FeedItem[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const title = text(item.title); const link = linkText(item.link); const publishedAt = dateIso(item.pubDate ?? item.published ?? item.updated);
    const guid = text(item.guid ?? item.id) || link || title;
    if (!title || !guid) continue;
    out.push({ title, link, publishedAt, guid });
  }
  return out;
}

async function persistRaw(feed: typeof FEEDS[number], xml: string, observedAt: string): Promise<string> {
  const raw = utf8(xml), hash = await sha(raw), z = gzip(raw, { level: 6 }); const path = `official_macro/${feed.id}/${observedAt.slice(0,10)}/${hash}.xml.gz`;
  const up = await supabase.storage.from(BUCKET).upload(path, z, { contentType: "application/gzip", upsert: false, cacheControl: "31536000" });
  if (up.error) { const msg = String(up.error.message ?? "").toLowerCase(); const status = String((up.error as {statusCode?: string|number}).statusCode ?? ""); if (status !== "409" && !msg.includes("exist") && !msg.includes("duplicate")) throw up.error; }
  const id = await sha(`${feed.id}|${observedAt}|${hash}`); const ins = await supabase.from("brian_raw_captures").insert({ capture_id: id, provider: feed.id, record_type: "official_macro_rss", observed_at: observedAt, captured_at: new Date().toISOString(), provenance_uri: feed.url, payload_hash: hash, payload: { storage_bucket: BUCKET, storage_path: path, content_type: "application/rss+xml", content_encoding: "gzip", official_source: true, organization: feed.org, topic: feed.topic } }); if (ins.error) throw ins.error; return id;
}

async function fetchFeed(feed: typeof FEEDS[number]) {
  const r = await fetch(feed.url, { headers: { accept: "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.1", "user-agent": "Brian-Market-OS/1.0 official-macro-research contact=owner" }, signal: AbortSignal.timeout(8_000) });
  if (!r.ok) throw new Error(`${feed.id} HTTP ${r.status}`); const xml = await r.text(); if (!xml.trim()) throw new Error(`${feed.id} empty feed`); return xml;
}

async function recordRun(started: string, status: string, observed: number, stored: number, degraded: string[], error?: unknown) {
  const finished = new Date().toISOString(), id = await sha(`${COLLECTOR_ID}|${started}|${finished}|${status}`); await supabase.from("brian_collector_runs").insert({ run_id: id, collector_id: COLLECTOR_ID, started_at: started, finished_at: finished, status, observed_records: observed, stored_records: stored, degraded_sources: degraded, error_class: error ? "OFFICIAL_MACRO_ERROR" : null, error_message: error ? errorText(error).slice(0,1500) : null, evidence_class: EVIDENCE, shadow_only: true, live_execution: false, metadata: { feed_count: FEEDS.length } });
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return json({ error: "POST required" }, 405);
  const started = new Date().toISOString();
  try {
    const lease = await withCollectorLease(supabase, COLLECTOR_ID, LEASE_SECONDS, async () => {
      const observedAt = new Date().toISOString(), nowMs = Date.parse(observedAt); const results = await Promise.allSettled(FEEDS.map(async (feed) => ({ feed, xml: await fetchFeed(feed) })));
      const degraded: string[] = []; const events: Record<string, unknown>[] = []; const observations: Record<string, unknown>[] = []; let observed = 0;
      for (let i = 0; i < results.length; i++) {
        const result = results[i], feed = FEEDS[i]; if (result.status === "rejected") { degraded.push(`${feed.id}:${errorText(result.reason)}`); continue; }
        const xml = result.value.xml; const captureId = await persistRaw(feed, xml, observedAt); const items = parseItems(xml).slice(0, 30); observed += items.length;
        for (const item of items) {
          const eventId = await sha(`official-macro|${feed.id}|${item.guid}|${item.title}`); const fingerprint = await sha(`${feed.id}|${item.title}|${item.link}`);
          events.push({ event_id: eventId, asset: "MACRO", event_kind: "OFFICIAL_MACRO_RELEASE", source_kind: `${feed.org}_OFFICIAL`, source_id: feed.id, published_at: item.publishedAt, first_observed_at: observedAt, captured_at: new Date().toISOString(), claim: item.title, direction: 0, magnitude: 1, trust_class: "OFFICIAL_PRIMARY", entity_confidence: 1, content_fingerprint: fingerprint, corroboration_key: await sha(`${feed.topic}|${item.title.toLowerCase().replace(/[^a-z0-9]+/g," ").trim()}`), provenance_uri: item.link || feed.url, pit_verified: true, raw_capture_id: captureId, metadata: { organization: feed.org, topic: feed.topic, rss_feed: feed.url, direction_not_inferred: true } });
          const publishedMs = item.publishedAt ? Date.parse(item.publishedAt) : NaN; if (!Number.isFinite(publishedMs) || Math.abs(nowMs - publishedMs) > RECENT_EVENT_MS) continue;
          const eye = await sha(`official-macro-event|${feed.id}`), obsId = await sha(`${eye}|${eventId}|${observedAt}`);
          observations.push({ observation_id: obsId, eye_id: eye, template_id: "official-macro-event", asset_id: "global:MACRO", market_domain: "macro", sensor_family: "official_macro_event", horizon: "EVENT_DRIVEN", independent_group: `official_macro_${feed.org.toLowerCase()}`, observed_at: observedAt, direction: 0, strength: 1, confidence: 1, reliability: 1, available: true, source_ids: [captureId], reason: `${feed.org} official ${feed.topic} release observed; direction intentionally delegated to market reaction sensors`, evidence_class: EVIDENCE, shadow_only: true, live_execution: false, metadata: { event_id: eventId, published_at: item.publishedAt, title: item.title, provenance_uri: item.link || feed.url, direction_not_inferred: true } });
        }
      }
      if (!events.length && degraded.length === FEEDS.length) throw new Error(`all official macro feeds failed: ${degraded.join(" | ")}`);
      if (events.length) { const ins = await supabase.from("brian_intel_events").upsert(events, { onConflict: "event_id", ignoreDuplicates: true }); if (ins.error) throw ins.error; }
      if (observations.length) { const ins = await supabase.from("brian_sensor_observations").insert(observations); if (ins.error) throw ins.error; }
      const status = degraded.length ? "DEGRADED" : "SUCCESS"; await recordRun(started, status, observed, events.length + observations.length, degraded);
      return json({ status, feeds_ok: FEEDS.length - degraded.length, feeds_failed: degraded.length, observed_items: observed, intel_events: events.length, recent_macro_observations: observations.length, degraded_sources: degraded, direction_inference: false, shadow_only: true, live_execution: false });
    });
    if (lease.contended) return json({ status: "SKIPPED_LEASE_CONTENDED", shadow_only: true, live_execution: false }); return lease.value!;
  } catch (error) { try { await recordRun(started, "FAILED", 0, 0, [], error); } catch {} return json({ status: "FAILED_CLOSED", error: errorText(error), shadow_only: true, live_execution: false }, 500); }
});

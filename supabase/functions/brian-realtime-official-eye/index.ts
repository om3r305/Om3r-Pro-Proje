import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { XMLParser } from "npm:fast-xml-parser@4.5.0";
import { gzip } from "npm:pako@2.1.0";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(SUPABASE_URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });

const VERSION = "brian.realtime-official-eye.v1";
const COLLECTOR_ID = "brian-realtime-official-eye-v1";
const BUCKET = "brian-intelligence-raw";
const INTERNAL_KEY_SHA256 = "b0549b2b41a5b832b37455389583e1d166d210490a8c6fe43cda2748aca7c38a";
const MAX_ITEMS = 80;
const MAX_AGE_MS = 48 * 60 * 60 * 1000;

type Json = Record<string, unknown>;
type Endpoint = {
  endpoint_id: string;
  source_id: string;
  organization: string;
  canonical_domain: string;
  endpoint_url: string;
  endpoint_kind: string;
  tier: string;
  category: string;
  region: string;
  priority: number;
  metadata: Json;
};
type FeedItem = { title: string; link: string; guid: string; publishedAt: string | null; categories: string[] };

function out(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}
function errText(error: unknown) {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  try { return JSON.stringify(error); } catch { return String(error); }
}
function arr<T>(value: T | T[] | null | undefined): T[] { return value == null ? [] : Array.isArray(value) ? value : [value]; }
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
  return value.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, " ").trim().slice(0, 500);
}
async function sha(value: string | Uint8Array) {
  const bytes = typeof value === "string" ? new TextEncoder().encode(value) : value;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}
function sameSecret(a: string, b: string) {
  if (!a || !b || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}
async function requireInternal(req: Request) {
  const supplied = (req.headers.get("x-brian-internal-key") ?? "").trim();
  if (!supplied) throw new Error("UNAUTHORIZED");
  const digest = await sha(supplied);
  if (!sameSecret(digest, INTERNAL_KEY_SHA256)) throw new Error("UNAUTHORIZED");
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
    const host = url.hostname.replace(/^www\./, "");
    const path = url.pathname.replace(/\/{2,}/g, "/") || "/";
    return "https://" + host + path + url.search;
  } catch { return raw; }
}
function stableItemIdentity(endpoint: Endpoint, item: FeedItem) {
  if (/^https?:\/\//i.test(item.guid)) return canonicalSourceUrl(item.guid, endpoint.canonical_domain);
  if (/^https?:\/\//i.test(item.link)) return canonicalSourceUrl(item.link, endpoint.canonical_domain);
  return item.guid || item.link || item.title;
}
function parseFeed(xml: string): FeedItem[] {
  const parser = new XMLParser({ ignoreAttributes: false, trimValues: true, processEntities: true });
  const doc = parser.parse(xml) as Json;
  const rss = (doc.rss as Json | undefined)?.channel as Json | undefined;
  const atom = doc.feed as Json | undefined;
  const rdf = (doc["rdf:RDF"] ?? doc.RDF) as Json | undefined;
  const raw = arr((rss?.item ?? atom?.entry ?? rdf?.item) as Json | Json[] | undefined);
  const items: FeedItem[] = [];
  for (const row of raw.slice(0, MAX_ITEMS)) {
    if (!row || typeof row !== "object") continue;
    const title = txt(row.title).replace(/\s+/g, " ").trim();
    const links = arr(row.link as unknown);
    let href = "";
    for (const candidate of links) { href = link(candidate); if (href) break; }
    const guid = txt(row.guid ?? row.id) || href || title;
    const publishedAt = iso(row.pubDate ?? row.published ?? row.updated ?? row["dc:date"]);
    const categories = arr(row.category as unknown).map((x) => txt(x)).filter(Boolean);
    if (!title || !guid) continue;
    items.push({ title: title.slice(0, 1500), link: href, guid, publishedAt, categories });
  }
  return items;
}
function decodeHtmlText(value: string) {
  return value.replace(/<script[\s\S]*?<\/script>/gi, " ").replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ").replace(/&nbsp;/gi, " ").replace(/&amp;/gi, "&").replace(/&quot;/gi, '"')
    .replace(/&#39;/gi, "'").replace(/&lt;/gi, "<").replace(/&gt;/gi, ">").replace(/\s+/g, " ").trim();
}
function parseHtmlLinks(raw: string, endpoint: Endpoint): FeedItem[] {
  const prefix = String(endpoint.metadata?.html_path_prefix ?? "").trim();
  if (!prefix.startsWith("/")) throw new Error("HTML_PATH_PREFIX_REQUIRED");
  const items: FeedItem[] = [];
  const seen = new Set<string>();
  const anchor = /<a\b[^>]*\bhref\s*=\s*(["'])(.*?)\1[^>]*>([\s\S]*?)<\/a>/gi;
  const datePatterns = [
    /\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b/i,
    /\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}\b/i,
    /\b\d{1,2}\/\d{1,2}\/\d{4}\b/,
  ];
  let match: RegExpExecArray | null;
  while ((match = anchor.exec(raw)) && items.length < MAX_ITEMS) {
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
    } catch { continue; }
    if (seen.has(absolute)) continue;
    seen.add(absolute);
    let dateValue = "";
    for (const p of datePatterns) {
      const m = title.match(p);
      if (m?.[0]) { dateValue = m[0]; title = title.replace(m[0], "").replace(/^\s*[-|:–—]\s*/, "").trim(); break; }
    }
    if (!dateValue) {
      const context = decodeHtmlText(raw.slice(Math.max(0, match.index - 1200), match.index));
      for (const p of datePatterns) {
        const flags = p.flags.includes("g") ? p.flags : p.flags + "g";
        const all = [...context.matchAll(new RegExp(p.source, flags))];
        if (all.length) dateValue = all[all.length - 1][0];
      }
    }
    if (!title) continue;
    items.push({ title: title.slice(0, 1500), link: absolute, guid: absolute, publishedAt: dateValue ? iso(dateValue) : null, categories: [] });
  }
  return items;
}
function freshEnough(item: FeedItem, nowMs: number) {
  if (!item.publishedAt) return true;
  const ms = Date.parse(item.publishedAt);
  return Number.isFinite(ms) && ms <= nowMs + 5 * 60_000 && nowMs - ms <= MAX_AGE_MS;
}
async function fetchSource(endpoint: Endpoint) {
  const isHtml = endpoint.endpoint_kind === "HTML";
  const response = await fetch(endpoint.endpoint_url, {
    redirect: "follow",
    headers: {
      accept: isHtml ? "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"
        : "application/rss+xml,application/atom+xml,application/xml,text/xml;q=0.9,*/*;q=0.1",
      "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/140.0 Safari/537.36 BrianRealtimeOfficialEye/1.0",
      "accept-language": "en-US,en;q=0.9",
    },
    signal: AbortSignal.timeout(isHtml ? 12000 : 9000),
  });
  if (!response.ok) throw new Error("HTTP_" + response.status);
  const finalHost = new URL(response.url || endpoint.endpoint_url).hostname;
  if (!hostMatches(finalHost, endpoint.canonical_domain)) throw new Error("ORIGIN_MISMATCH:" + finalHost);
  const body = await response.text();
  if (!body.trim()) throw new Error("EMPTY_SOURCE");
  return { body, status: response.status, contentType: response.headers.get("content-type") ?? "" };
}
async function getState(endpointId: string) {
  const q = await db.from("brian_realtime_source_state").select("payload_hash").eq("endpoint_id", endpointId).maybeSingle();
  if (q.error) throw q.error;
  return q.data?.payload_hash ? String(q.data.payload_hash) : "";
}
async function setState(endpoint: Endpoint, payloadHash: string, status: number, changed: boolean, error: string | null = null) {
  const now = new Date().toISOString();
  const row: Json = {
    endpoint_id: endpoint.endpoint_id,
    payload_hash: payloadHash || null,
    last_fetch_at: now,
    last_success_at: error ? null : now,
    last_http_status: status || null,
    last_error: error,
    updated_at: now,
  };
  if (changed) row.last_change_at = now;
  const q = await db.from("brian_realtime_source_state").upsert(row, { onConflict: "endpoint_id" });
  if (q.error) throw q.error;
  if (!error) {
    await db.from("brian_source_endpoints_v2").update({ lifecycle_state: "ACTIVE", updated_at: now }).eq("endpoint_id", endpoint.endpoint_id);
  }
}
async function persistRaw(endpoint: Endpoint, payloadHash: string, body: string) {
  const ext = endpoint.endpoint_kind === "HTML" ? ".html.gz" : ".xml.gz";
  const path = `source-arch-v2/${endpoint.endpoint_id}/${new Date().toISOString().slice(0,10)}/${payloadHash}${ext}`;
  const compressed = gzip(new TextEncoder().encode(body), { level: 6 });
  const upload = await db.storage.from(BUCKET).upload(path, compressed, {
    contentType: "application/gzip",
    upsert: false,
    cacheControl: "31536000",
  });
  if (upload.error) {
    const msg = String(upload.error.message ?? "").toLowerCase();
    if (!msg.includes("exist") && !msg.includes("duplicate")) throw upload.error;
  }
  return path;
}
const CORE_ASSETS = ["crypto:BTCUSDT","crypto:ETHUSDT","crypto:SOLUSDT","crypto:BNBUSDT","crypto:XRPUSDT","crypto:ADAUSDT"];
function explicitAssets(claim: string) {
  const out = new Set<string>();
  const map: Array<[RegExp,string]> = [
    [/\bbitcoin\b|\bbtc\b/i,"crypto:BTCUSDT"],[/\bethereum\b|\bether\b|\beth\b/i,"crypto:ETHUSDT"],
    [/\bsolana\b|\bsol\b/i,"crypto:SOLUSDT"],[/\bbnb\b|binance coin/i,"crypto:BNBUSDT"],
    [/\bxrp\b|\bripple\b/i,"crypto:XRPUSDT"],[/\bcardano\b|\bada\b/i,"crypto:ADAUSDT"],
  ];
  for (const [re,a] of map) if (re.test(claim)) out.add(a);
  return [...out];
}
function catalystBinding(event: Json) {
  const claim = String(event.claim ?? "");
  const metadata = (event.metadata ?? {}) as Json;
  const category = String(metadata.category ?? "");
  const sourceId = String(event.source_id ?? "");
  const monetarySource = /CENTRAL_BANK|MACRO_INFLATION|MACRO_EMPLOYMENT|CENTRAL_BANK_MARKETS/i.test(category)
    || /official:(fed|ecb|boj|boe|boc|bundesbank|nyfed):/i.test(sourceId);
  const monetary = monetarySource
    && /\bfomc\b|monetary policy|policy rate|interest rate|rate cut|rate hike|governing council|inflation|economic projections|minutes|central bank/i.test(claim);
  const broadCrypto = /FINANCIAL_REGULATION|DERIVATIVES|TREASURY_POLICY_SANCTIONS/i.test(category)
    && /crypto|digital asset|bitcoin|ethereum|stablecoin|blockchain|exchange-traded|\betf\b|token/i.test(claim);
  const explicit = explicitAssets(claim);
  if (monetary) return { relevant: true, isMacro: true, assets: CORE_ASSETS, policy: "MONETARY_CORE_BASKET" };
  if (explicit.length) return { relevant: true, isMacro: false, assets: explicit, policy: "EXPLICIT_ASSET_MENTION" };
  if (broadCrypto) return { relevant: true, isMacro: false, assets: ["crypto:BTCUSDT","crypto:ETHUSDT"], policy: "SYSTEMIC_CRYPTO_POLICY" };
  return { relevant: false, isMacro: false, assets: [] as string[], policy: "NO_CATALYST_BINDING" };
}
async function recordRun(startedAt: string, status: string, observed: number, stored: number, degraded: string[], metadata: Json = {}) {
  const finishedAt = new Date().toISOString();
  const runId = await sha(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q = await db.from("brian_collector_runs").insert({
    run_id: runId, collector_id: COLLECTOR_ID, started_at: startedAt, finished_at: finishedAt,
    status, observed_records: observed, stored_records: stored, degraded_sources: degraded.slice(0,24),
    error_class: null, error_message: null, evidence_class: "PROSPECTIVE_DEVELOPMENT_SHADOW",
    shadow_only: true, live_execution: false, metadata: { version: VERSION, ...metadata },
  });
  if (q.error) console.error("collector receipt", q.error.message);
}
async function observe(endpoint: Endpoint) {
  const observedAt = new Date().toISOString();
  const nowMs = Date.parse(observedAt);
  const fetched = await fetchSource(endpoint);
  const payloadHash = await sha(fetched.body);
  const priorHash = await getState(endpoint.endpoint_id);
  if (priorHash === payloadHash) {
    await setState(endpoint, payloadHash, fetched.status, false);
    return { endpoint_id: endpoint.endpoint_id, unchanged: true, parsed: 0, stored: 0, catalysts: 0 };
  }

  const items = endpoint.endpoint_kind === "HTML" ? parseHtmlLinks(fetched.body, endpoint) : parseFeed(fetched.body);
  const selected = items.filter((item) => freshEnough(item, nowMs));
  const storagePath = await persistRaw(endpoint, payloadHash, fetched.body);
  const captureId = await sha(`realtime-eye|${endpoint.endpoint_id}|${payloadHash}`);
  const capture = await db.from("brian_raw_captures").upsert({
    capture_id: captureId,
    provider: `realtime_eye:${endpoint.endpoint_id}`,
    record_type: "official_source_fastlane",
    observed_at: observedAt,
    captured_at: new Date().toISOString(),
    provenance_uri: endpoint.endpoint_url,
    payload_hash: payloadHash,
    payload: {
      storage_bucket: BUCKET, storage_path: storagePath, endpoint_id: endpoint.endpoint_id,
      source_id: endpoint.source_id, organization: endpoint.organization, category: endpoint.category,
      runtime: VERSION, decision_evidence_locked: true, shadow_only: true, live_execution: false,
    },
  }, { onConflict: "capture_id", ignoreDuplicates: true });
  if (capture.error) throw capture.error;

  const events: Json[] = [];
  for (const item of selected) {
    const stable = stableItemIdentity(endpoint, item);
    const eventId = await sha(`official-fastlane-v1|${endpoint.endpoint_id}|${stable}`);
    const fingerprint = await sha(`${endpoint.endpoint_id}|${item.title}|${item.link}`);
    const normalized = normalizeClaim(item.title);
    events.push({
      event_id: eventId, asset: "GLOBAL", event_kind: "OFFICIAL_SOURCE_ITEM",
      source_kind: endpoint.tier, source_id: endpoint.source_id, published_at: item.publishedAt,
      first_observed_at: observedAt, captured_at: new Date().toISOString(), claim: item.title,
      direction: 0, magnitude: Math.max(.1, Math.min(1, Number(endpoint.priority || 50) / 100)),
      trust_class: "OFFICIAL_PRIMARY", entity_confidence: 1, content_fingerprint: fingerprint,
      corroboration_key: normalized ? await sha(`${endpoint.category}|${normalized}`) : null,
      provenance_uri: item.link || endpoint.endpoint_url, pit_verified: true, raw_capture_id: captureId,
      metadata: {
        runtime: VERSION, endpoint_id: endpoint.endpoint_id, organization: endpoint.organization,
        tier: endpoint.tier, category: endpoint.category, region: endpoint.region, categories: item.categories,
        direction_not_inferred: true, eligible_for_decision_evidence: false,
        decision_evidence_locked: true, external_content_used_as_instruction: false,
        shadow_only: true, live_execution: false,
      },
    });
  }

  if (events.length) {
    const q = await db.from("brian_intel_events").upsert(events, { onConflict: "event_id", ignoreDuplicates: true });
    if (q.error) throw q.error;
  }

  let catalysts = 0;
  for (const event of events) {
    const binding = catalystBinding(event);
    if (!binding.relevant) continue;
    const rpc = await db.rpc("brian_catalyst_ingest_official_event_v23", {
      p_event: { ...event, is_macro: binding.isMacro, metadata: { ...(event.metadata as Json), catalyst_binding_policy: binding.policy, catalyst_bound_assets: binding.assets } },
      p_assets: binding.assets,
    });
    if (rpc.error) throw rpc.error;
    if (Array.isArray(rpc.data)) catalysts += rpc.data.length;
  }

  await setState(endpoint, payloadHash, fetched.status, true);
  return { endpoint_id: endpoint.endpoint_id, unchanged: false, parsed: items.length, selected: selected.length, stored: events.length, catalysts };
}
async function run() {
  const startedAt = new Date().toISOString();
  const q = await db.from("brian_source_endpoints_v2")
    .select("endpoint_id,source_id,organization,canonical_domain,endpoint_url,endpoint_kind,tier,category,region,priority,metadata")
    .eq("collector_owner", "brian-realtime-official-eye")
    .eq("official_origin", true)
    .order("priority", { ascending: false })
    .limit(20);
  if (q.error) throw q.error;
  const endpoints = (q.data ?? []) as Endpoint[];
  const results = await Promise.all(endpoints.map(async (endpoint) => {
    try { return await observe(endpoint); }
    catch (error) {
      const message = errText(error).slice(0, 800);
      try { await setState(endpoint, "", 0, false, message); } catch {}
      return { endpoint_id: endpoint.endpoint_id, error: message };
    }
  }));
  const degraded = results.filter((r) => "error" in r).map((r) => `${r.endpoint_id}:${String((r as Json).error)}`);
  const observed = results.reduce((s,r) => s + Number((r as Json).parsed ?? 0), 0);
  const stored = results.reduce((s,r) => s + Number((r as Json).stored ?? 0), 0);
  const catalysts = results.reduce((s,r) => s + Number((r as Json).catalysts ?? 0), 0);
  await recordRun(startedAt, degraded.length ? "DEGRADED" : "SUCCESS", observed, stored, degraded, {
    sources: endpoints.length, catalysts, parallel: true, poll_seconds: 60,
  });
  return {
    status: degraded.length ? "DEGRADED" : "SUCCESS",
    version: VERSION, sources: endpoints.length, stored_events: stored, catalysts,
    degraded_sources: degraded, results, shadow_only: true, live_execution: false,
  };
}

Deno.serve(async (req: Request) => {
  if (req.method === "GET") return out({ status: "OK", version: VERSION, shadow_only: true, live_execution: false });
  if (req.method !== "POST") return out({ error: "POST required" }, 405);
  try { await requireInternal(req); } catch { return out({ status: "UNAUTHORIZED" }, 401); }
  try { return out(await run()); }
  catch (error) {
    return out({ status: "FAILED_CLOSED", version: VERSION, error: errText(error).slice(0,1200), shadow_only: true, live_execution: false }, 500);
  }
});

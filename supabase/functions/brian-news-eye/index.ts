import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, { auth: { persistSession: false, autoRefreshToken: false } });

const COLLECTOR_ID = "phase39-multisource-news";
const EVIDENCE = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const BUCKET = "brian-intelligence-raw";
const OFFICIAL = [
  { id: "FED", url: "https://www.federalreserve.gov/feeds/press_all.xml" },
  { id: "ECB", url: "https://www.ecb.europa.eu/rss/press.html" },
  { id: "SEC", url: "https://www.sec.gov/news/pressreleases.rss" },
] as const;
const GDELT = "https://api.gdeltproject.org/api/v2/doc/doc?query=(bitcoin%20OR%20ethereum%20OR%20solana%20OR%20xrp%20OR%20cryptocurrency%20OR%20crypto%20OR%20binance%20OR%20gold%20OR%20silver%20OR%20oil%20OR%20Federal%20Reserve%20OR%20ECB)&mode=ArtList&maxrecords=75&format=json&sort=HybridRel&timespan=30min";

type FeedItem = { source: string; title: string; url: string; publishedAt: string | null; official: boolean };
function reply(body: unknown, status = 200) { return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } }); }
function bytes(value: string) { return new TextEncoder().encode(value); }
async function sha(value: string | Uint8Array) { const raw = typeof value === "string" ? bytes(value) : value; const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", raw)); return [...digest].map(x => x.toString(16).padStart(2, "0")).join(""); }
function cleanXml(value: string) { return value.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1").replace(/<[^>]+>/g, " ").replace(/&amp;/g, "&").replace(/&quot;/g, '"').replace(/&#39;|&apos;/g, "'").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/\s+/g, " ").trim(); }
function tag(block: string, name: string) { const match = block.match(new RegExp(`<${name}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${name}>`, "i")); return match ? cleanXml(match[1]) : ""; }
function link(block: string) { const plain = tag(block, "link"); if (plain.startsWith("http")) return plain; const href = block.match(/<link[^>]+href=["']([^"']+)["'][^>]*>/i); return href?.[1] ?? ""; }
function safeDate(value: string, observedAt: string) { if (!value) return null; const ms = Date.parse(value); if (!Number.isFinite(ms) || ms > Date.parse(observedAt) + 60_000) return null; return new Date(ms).toISOString(); }
function parseRss(source: string, xml: string, observedAt: string): FeedItem[] { const blocks = [...xml.matchAll(/<(item|entry)(?:\s[^>]*)?>([\s\S]*?)<\/\1>/gi)].map(m => m[2]); return blocks.slice(0, 40).map(block => { const title = tag(block, "title"); const url = link(block); const published = tag(block, "pubDate") || tag(block, "updated") || tag(block, "published") || tag(block, "dc:date"); return { source, title, url, publishedAt: safeDate(published, observedAt), official: true }; }).filter(row => row.title && row.url); }
function asset(title: string) { const t = title.toLowerCase(); if (/bitcoin|\bbtc\b/.test(t)) return "BTCUSDT"; if (/ethereum|\beth\b/.test(t)) return "ETHUSDT"; if (/solana|\bsol\b/.test(t)) return "SOLUSDT"; if (/\bxrp\b|ripple/.test(t)) return "XRPUSDT"; if (/binance coin|\bbnb\b/.test(t)) return "BNBUSDT"; if (/gold/.test(t)) return "GOLD"; if (/silver/.test(t)) return "SILVER"; if (/\boil\b|brent|wti/.test(t)) return "OIL"; return "MACRO"; }
function assetId(value: string) { return /USDT$/.test(value) ? `crypto:${value}` : `global:${value}`; }
function domain(value: string) { return ["GOLD", "SILVER", "OIL"].includes(value) ? "commodity" : value === "MACRO" ? "macro" : "news"; }
async function fetchText(url: string, timeout = 6000) { const r = await fetch(url, { headers: { accept: "application/rss+xml, application/xml, text/xml, */*", "user-agent": "Brian-2026-News-Eye/2.0" }, signal: AbortSignal.timeout(timeout) }); if (!r.ok) throw new Error(`${r.status} ${url}`); return await r.text(); }
async function fetchGdelt(): Promise<{ payload: unknown | null; error: string | null }> { try { const r = await fetch(GDELT, { headers: { accept: "application/json", "user-agent": "Brian-2026-News-Eye/2.0" }, signal: AbortSignal.timeout(5000) }); if (!r.ok) throw new Error(`GDELT ${r.status}`); return { payload: await r.json(), error: null }; } catch (error) { return { payload: null, error: String(error) }; } }
async function rawCapture(payload: unknown, observedAt: string) { const canonical = JSON.stringify(payload); const raw = bytes(canonical); const hash = await sha(raw); const zipped = gzip(raw, { level: 6 }); const path = `news/phase39_multisource/${observedAt.slice(0,10)}/${hash}.json.gz`; const up = await supabase.storage.from(BUCKET).upload(path, zipped, { contentType: "application/gzip", upsert: false, cacheControl: "31536000" }); if (up.error && !String(up.error.message).toLowerCase().match(/exist|duplicate/)) throw up.error; const id = await sha(`phase39_multisource_news|${observedAt}|${hash}`); const ins = await supabase.from("brian_raw_captures").insert({ capture_id:id, provider:"official_rss_plus_gdelt", record_type:"phase39_news_v2", observed_at:observedAt, captured_at:new Date().toISOString(), provenance_uri:"Federal Reserve RSS + ECB RSS + SEC RSS + optional GDELT DOC 2.0", payload_hash:hash, payload:{storage_bucket:BUCKET,storage_path:path,content_type:"application/json",content_encoding:"gzip",uncompressed_byte_length:raw.byteLength,compressed_byte_length:zipped.byteLength} }); if (ins.error) throw ins.error; return id; }
async function recordRun(start: string, status: string, observed: number, stored: number, degraded: string[], error?: unknown) { const end = new Date().toISOString(); const id = await sha(`${COLLECTOR_ID}|${start}|${end}|${status}`); await supabase.from("brian_collector_runs").insert({ run_id:id, collector_id:COLLECTOR_ID, started_at:start, finished_at:end, status, observed_records:observed, stored_records:stored, degraded_sources:degraded, error_class:error ? "COLLECTOR_ERROR" : null, error_message:error ? String(error).slice(0,1000) : null, evidence_class:EVIDENCE, shadow_only:true, live_execution:false }); }

Deno.serve(async req => {
  if (req.method !== "POST") return reply({ error: "POST required" }, 405);
  const started = new Date().toISOString();
  try {
    const officialResults = await Promise.all(OFFICIAL.map(async source => { try { return { source, xml: await fetchText(source.url), error: null as string | null }; } catch (error) { return { source, xml: null, error: String(error) }; } }));
    const successfulOfficial = officialResults.filter(row => row.xml);
    if (!successfulOfficial.length) throw new Error("all official RSS sources unavailable");
    const gdelt = await fetchGdelt();
    const observedAt = new Date().toISOString();
    const rawPayload = { schema_version:"brian.phase39.multisource-news.v2", observed_at:observedAt, official:officialResults.map(row => ({ id:row.source.id, url:row.source.url, xml:row.xml, error:row.error })), gdelt };
    const captureId = await rawCapture(rawPayload, observedAt);
    const items: FeedItem[] = [];
    for (const row of successfulOfficial) items.push(...parseRss(row.source.id, row.xml!, observedAt));
    if (gdelt.payload && Array.isArray((gdelt.payload as Record<string,unknown>).articles)) {
      for (const article of ((gdelt.payload as Record<string,unknown>).articles as Record<string,unknown>[]).slice(0,75)) {
        const title = String(article.title ?? "").trim(), url = String(article.url ?? "").trim(); if (!title || !url) continue;
        items.push({ source:"GDELT", title, url, publishedAt:null, official:false });
      }
    }
    const events: Record<string,unknown>[] = [];
    const grouped = new Map<string,{ eventIds:string[]; sources:Set<string>; officialCount:number; discoveryCount:number }>();
    for (const item of items) {
      const a = asset(item.title); const fingerprint = await sha(`${item.source}|${item.url}|${item.title}`); const eventId = await sha(`phase39-news|${fingerprint}`);
      events.push({ event_id:eventId, asset:a, event_kind:item.official ? "OFFICIAL_NEWS" : "NEWS_HEADLINE", source_kind:item.official ? "OFFICIAL_RSS" : "GDELT_DISCOVERY", source_id:item.source, published_at:item.publishedAt, first_observed_at:observedAt, captured_at:new Date().toISOString(), claim:item.title, direction:0, magnitude:item.official ? 0.5 : 0.2, trust_class:item.official ? "OFFICIAL_PRIMARY" : "UNVERIFIED_DISCOVERY", entity_confidence:item.official ? 0.95 : 0.35, content_fingerprint:fingerprint, corroboration_key:await sha(`${a}|${item.title.toLowerCase().replace(/[^a-z0-9]+/g," ").trim()}`), provenance_uri:item.url, pit_verified:true, raw_capture_id:captureId, metadata:{ provider:item.source, official_primary:item.official, headline_pressure_only:!item.official, not_source_truth:!item.official, directional_inference_disabled:true } });
      const g = grouped.get(a) ?? { eventIds:[], sources:new Set<string>(), officialCount:0, discoveryCount:0 }; g.eventIds.push(eventId); g.sources.add(item.source); if (item.official) g.officialCount++; else g.discoveryCount++; grouped.set(a,g);
    }
    if (events.length) { const ins = await supabase.from("brian_intel_events").upsert(events, { onConflict:"event_id", ignoreDuplicates:true }); if (ins.error) throw ins.error; }
    const observations: Record<string,unknown>[] = [];
    for (const [a,g] of grouped) {
      const confidence = Math.min(0.95, 0.35 + 0.15 * Math.min(3,g.officialCount) + 0.04 * Math.min(4,g.sources.size)); const strength = Math.min(1, (g.officialCount * 2 + g.discoveryCount * 0.25) / 8); const eye = await sha(`multisource-news-attention|${a}`); const id = await sha(`${eye}|${observedAt}|${g.eventIds.join(",")}`);
      observations.push({ observation_id:id, eye_id:eye, template_id:"multisource-news-attention", asset_id:assetId(a), market_domain:domain(a), sensor_family:"news_attention", horizon:"EVENT_DRIVEN", independent_group:"news_multisource", observed_at:observedAt, direction:0, strength, confidence, reliability:g.officialCount ? 0.8 : 0.35, available:true, source_ids:[captureId], reason:`${g.officialCount} official and ${g.discoveryCount} discovery headlines from ${g.sources.size} source families; directional inference intentionally disabled`, evidence_class:EVIDENCE, shadow_only:true, live_execution:false, metadata:{ event_ids:g.eventIds, source_families:[...g.sources], official_count:g.officialCount, discovery_count:g.discoveryCount, directional_inference_disabled:true } });
    }
    if (observations.length) { const ins = await supabase.from("brian_sensor_observations").insert(observations); if (ins.error) throw ins.error; }
    const degraded = officialResults.filter(row => row.error).map(row => row.source.id); if (gdelt.error) degraded.push("GDELT"); const status = degraded.length ? "DEGRADED" : "SUCCESS";
    await recordRun(started,status,items.length,events.length+observations.length,degraded);
    return reply({ status:status === "SUCCESS" ? "CAPTURED" : "DEGRADED", official_sources_ok:successfulOfficial.map(row=>row.source.id), degraded_sources:degraded, articles:items.length, events:events.length, sensor_observations:observations.length, directional_inference:false, learning_enabled:false, live_execution:false, shadow_only:true });
  } catch (error) { await recordRun(started,"FAILED",0,0,[],error); return reply({ status:"FAILED", error:String(error), shadow_only:true },500); }
});

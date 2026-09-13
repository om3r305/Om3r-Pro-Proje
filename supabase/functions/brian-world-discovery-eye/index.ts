import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { gzip } from "npm:pako@2.1.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireCronAuth } from "../_shared/cron_auth.ts";
import { EVOLUTION_EVIDENCE_CLASS } from "../_shared/evolution_contract.ts";

const URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(URL, SERVICE, { auth: { persistSession: false, autoRefreshToken: false } });
const COLLECTOR_ID = "brian-world-discovery-eye-v1";
const BUCKET = "brian-intelligence-raw";
const LEASE_SECONDS = 240;
const GDELT_BASE = "https://api.gdeltproject.org/api/v2/doc/doc";
const GOOGLE_NEWS_RSS = "https://news.google.com/rss/search";

type Theme = { id: string; query: string; fallbackQuery: string };
const THEMES: Theme[] = [
  {
    id: "technology_ai",
    query: '("artificial intelligence" OR NVIDIA OR Apple OR Microsoft OR Alphabet OR Google OR Amazon OR OpenAI OR TSMC OR semiconductor OR GPU OR HBM OR datacenter OR robotics)',
    fallbackQuery: 'AI OR NVIDIA OR OpenAI OR semiconductor OR GPU',
  },
  {
    id: "macro_rates",
    query: '("Federal Reserve" OR FOMC OR ECB OR inflation OR CPI OR PCE OR "Treasury yield" OR "interest rate" OR recession OR unemployment OR payrolls)',
    fallbackQuery: 'Federal Reserve OR ECB OR inflation OR interest rates',
  },
  {
    id: "commodities_energy",
    query: '(gold OR silver OR oil OR Brent OR WTI OR OPEC OR natural gas OR copper OR uranium OR "energy supply" OR refinery OR pipeline)',
    fallbackQuery: 'oil OR gold OR OPEC OR natural gas OR copper',
  },
  {
    id: "geopolitics",
    query: '(war OR sanctions OR ceasefire OR invasion OR missile OR tariff OR trade war OR shipping OR "supply chain" OR Red Sea OR Taiwan)',
    fallbackQuery: 'sanctions OR ceasefire OR tariffs OR Red Sea OR Taiwan',
  },
  {
    id: "corporate_product",
    query: '(earnings OR guidance OR "product launch" OR acquisition OR merger OR buyback OR supplier OR partnership OR "investor day" OR "capital expenditure")',
    fallbackQuery: 'earnings OR guidance OR acquisition OR product launch',
  },
  {
    id: "crypto_regulation",
    query: '(bitcoin OR ethereum OR cryptocurrency OR stablecoin OR Binance OR Coinbase OR "crypto ETF" OR "token unlock" OR airdrop OR "crypto regulation" OR SEC)',
    fallbackQuery: 'bitcoin OR ethereum OR Coinbase OR crypto regulation OR ETF',
  },
];

type Article = {
  url?: string;
  title?: string;
  seendate?: string;
  publishedAt?: string | null;
  domain?: string;
  language?: string;
  sourcecountry?: string;
};

type ThemeResult = {
  articles: Article[];
  captureId: string;
  observedAt: string;
  provider: "gdelt_doc2" | "google_news_rss";
  fallbackUsed: boolean;
  primaryError?: string;
};

function out(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });
}

function errorText(error: unknown): string {
  return error instanceof Error ? `${error.name}: ${error.message}` : String(error);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function sha256(value: string | Uint8Array): Promise<string> {
  const bytes = typeof value === "string" ? new TextEncoder().encode(value) : value;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

function parseSeen(value?: string): string | null {
  if (!value) return null;
  const m = value.match(/^(\d{4})(\d{2})(\d{2})T?(\d{2})(\d{2})(\d{2})Z?$/);
  if (!m) return null;
  const date = new Date(`${m[1]}-${m[2]}-${m[3]}T${m[4]}:${m[5]}:${m[6]}Z`);
  return Number.isFinite(date.getTime()) ? date.toISOString() : null;
}

function selectedThemes(now = new Date()): Theme[] {
  const slot = Math.floor(now.getUTCMinutes() / 10) % THEMES.length;
  return [THEMES[slot], THEMES[(slot + 3) % THEMES.length]];
}

function decodeXml(value: string): string {
  return value
    .replace(/^<!\[CDATA\[|\]\]>$/g, "")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

function xmlTag(block: string, tag: string): string {
  const m = block.match(new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)<\\/${tag}>`, "i"));
  return m ? decodeXml(m[1].trim()) : "";
}

function parseGoogleNews(xml: string): Article[] {
  const out: Article[] = [];
  const items = xml.matchAll(/<item>([\s\S]*?)<\/item>/gi);
  for (const match of items) {
    const block = match[1];
    const title = xmlTag(block, "title").replace(/<[^>]+>/g, "").trim();
    const url = xmlTag(block, "link").trim();
    const pubDate = xmlTag(block, "pubDate").trim();
    const sourceMatch = block.match(/<source(?:\s+url="([^"]+)")?[^>]*>([\s\S]*?)<\/source>/i);
    const sourceUrl = sourceMatch?.[1] || "";
    const sourceName = sourceMatch ? decodeXml(sourceMatch[2].replace(/<[^>]+>/g, "").trim()) : "google-news";
    let publishedAt: string | null = null;
    if (pubDate) {
      const d = new Date(pubDate);
      if (Number.isFinite(d.getTime())) publishedAt = d.toISOString();
    }
    if (!title || !url) continue;
    let domain = sourceName || "google-news";
    try {
      if (sourceUrl) domain = new URL(sourceUrl).hostname.replace(/^www\./, "") || domain;
    } catch (_) {}
    out.push({ url, title, publishedAt, domain, language: "en", sourcecountry: "US" });
    if (out.length >= 75) break;
  }
  return out;
}

async function rawCapture(
  theme: Theme,
  payload: unknown,
  observedAt: string,
  provider: string,
  provenanceUri: string,
  contentType = "application/json",
): Promise<string> {
  const raw = new TextEncoder().encode(typeof payload === "string" ? payload : JSON.stringify(payload));
  const hash = await sha256(raw);
  const compressed = gzip(raw, { level: 6 });
  const path = `world-discovery/${theme.id}/${observedAt.slice(0, 10)}/${hash}.gz`;
  const upload = await db.storage.from(BUCKET).upload(path, compressed, {
    contentType: "application/gzip",
    upsert: false,
    cacheControl: "31536000",
  });
  if (upload.error && !String(upload.error.message).toLowerCase().match(/exist|duplicate/)) throw upload.error;
  const captureId = await sha256(`${COLLECTOR_ID}|${theme.id}|${provider}|${observedAt}|${hash}`);
  const q = await db.from("brian_raw_captures").upsert({
    capture_id: captureId,
    provider,
    record_type: `world_discovery:${theme.id}`,
    observed_at: observedAt,
    captured_at: new Date().toISOString(),
    provenance_uri: provenanceUri,
    payload_hash: hash,
    payload: {
      storage_bucket: BUCKET,
      storage_path: path,
      content_type: contentType,
      content_encoding: "gzip",
      theme: theme.id,
      provider,
    },
  }, { onConflict: "capture_id", ignoreDuplicates: true });
  if (q.error) throw q.error;
  return captureId;
}

async function fetchGdelt(theme: Theme, query: string, timespan: string): Promise<Omit<ThemeResult, "fallbackUsed" | "primaryError">> {
  const params = new URLSearchParams({
    query,
    mode: "ArtList",
    maxrecords: "75",
    format: "json",
    sort: "HybridRel",
    timespan,
  });
  const requestUrl = `${GDELT_BASE}?${params}`;
  const response = await fetch(requestUrl, {
    headers: { accept: "application/json" },
    signal: AbortSignal.timeout(18000),
  });
  if (!response.ok) {
    const body = (await response.text().catch(() => "")).slice(0, 220).replace(/\s+/g, " ");
    throw new Error(`GDELT:${theme.id}:${response.status}${body ? `:${body}` : ""}`);
  }
  const text = await response.text();
  let payload: any;
  try {
    payload = JSON.parse(text);
  } catch (_) {
    throw new Error(`GDELT:${theme.id}:INVALID_JSON:${text.slice(0, 180).replace(/\s+/g, " ")}`);
  }
  const articles = Array.isArray(payload?.articles) ? payload.articles as Article[] : [];
  if (!articles.length) throw new Error(`GDELT:${theme.id}:EMPTY`);
  const observedAt = new Date().toISOString();
  const captureId = await rawCapture(theme, payload, observedAt, "gdelt_doc2", requestUrl);
  return { articles, captureId, observedAt, provider: "gdelt_doc2" };
}

async function fetchGoogleNews(theme: Theme): Promise<Omit<ThemeResult, "fallbackUsed" | "primaryError">> {
  const params = new URLSearchParams({
    q: `(${theme.fallbackQuery}) when:6h`,
    hl: "en-US",
    gl: "US",
    ceid: "US:en",
  });
  const requestUrl = `${GOOGLE_NEWS_RSS}?${params}`;
  const response = await fetch(requestUrl, {
    headers: {
      accept: "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
      "user-agent": "Mozilla/5.0 (compatible; BrianWorldDiscovery/1.1)",
    },
    signal: AbortSignal.timeout(14000),
  });
  if (!response.ok) throw new Error(`GOOGLE_NEWS:${theme.id}:${response.status}`);
  const xml = await response.text();
  const articles = parseGoogleNews(xml);
  if (!articles.length) throw new Error(`GOOGLE_NEWS:${theme.id}:EMPTY`);
  const observedAt = new Date().toISOString();
  const captureId = await rawCapture(theme, xml, observedAt, "google_news_rss", requestUrl, "application/rss+xml");
  return { articles, captureId, observedAt, provider: "google_news_rss" };
}

async function fetchTheme(theme: Theme): Promise<ThemeResult> {
  const primaryErrors: string[] = [];
  const attempts = [
    { query: theme.query, timespan: "2h" },
    { query: theme.fallbackQuery, timespan: "6h" },
  ];
  for (let i = 0; i < attempts.length; i++) {
    try {
      const result = await fetchGdelt(theme, attempts[i].query, attempts[i].timespan);
      return { ...result, fallbackUsed: i > 0, primaryError: primaryErrors.join(" | ") || undefined };
    } catch (error) {
      primaryErrors.push(errorText(error));
      if (i < attempts.length - 1) await sleep(700);
    }
  }
  const fallback = await fetchGoogleNews(theme);
  return { ...fallback, fallbackUsed: true, primaryError: primaryErrors.join(" | ") };
}

async function recordRun(
  startedAt: string,
  status: "SUCCESS" | "DEGRADED" | "FAILED" | "SKIPPED",
  observed: number,
  stored: number,
  degraded: string[],
  metadata: Record<string, unknown> = {},
  error?: unknown,
): Promise<void> {
  const finishedAt = new Date().toISOString();
  const runId = await sha256(`${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`);
  const q = await db.from("brian_collector_runs").insert({
    run_id: runId,
    collector_id: COLLECTOR_ID,
    started_at: startedAt,
    finished_at: finishedAt,
    status,
    observed_records: observed,
    stored_records: stored,
    degraded_sources: degraded,
    error_class: error ? "WORLD_DISCOVERY_ERROR" : null,
    error_message: error ? errorText(error).slice(0, 1000) : null,
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
    metadata: { directional_vote: false, discovery_only: true, theme_rotation: true, ...metadata },
  });
  if (q.error) console.error("world discovery receipt", q.error.message);
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return out({ error: "POST required" }, 405);
  const startedAt = new Date().toISOString();
  try {
    await requireCronAuth(req, db);
  } catch (error) {
    return out({ error: String(error), shadow_only: true, live_execution: false }, 401);
  }

  try {
    const lease = await withCollectorLease(db, COLLECTOR_ID, LEASE_SECONDS, async () => {
      const themes = selectedThemes();
      const events: Record<string, unknown>[] = [];
      const failedThemes: string[] = [];
      const providerFallbacks: string[] = [];
      const providerByTheme: Record<string, string> = {};
      const themeErrors: Record<string, string> = {};
      let observed = 0;

      for (const theme of themes) {
        try {
          const result = await fetchTheme(theme);
          providerByTheme[theme.id] = result.provider;
          if (result.fallbackUsed) providerFallbacks.push(theme.id);
          if (result.primaryError) themeErrors[theme.id] = result.primaryError.slice(0, 500);
          observed += result.articles.length;
          for (const article of result.articles) {
            const url = String(article.url ?? "").trim();
            const title = String(article.title ?? "").trim();
            if (!url || !title) continue;
            const publishedAt = article.publishedAt || parseSeen(article.seendate);
            const fingerprint = await sha256(`${url}|${title}`);
            const eventId = await sha256(`${COLLECTOR_ID}|${theme.id}|${fingerprint}`);
            events.push({
              event_id: eventId,
              asset: "GLOBAL_WORLD",
              event_kind: "WORLD_DISCOVERY",
              source_kind: result.provider === "gdelt_doc2" ? "GDELT_DISCOVERY" : "GOOGLE_NEWS_DISCOVERY",
              source_id: String(article.domain ?? result.provider),
              published_at: publishedAt,
              first_observed_at: result.observedAt,
              captured_at: new Date().toISOString(),
              claim: title,
              direction: 0,
              magnitude: 0.25,
              trust_class: "UNVERIFIED_DISCOVERY",
              entity_confidence: result.provider === "gdelt_doc2" ? 0.35 : 0.30,
              content_fingerprint: fingerprint,
              corroboration_key: await sha256(`${theme.id}|${title.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim()}`),
              provenance_uri: url,
              pit_verified: true,
              raw_capture_id: result.captureId,
              metadata: {
                world_theme: theme.id,
                provider: result.provider,
                gdelt_seen_at: result.provider === "gdelt_doc2" ? publishedAt : null,
                domain: article.domain,
                language: article.language,
                source_country: article.sourcecountry,
                discovery_only: true,
                directional_vote: false,
                requires_truth_engine: true,
                provider_fallback: result.fallbackUsed,
              },
            });
          }
        } catch (error) {
          failedThemes.push(theme.id);
          themeErrors[theme.id] = errorText(error).slice(0, 500);
          console.error("world discovery theme", theme.id, error);
        }
      }

      if (events.length) {
        const q = await db.from("brian_intel_events").upsert(events, { onConflict: "event_id", ignoreDuplicates: true });
        if (q.error) throw q.error;
      }

      const allFailed = failedThemes.length === themes.length;
      const status = allFailed ? "FAILED" : (failedThemes.length || providerFallbacks.length) ? "DEGRADED" : "SUCCESS";
      const diagnostics = { provider_fallbacks: providerFallbacks, provider_by_theme: providerByTheme, theme_errors: themeErrors };
      const terminalError = allFailed
        ? new Error(`all world discovery themes failed | ${Object.entries(themeErrors).map(([k, v]) => `${k}:${v}`).join(" || ")}`)
        : undefined;
      await recordRun(startedAt, status, observed, events.length, [...failedThemes, ...providerFallbacks], diagnostics, terminalError);
      return {
        status,
        collector_id: COLLECTOR_ID,
        themes: themes.map((row) => row.id),
        observed_articles: observed,
        stored_event_candidates: events.length,
        failed_themes: failedThemes,
        provider_fallbacks: providerFallbacks,
        provider_by_theme: providerByTheme,
        theme_errors: themeErrors,
        discovery_only: true,
        directional_vote: false,
        direct_alpha_influence: false,
        cloud_independent: true,
        shadow_only: true,
        live_execution: false,
      };
    });

    if (lease.contended) {
      await recordRun(startedAt, "SKIPPED", 0, 0, [], { lease_contended: true });
      return out({ status: "SKIPPED_LEASE_CONTENDED", collector_id: COLLECTOR_ID, shadow_only: true, live_execution: false });
    }
    return out(lease.value);
  } catch (error) {
    await recordRun(startedAt, "FAILED", 0, 0, [], { unhandled: true }, error);
    return out({ status: "FAILED", collector_id: COLLECTOR_ID, error: errorText(error), shadow_only: true, live_execution: false }, 500);
  }
});

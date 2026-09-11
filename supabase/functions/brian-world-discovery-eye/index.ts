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

type Theme = { id: string; query: string };
const THEMES: Theme[] = [
  {
    id: "technology_ai",
    query: '("artificial intelligence" OR NVIDIA OR Apple OR Microsoft OR Alphabet OR Google OR Amazon OR OpenAI OR TSMC OR semiconductor OR GPU OR HBM OR datacenter OR robotics)',
  },
  {
    id: "macro_rates",
    query: '("Federal Reserve" OR FOMC OR ECB OR inflation OR CPI OR PCE OR "Treasury yield" OR "interest rate" OR recession OR unemployment OR payrolls)',
  },
  {
    id: "commodities_energy",
    query: '(gold OR silver OR oil OR Brent OR WTI OR OPEC OR natural gas OR copper OR uranium OR "energy supply" OR refinery OR pipeline)',
  },
  {
    id: "geopolitics",
    query: '(war OR sanctions OR ceasefire OR invasion OR missile OR tariff OR trade war OR shipping OR "supply chain" OR Red Sea OR Taiwan)',
  },
  {
    id: "corporate_product",
    query: '(earnings OR guidance OR "product launch" OR acquisition OR merger OR buyback OR supplier OR partnership OR "investor day" OR "capital expenditure")',
  },
  {
    id: "crypto_regulation",
    query: '(bitcoin OR ethereum OR cryptocurrency OR stablecoin OR Binance OR Coinbase OR "crypto ETF" OR "token unlock" OR airdrop OR "crypto regulation" OR SEC)',
  },
];

type Article = { url?: string; title?: string; seendate?: string; domain?: string; language?: string; sourcecountry?: string };

function out(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status, headers: { "content-type": "application/json", "cache-control": "no-store" } });
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

async function rawCapture(theme: Theme, payload: unknown, observedAt: string): Promise<string> {
  const raw = new TextEncoder().encode(JSON.stringify(payload));
  const hash = await sha256(raw);
  const compressed = gzip(raw, { level: 6 });
  const path = `world-discovery/${theme.id}/${observedAt.slice(0, 10)}/${hash}.json.gz`;
  const upload = await db.storage.from(BUCKET).upload(path, compressed, {
    contentType: "application/gzip",
    upsert: false,
    cacheControl: "31536000",
  });
  if (upload.error && !String(upload.error.message).toLowerCase().match(/exist|duplicate/)) throw upload.error;
  const captureId = await sha256(`${COLLECTOR_ID}|${theme.id}|${observedAt}|${hash}`);
  const q = await db.from("brian_raw_captures").upsert({
    capture_id: captureId,
    provider: "gdelt_doc2",
    record_type: `world_discovery:${theme.id}`,
    observed_at: observedAt,
    captured_at: new Date().toISOString(),
    provenance_uri: GDELT_BASE,
    payload_hash: hash,
    payload: { storage_bucket: BUCKET, storage_path: path, content_type: "application/json", content_encoding: "gzip", theme: theme.id },
  }, { onConflict: "capture_id", ignoreDuplicates: true });
  if (q.error) throw q.error;
  return captureId;
}

async function fetchTheme(theme: Theme): Promise<{ articles: Article[]; captureId: string; observedAt: string }> {
  const params = new URLSearchParams({
    query: theme.query,
    mode: "ArtList",
    maxrecords: "75",
    format: "json",
    sort: "HybridRel",
    timespan: "45min",
  });
  const response = await fetch(`${GDELT_BASE}?${params}`, {
    headers: { accept: "application/json", "user-agent": "Brian-Evolution-World-Discovery/1.0" },
    signal: AbortSignal.timeout(14000),
  });
  if (!response.ok) throw new Error(`GDELT:${theme.id}:${response.status}`);
  const payload = await response.json();
  const observedAt = new Date().toISOString();
  const captureId = await rawCapture(theme, payload, observedAt);
  return { articles: Array.isArray(payload?.articles) ? payload.articles as Article[] : [], captureId, observedAt };
}

async function recordRun(startedAt: string, status: "SUCCESS" | "DEGRADED" | "FAILED" | "SKIPPED", observed: number, stored: number, degraded: string[], error?: unknown): Promise<void> {
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
    error_message: error ? String(error).slice(0, 1000) : null,
    evidence_class: EVOLUTION_EVIDENCE_CLASS,
    shadow_only: true,
    live_execution: false,
    metadata: { directional_vote: false, discovery_only: true, theme_rotation: true },
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
      const degraded: string[] = [];
      let observed = 0;

      for (const theme of themes) {
        try {
          const result = await fetchTheme(theme);
          observed += result.articles.length;
          for (const article of result.articles) {
            const url = String(article.url ?? "").trim();
            const title = String(article.title ?? "").trim();
            if (!url || !title) continue;
            const publishedAt = parseSeen(article.seendate);
            const fingerprint = await sha256(`${url}|${title}`);
            const eventId = await sha256(`${COLLECTOR_ID}|${theme.id}|${fingerprint}`);
            events.push({
              event_id: eventId,
              asset: "GLOBAL_WORLD",
              event_kind: "WORLD_DISCOVERY",
              source_kind: "GDELT_DISCOVERY",
              source_id: String(article.domain ?? "unknown"),
              published_at: publishedAt,
              first_observed_at: result.observedAt,
              captured_at: new Date().toISOString(),
              claim: title,
              direction: 0,
              magnitude: 0.25,
              trust_class: "UNVERIFIED_DISCOVERY",
              entity_confidence: 0.35,
              content_fingerprint: fingerprint,
              corroboration_key: await sha256(`${theme.id}|${title.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim()}`),
              provenance_uri: url,
              pit_verified: true,
              raw_capture_id: result.captureId,
              metadata: {
                world_theme: theme.id,
                gdelt_seen_at: publishedAt,
                domain: article.domain,
                language: article.language,
                source_country: article.sourcecountry,
                discovery_only: true,
                directional_vote: false,
                requires_truth_engine: true,
              },
            });
          }
        } catch (error) {
          degraded.push(theme.id);
          console.error("world discovery theme", theme.id, error);
        }
      }

      if (events.length) {
        const q = await db.from("brian_intel_events").upsert(events, { onConflict: "event_id", ignoreDuplicates: true });
        if (q.error) throw q.error;
      }
      const status = degraded.length === themes.length ? "FAILED" : degraded.length ? "DEGRADED" : "SUCCESS";
      await recordRun(startedAt, status, observed, events.length, degraded, status === "FAILED" ? new Error("all world discovery themes failed") : undefined);
      return {
        status,
        collector_id: COLLECTOR_ID,
        themes: themes.map((row) => row.id),
        observed_articles: observed,
        stored_event_candidates: events.length,
        degraded_themes: degraded,
        discovery_only: true,
        directional_vote: false,
        direct_alpha_influence: false,
        cloud_independent: true,
        shadow_only: true,
        live_execution: false,
      };
    });

    if (lease.contended) {
      await recordRun(startedAt, "SKIPPED", 0, 0, []);
      return out({ status: "SKIPPED_LEASE_CONTENDED", collector_id: COLLECTOR_ID, shadow_only: true, live_execution: false });
    }
    return out(lease.value);
  } catch (error) {
    await recordRun(startedAt, "FAILED", 0, 0, [], error);
    return out({ status: "FAILED", collector_id: COLLECTOR_ID, error: String(error), shadow_only: true, live_execution: false }, 500);
  }
});

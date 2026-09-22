import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { buildEligibleRows, CONFIG, diffEligibility, indexRows, scoreCandidates } from "./logic.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const RAW_BUCKET = "brian-intelligence-raw";
const PROVIDER = "binance_public";
const COLLECTOR_ID = "brian-universe-collector";
const MIN_INTERVAL_SECONDS = 780;
const LEASE_SECONDS = 420;

const EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo";
const TICKER_24H = "https://api.binance.com/api/v3/ticker/24hr";
const BOOK_TICKER = "https://api.binance.com/api/v3/ticker/bookTicker";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store" },
  });
}

function utf8(value: string): Uint8Array {
  return new TextEncoder().encode(value);
}

async function sha256Hex(value: string | Uint8Array): Promise<string> {
  const bytes = typeof value === "string" ? utf8(value) : value;
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchJson(url: string, required: boolean): Promise<{payload: unknown; observedAt: string; degraded: boolean}> {
  try {
    const response = await fetch(url, {
      method: "GET",
      headers: { "accept": "application/json", "user-agent": "Brian-2026-Shadow-Research/2.0" },
      signal: AbortSignal.timeout(8_000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    return { payload, observedAt: new Date().toISOString(), degraded: false };
  } catch (error) {
    if (required) throw error;
    return { payload: [], observedAt: new Date().toISOString(), degraded: true };
  }
}

async function persistRaw(recordType: string, payload: unknown, observedAt: string, provenanceUri: string): Promise<string> {
  const canonical = JSON.stringify(payload);
  const bytes = utf8(canonical);
  const payloadHash = await sha256Hex(bytes);
  const compressed = gzip(bytes, { level: 6 });
  const date = observedAt.slice(0, 10);
  const path = `${PROVIDER}/${recordType}/${date}/${payloadHash}.json.gz`;

  const upload = await supabase.storage.from(RAW_BUCKET).upload(path, compressed, {
    contentType: "application/gzip",
    upsert: false,
    cacheControl: "31536000",
  });
  if (upload.error) {
    const status = String((upload.error as {statusCode?: string | number}).statusCode ?? "");
    const msg = String(upload.error.message ?? "");
    if (status !== "409" && !msg.toLowerCase().includes("exist") && !msg.toLowerCase().includes("duplicate")) {
      throw upload.error;
    }
  }

  const capturedAt = new Date().toISOString();
  const captureId = await sha256Hex(`${PROVIDER}|${recordType}|${observedAt}|${payloadHash}`);
  const payloadPointer = {
    storage_bucket: RAW_BUCKET,
    storage_path: path,
    uncompressed_byte_length: bytes.byteLength,
    compressed_byte_length: compressed.byteLength,
    content_type: "application/json",
    content_encoding: "gzip",
  };

  const insert = await supabase.from("brian_raw_captures").insert({
    capture_id: captureId,
    provider: PROVIDER,
    record_type: recordType,
    observed_at: observedAt,
    captured_at: capturedAt,
    provenance_uri: provenanceUri,
    payload_hash: payloadHash,
    payload: payloadPointer,
  });
  if (insert.error) throw insert.error;
  return captureId;
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return jsonResponse({ error: "POST required" }, 405);

  try {
    const last = await supabase
      .from("brian_universe_snapshots")
      .select("observed_at,candidates")
      .order("observed_at", { ascending: false })
      .limit(1)
      .maybeSingle();
    if (last.error) throw last.error;
    if (last.data?.observed_at) {
      const ageSeconds = (Date.now() - Date.parse(last.data.observed_at)) / 1000;
      if (Number.isFinite(ageSeconds) && ageSeconds < MIN_INTERVAL_SECONDS) {
        return jsonResponse({ status: "SKIPPED_RATE_GUARD", age_seconds: Math.max(0, ageSeconds), shadow_only: true });
      }
    }

    const lease = await withCollectorLease(supabase, COLLECTOR_ID, LEASE_SECONDS, async () => {
    const exchange = await fetchJson(EXCHANGE_INFO, true);
    const ticker = await fetchJson(TICKER_24H, true);
    const book = await fetchJson(BOOK_TICKER, false);
    const snapshotObservedAt = new Date().toISOString();

    const exchangeObj = exchange.payload as Record<string, unknown>;
    if (!exchangeObj || typeof exchangeObj !== "object" || !Array.isArray(exchangeObj.symbols)) {
      throw new Error("invalid Binance exchangeInfo response");
    }
    const tickerMap = indexRows(ticker.payload);
    const bookMap = book.degraded ? new Map<string, Record<string, unknown>>() : indexRows(book.payload);

    const captureIds: string[] = [];
    captureIds.push(await persistRaw("exchange_info", exchange.payload, exchange.observedAt, EXCHANGE_INFO));
    captureIds.push(await persistRaw("ticker_24h", ticker.payload, ticker.observedAt, TICKER_24H));
    if (!book.degraded) captureIds.push(await persistRaw("book_ticker", book.payload, book.observedAt, BOOK_TICKER));

    const rows = buildEligibleRows(exchangeObj.symbols as unknown[], tickerMap, bookMap, CONFIG);
    const candidates = scoreCandidates(rows, CONFIG);
    const selected = candidates.slice(0, CONFIG.top_n);
    const eligibleSymbols = rows.map((r) => r.symbol).sort();

    const previousPayload = (last.data?.candidates && typeof last.data.candidates === "object")
      ? last.data.candidates as Record<string, unknown>
      : null;
    const previousEligible = previousPayload && Array.isArray(previousPayload.eligible_symbols)
      ? previousPayload.eligible_symbols.map(String)
      : null;
    const { comparable, newlyObserved, disappeared } = diffEligibility(previousEligible, eligibleSymbols);

    const snapshotPayload = {
      schema_version: "brian.universe-snapshot.v1",
      collector_version: "2",
      source: PROVIDER,
      config: CONFIG,
      eligible_symbols: eligibleSymbols,
      rejected_count: (exchangeObj.symbols as unknown[]).length - rows.length,
      degraded_sources: book.degraded ? ["book_ticker"] : [],
      comparable,
      newly_observed_symbols: newlyObserved,
      disappeared_symbols: disappeared,
      candidates: selected,
    };
    const snapshotId = await sha256Hex(`universe|${snapshotObservedAt}|${captureIds.join("|")}`);
    const snapshotInsert = await supabase.from("brian_universe_snapshots").insert({
      snapshot_id: snapshotId,
      provider: PROVIDER,
      observed_at: snapshotObservedAt,
      eligible_count: eligibleSymbols.length,
      candidates: snapshotPayload,
      raw_capture_ids: captureIds,
    });
    if (snapshotInsert.error) throw snapshotInsert.error;

      return jsonResponse({
        status: "CAPTURED",
        snapshot_id: snapshotId,
        observed_at: snapshotObservedAt,
        eligible_count: eligibleSymbols.length,
        top_candidates: selected.slice(0, 10).map((x) => ({ symbol: x.symbol, radar_score: x.radar_score, reasons: x.reasons })),
        newly_observed_symbols: newlyObserved,
        degraded_sources: snapshotPayload.degraded_sources,
        raw_encoding: "gzip",
        shadow_only: true,
      });
    });
    // Contended: another invocation already owns this collector's lease. No collector work has
    // run and no data has been written -- see supabase/functions/_shared/collector_lease.ts.
    if (lease.contended) return jsonResponse({ status: "SKIPPED_LEASE_CONTENDED", collector_id: COLLECTOR_ID, shadow_only: true });
    return lease.value!;
  } catch (error) {
    console.error("brian-universe-collector failed", error);
    return jsonResponse({ status: "FAILED_CLOSED", error: String(error instanceof Error ? error.message : error), shadow_only: true }, 500);
  }
});

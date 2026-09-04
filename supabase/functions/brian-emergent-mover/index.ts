import { createClient } from "npm:@supabase/supabase-js@2";
import { gzip } from "npm:pako@2.1.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import {
  buildEmergentMoverFrame,
  buildEmergentMoverReport,
  parseEmergentMoverFrame,
  type EmergentMarketRow,
  type EmergentMoverFrame,
} from "../_shared/emergent_mover.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const supabase = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const RAW_BUCKET = "brian-intelligence-raw";
const PROVIDER = "binance_public";
const COLLECTOR_ID = "brian-emergent-mover";
const MIN_INTERVAL_SECONDS = 240;
const MAX_UNIVERSE_AGE_MS = 30 * 60 * 1000;
const LEASE_SECONDS = 180;
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
  return [...digest].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function strictNumber(value: unknown, field: string): number {
  if (typeof value !== "string" && typeof value !== "number") throw new Error(`${field} must be numeric`);
  if (typeof value === "string" && !value.trim()) throw new Error(`${field} cannot be empty`);
  const out = Number(value);
  if (!Number.isFinite(out)) throw new Error(`${field} must be finite`);
  return out;
}

async function fetchJson(
  url: string,
  required: boolean,
): Promise<{ payload: unknown; observedAt: string; degraded: boolean }> {
  try {
    const response = await fetch(url, {
      headers: { "accept": "application/json", "user-agent": "Brian-2026-Emergent-Mover-Shadow/1.0" },
      signal: AbortSignal.timeout(8_000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    // Point-in-time observation is stamped after the provider response is received.
    return { payload, observedAt: new Date().toISOString(), degraded: false };
  } catch (error) {
    if (required) throw error;
    return { payload: [], observedAt: new Date().toISOString(), degraded: true };
  }
}

function indexRows(payload: unknown): Map<string, Record<string, unknown>> {
  if (!Array.isArray(payload)) throw new Error("expected Binance array response");
  const out = new Map<string, Record<string, unknown>>();
  for (const item of payload) {
    if (!item || typeof item !== "object") continue;
    const row = item as Record<string, unknown>;
    const symbol = String(row.symbol ?? "").trim().toUpperCase();
    if (symbol) out.set(symbol, row);
  }
  return out;
}

async function persistRaw(
  recordType: string,
  payload: unknown,
  observedAt: string,
  provenanceUri: string,
): Promise<string> {
  const canonical = JSON.stringify(payload);
  const bytes = utf8(canonical);
  const payloadHash = await sha256Hex(bytes);
  const compressed = gzip(bytes, { level: 6 });
  const path = `${PROVIDER}/${recordType}/${observedAt.slice(0, 10)}/${payloadHash}.json.gz`;

  const upload = await supabase.storage.from(RAW_BUCKET).upload(path, compressed, {
    contentType: "application/gzip",
    upsert: false,
    cacheControl: "31536000",
  });
  if (upload.error) {
    const status = String((upload.error as { statusCode?: string | number }).statusCode ?? "");
    const message = String(upload.error.message ?? "").toLowerCase();
    if (status !== "409" && !message.includes("exist") && !message.includes("duplicate")) {
      throw upload.error;
    }
  }

  const capturedAt = new Date().toISOString();
  const captureId = await sha256Hex(`${PROVIDER}|${recordType}|${observedAt}|${payloadHash}`);
  const insert = await supabase.from("brian_raw_captures").insert({
    capture_id: captureId,
    provider: PROVIDER,
    record_type: recordType,
    observed_at: observedAt,
    captured_at: capturedAt,
    provenance_uri: provenanceUri,
    payload_hash: payloadHash,
    payload: {
      storage_bucket: RAW_BUCKET,
      storage_path: path,
      uncompressed_byte_length: bytes.byteLength,
      compressed_byte_length: compressed.byteLength,
      content_type: "application/json",
      content_encoding: "gzip",
    },
  });
  if (insert.error) throw insert.error;
  return captureId;
}

function eligibleSymbolsFromSnapshot(payload: unknown): string[] {
  if (!payload || typeof payload !== "object") throw new Error("universe snapshot payload missing");
  const raw = payload as Record<string, unknown>;
  if (!Array.isArray(raw.eligible_symbols)) throw new Error("universe snapshot eligible_symbols missing");
  const symbols = [...new Set(raw.eligible_symbols.map((value) => String(value).trim().toUpperCase()).filter(Boolean))];
  symbols.sort();
  if (!symbols.length) throw new Error("universe snapshot has no eligible symbols");
  return symbols;
}

function marketRows(
  eligibleSymbols: string[],
  tickerPayload: unknown,
  bookPayload: unknown,
  bookDegraded: boolean,
): { rows: EmergentMarketRow[]; dropped: string[] } {
  const ticker = indexRows(tickerPayload);
  const books = bookDegraded ? new Map<string, Record<string, unknown>>() : indexRows(bookPayload);
  const rows: EmergentMarketRow[] = [];
  const dropped: string[] = [];

  for (const symbol of eligibleSymbols) {
    const tickerRow = ticker.get(symbol);
    if (!tickerRow) {
      dropped.push(symbol);
      continue;
    }
    try {
      const lastPrice = strictNumber(tickerRow.lastPrice, `${symbol}.lastPrice`);
      const quoteVolume = strictNumber(tickerRow.quoteVolume, `${symbol}.quoteVolume`);
      const trades = strictNumber(tickerRow.count, `${symbol}.count`);
      const change = strictNumber(tickerRow.priceChangePercent, `${symbol}.priceChangePercent`);
      const high = strictNumber(tickerRow.highPrice, `${symbol}.highPrice`);
      const low = strictNumber(tickerRow.lowPrice, `${symbol}.lowPrice`);
      if (!(lastPrice > 0) || quoteVolume < 0 || !Number.isInteger(trades) || trades < 0 || low < 0 || high < low) {
        throw new Error(`${symbol} market invariants failed`);
      }

      let spreadBps: number | null = null;
      const book = books.get(symbol);
      if (book) {
        try {
          const bid = strictNumber(book.bidPrice, `${symbol}.bidPrice`);
          const ask = strictNumber(book.askPrice, `${symbol}.askPrice`);
          if (bid > 0 && ask >= bid) {
            const mid = (bid + ask) / 2;
            spreadBps = 10_000 * (ask - bid) / mid;
          }
        } catch {
          // Per-symbol top-of-book corruption degrades spread context only. The ticker row remains
          // prospective evidence and no synthetic spread is fabricated.
          spreadBps = null;
        }
      }

      rows.push({
        symbol,
        last_price: lastPrice,
        quote_volume_24h: quoteVolume,
        trades_24h: trades,
        price_change_pct_24h: change,
        high_price_24h: high,
        low_price_24h: low,
        spread_bps: spreadBps,
      });
    } catch {
      dropped.push(symbol);
    }
  }
  return { rows, dropped: dropped.sort() };
}

Deno.serve(async (req: Request) => {
  if (req.method !== "POST") return jsonResponse({ error: "POST required" }, 405);

  try {
    const lease = await withCollectorLease(supabase, COLLECTOR_ID, LEASE_SECONDS, async () => {
      // Rate guard is intentionally INSIDE the lease. That closes the race where two invocations
      // both observe an old frame before one of them obtains and releases the lease.
      const latestFrame = await supabase
        .from("brian_emergent_mover_frames")
        .select("observed_at,state")
        .order("observed_at", { ascending: false })
        .limit(1)
        .maybeSingle();
      if (latestFrame.error) throw latestFrame.error;
      if (latestFrame.data?.observed_at) {
        const ageSeconds = (Date.now() - Date.parse(latestFrame.data.observed_at)) / 1000;
        if (Number.isFinite(ageSeconds) && ageSeconds < MIN_INTERVAL_SECONDS) {
          return jsonResponse({
            status: "SKIPPED_RATE_GUARD",
            age_seconds: Math.max(0, ageSeconds),
            live_execution: false,
            shadow_only: true,
          });
        }
      }

      const universe = await supabase
        .from("brian_universe_snapshots")
        .select("snapshot_id,observed_at,candidates")
        .order("observed_at", { ascending: false })
        .limit(1)
        .maybeSingle();
      if (universe.error) throw universe.error;
      if (!universe.data?.snapshot_id || !universe.data?.observed_at) {
        throw new Error("no universe snapshot available");
      }
      const universeAgeMs = Date.now() - Date.parse(universe.data.observed_at);
      if (!Number.isFinite(universeAgeMs) || universeAgeMs < 0 || universeAgeMs > MAX_UNIVERSE_AGE_MS) {
        return jsonResponse({
          status: "SKIPPED_STALE_UNIVERSE",
          universe_observed_at: universe.data.observed_at,
          live_execution: false,
          shadow_only: true,
        });
      }
      const eligibleSymbols = eligibleSymbolsFromSnapshot(universe.data.candidates);

      const ticker = await fetchJson(TICKER_24H, true);
      const book = await fetchJson(BOOK_TICKER, false);
      const observedAt = new Date().toISOString();

      // Persist provider truth before the scout interprets or ranks it.
      const rawCaptureIds = [
        await persistRaw("emergent_ticker_24h", ticker.payload, ticker.observedAt, TICKER_24H),
      ];
      if (!book.degraded) {
        rawCaptureIds.push(await persistRaw("emergent_book_ticker", book.payload, book.observedAt, BOOK_TICKER));
      }

      const mapped = marketRows(eligibleSymbols, ticker.payload, book.payload, book.degraded);
      if (!mapped.rows.length) throw new Error("no valid emergent mover market rows");
      const currentFrame = buildEmergentMoverFrame(mapped.rows, { observed_at: observedAt, source: PROVIDER });

      let previousFrame: EmergentMoverFrame | null = null;
      if (latestFrame.data?.state) {
        // Internal persistence is still treated as untrusted input: schema/lineage/ranges are
        // revalidated before comparison instead of cast-through.
        previousFrame = parseEmergentMoverFrame(latestFrame.data.state);
      }
      const report = buildEmergentMoverReport(previousFrame, currentFrame);
      const frameId = await sha256Hex(
        `emergent|${universe.data.snapshot_id}|${observedAt}|${rawCaptureIds.join("|")}`,
      );
      const insert = await supabase.from("brian_emergent_mover_frames").insert({
        frame_id: frameId,
        universe_snapshot_id: universe.data.snapshot_id,
        provider: PROVIDER,
        observed_at: observedAt,
        baseline_observed_at: report.baseline_observed_at,
        comparison_age_ms: report.comparison_age_ms,
        comparable: report.comparable,
        eligible_count: currentFrame.rows.length,
        dropped_symbol_count: mapped.dropped.length,
        degraded_sources: book.degraded ? ["book_ticker"] : [],
        raw_capture_ids: rawCaptureIds,
        state: currentFrame,
        report,
        evidence_class: "PROSPECTIVE_DEVELOPMENT_SHADOW",
        shadow_only: true,
        live_execution: false,
      });
      if (insert.error) throw insert.error;

      return jsonResponse({
        status: "CAPTURED",
        frame_id: frameId,
        observed_at: observedAt,
        universe_snapshot_id: universe.data.snapshot_id,
        comparable: report.comparable,
        comparison_issue: report.comparison_issue,
        compared_symbol_count: report.compared_symbol_count,
        newly_observed_symbols: report.newly_observed_symbols,
        disappeared_symbols: report.disappeared_symbols,
        top_emergent_movers: report.candidates.slice(0, 10).map((candidate) => ({
          rank: candidate.rank,
          symbol: candidate.symbol,
          attention_score: candidate.attention_score,
          observed_change_direction: candidate.observed_change_direction,
          reasons: candidate.reasons,
        })),
        degraded_sources: book.degraded ? ["book_ticker"] : [],
        dropped_symbols: mapped.dropped,
        measurement_notes: report.measurement_notes,
        evidence_class: "PROSPECTIVE_DEVELOPMENT_SHADOW",
        learning_enabled: false,
        live_execution: false,
        shadow_only: true,
      });
    });

    if (lease.contended) {
      return jsonResponse({
        status: "SKIPPED_LEASE_CONTENDED",
        collector_id: COLLECTOR_ID,
        live_execution: false,
        shadow_only: true,
      });
    }
    return lease.value!;
  } catch (error) {
    console.error("brian-emergent-mover failed", error);
    return jsonResponse({
      status: "FAILED_CLOSED",
      error: String(error instanceof Error ? error.message : error),
      learning_enabled: false,
      live_execution: false,
      shadow_only: true,
    }, 500);
  }
});

import { createClient } from "npm:@supabase/supabase-js@2.116.0";
import { withCollectorLease } from "../_shared/collector_lease.ts";
import { requireRealtimeInternal } from "../_shared/realtime_internal_auth.ts";
import {
  compileL2Cost,
  type DynamicCostQuote,
} from "../_shared/dynamic_cost.ts";
import { parseBinanceDepthSnapshotRaw } from "../_shared/binance_l2_wire.ts";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL")!;
const SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
const db = createClient(SUPABASE_URL, SERVICE_ROLE_KEY, {
  auth: { persistSession: false, autoRefreshToken: false },
});

const VERSION = "brian.readiness-cost-sampler.v1";
const COLLECTOR_ID = "brian-readiness-cost-sampler-v1";
const EVIDENCE = "PROSPECTIVE_DEVELOPMENT_SHADOW";
const ASSETS = [
  "crypto:BTCUSDT",
  "crypto:ETHUSDT",
  "crypto:SOLUSDT",
  "crypto:XRPUSDT",
] as const;
const NOTIONAL_USD = 20;
const FEE_BPS = 10;
const L2_DEPTH_LIMIT = 100;
const LEASE_SECONDS = 45;

type Side = "BUY" | "SELL";

type Sample = {
  assetId: string;
  symbol: string;
  observedAt: string;
  sourceId: string;
  lastUpdateId: string;
  fetchedAt: string;
  chosenSide: Side;
  chosen: DynamicCostQuote;
  buyCostBps: number;
  sellCostBps: number;
};

function out(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
    },
  });
}

function errText(error: unknown): string {
  if (error instanceof Error) return `${error.name}: ${error.message}`;
  if (error && typeof error === "object") {
    const row = error as Record<string, unknown>;
    const fields = ["code", "message", "details", "hint", "status", "statusText"]
      .filter((key) => row[key] != null)
      .map((key) => `${key}=${String(row[key])}`);
    if (fields.length) return fields.join(" | ");
  }
  return String(error);
}

async function sha256Hex(value: string): Promise<string> {
  const digest = new Uint8Array(
    await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value)),
  );
  return [...digest].map((b) => b.toString(16).padStart(2, "0")).join("");
}

async function sampleAsset(assetId: string): Promise<Sample> {
  const symbol = assetId.includes(":") ? assetId.split(":", 2)[1] : assetId;
  if (!/^[A-Z0-9]+USDT$/.test(symbol)) {
    throw new Error(`unsupported readiness cost asset ${assetId}`);
  }

  const response = await fetch(
    `https://api.binance.com/api/v3/depth?symbol=${encodeURIComponent(symbol)}&limit=${L2_DEPTH_LIMIT}`,
    {
      headers: {
        accept: "application/json",
        "user-agent": "Brian-Readiness-Cost-Sampler/1.0",
      },
      signal: AbortSignal.timeout(4_000),
    },
  );
  if (!response.ok) {
    throw new Error(`Binance depth HTTP ${response.status} for ${symbol}`);
  }

  const raw = await response.text();
  const snapshot = parseBinanceDepthSnapshotRaw(raw);
  const bids = snapshot.bids.map(([price, size]) => ({ price, size }));
  const asks = snapshot.asks.map(([price, size]) => ({ price, size }));

  const buy = compileL2Cost({
    side: "BUY",
    notionalUsd: NOTIONAL_USD,
    feeBps: FEE_BPS,
    bids,
    asks,
  });
  const sell = compileL2Cost({
    side: "SELL",
    notionalUsd: NOTIONAL_USD,
    feeBps: FEE_BPS,
    bids,
    asks,
  });

  if (!buy.fillable || !sell.fillable) {
    throw new Error(`visible L2 cannot fill readiness notional for ${symbol}`);
  }

  const chosen = buy.estimatedRoundTripCostBps >= sell.estimatedRoundTripCostBps
    ? buy
    : sell;
  const chosenSide: Side = chosen.side;
  const fetchedAt = new Date().toISOString();

  return {
    assetId,
    symbol,
    observedAt: fetchedAt,
    sourceId: `binance_public_rest_depth:${symbol}:${snapshot.lastUpdateId}`,
    lastUpdateId: String(snapshot.lastUpdateId),
    fetchedAt,
    chosenSide,
    chosen,
    buyCostBps: buy.estimatedRoundTripCostBps,
    sellCostBps: sell.estimatedRoundTripCostBps,
  };
}

async function recordRun(
  startedAt: string,
  status: "SUCCESS" | "DEGRADED" | "FAILED",
  stored: number,
  degraded: string[],
  error?: unknown,
) {
  const finishedAt = new Date().toISOString();
  const runId = await sha256Hex(
    `${COLLECTOR_ID}|${startedAt}|${finishedAt}|${status}`,
  );
  const q = await db.from("brian_collector_runs").insert({
    run_id: runId,
    collector_id: COLLECTOR_ID,
    started_at: startedAt,
    finished_at: finishedAt,
    status,
    observed_records: ASSETS.length,
    stored_records: stored,
    degraded_sources: degraded,
    error_class: error ? "READINESS_COST_SAMPLER_ERROR" : null,
    error_message: error ? errText(error).slice(0, 1500) : null,
    evidence_class: EVIDENCE,
    shadow_only: true,
    live_execution: false,
    metadata: {
      sampler_version: VERSION,
      assets: ASSETS,
      notional_usd: NOTIONAL_USD,
      source: "binance_public_rest_depth_snapshot",
      purpose: "PHASE115_DYNAMIC_COST_READINESS",
      selection: "MAX_BUY_SELL_ROUND_TRIP_COST",
    },
  });
  if (q.error) console.error("readiness cost run receipt", q.error.message);
}

Deno.serve(async (req: Request) => {
  if (req.method === "GET") {
    return out({
      status: "OK",
      version: VERSION,
      assets: ASSETS,
      notional_usd: NOTIONAL_USD,
      shadow_only: true,
      live_execution: false,
    });
  }
  if (req.method !== "POST") return out({ error: "POST required" }, 405);

  try {
    await requireRealtimeInternal(req);
  } catch {
    return out({ status: "UNAUTHORIZED", version: VERSION }, 401);
  }

  const startedAt = new Date().toISOString();

  try {
    const lease = await withCollectorLease(
      db,
      COLLECTOR_ID,
      LEASE_SECONDS,
      async () => {
        const settled = await Promise.allSettled(
          ASSETS.map((asset) => sampleAsset(asset)),
        );
        const samples: Sample[] = [];
        const degraded: string[] = [];

        for (let i = 0; i < settled.length; i++) {
          const result = settled[i];
          if (result.status === "fulfilled") {
            samples.push(result.value);
          } else {
            degraded.push(
              `${ASSETS[i]}:${errText(result.reason).slice(0, 320)}`,
            );
          }
        }

        const rows = await Promise.all(samples.map(async (sample) => {
          const q = sample.chosen;
          const quoteId = await sha256Hex(
            `${VERSION}|cost|${sample.assetId}|${sample.observedAt}|${sample.chosenSide}|${sample.lastUpdateId}`,
          );
          return {
            quote_id: quoteId,
            compiler_version: VERSION,
            asset_id: sample.assetId,
            observed_at: sample.observedAt,
            side: q.side,
            requested_notional_usd: q.requestedNotionalUsd,
            filled_notional_usd: q.filledNotionalUsd,
            fill_ratio: q.fillRatio,
            fillable: q.fillable,
            fee_bps: q.feeBps,
            spread_bps: q.spreadBps,
            depth_slippage_bps: q.depthSlippageBps,
            one_way_cost_bps: q.oneWayCostBps,
            estimated_round_trip_cost_bps: q.estimatedRoundTripCostBps,
            quality: q.quality,
            source_ids: [sample.sourceId],
            reason:
              "conservative direction-agnostic readiness cost: max observed BUY/SELL round-trip cost on one public L2 snapshot",
            metadata: {
              source: "binance_public_rest_depth_snapshot",
              purpose: "PHASE115_DYNAMIC_COST_READINESS",
              selection: "MAX_BUY_SELL_ROUND_TRIP_COST",
              buy_round_trip_cost_bps: sample.buyCostBps,
              sell_round_trip_cost_bps: sample.sellCostBps,
              selected_side: sample.chosenSide,
              last_update_id: sample.lastUpdateId,
              fetched_at: sample.fetchedAt,
              depth_limit: L2_DEPTH_LIMIT,
              notional_usd: NOTIONAL_USD,
            },
            evidence_class: EVIDENCE,
            shadow_only: true,
            live_execution: false,
          };
        }));

        if (rows.length) {
          const ins = await db.from("brian_dynamic_cost_quotes").insert(rows);
          if (ins.error) {
            throw new Error(
              `brian_dynamic_cost_quotes insert failed: ${ins.error.message}`,
            );
          }
        }

        return { rows, degraded };
      },
    );

    if (lease.contended) {
      return out({
        status: "SKIPPED_BUSY",
        version: VERSION,
        shadow_only: true,
        live_execution: false,
      });
    }

    const rows = lease.value?.rows ?? [];
    const degraded = lease.value?.degraded ?? [];
    const status = degraded.length ? "DEGRADED" : "SUCCESS";
    await recordRun(startedAt, status, rows.length, degraded);

    return out({
      status,
      version: VERSION,
      observed_at: new Date().toISOString(),
      assets_requested: ASSETS.length,
      quotes_stored: rows.length,
      degraded_sources: degraded,
      notional_usd: NOTIONAL_USD,
      shadow_only: true,
      live_execution: false,
    }, degraded.length ? 207 : 200);
  } catch (error) {
    await recordRun(startedAt, "FAILED", 0, [], error);
    return out({
      status: "FAILED_CLOSED",
      version: VERSION,
      error: errText(error).slice(0, 1200),
      shadow_only: true,
      live_execution: false,
    }, 500);
  }
});

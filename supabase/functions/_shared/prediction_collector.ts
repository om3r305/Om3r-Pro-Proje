// Prediction Intelligence -- Phase 1b bounded collection cycle (pure; no database access).
//
// One cycle = a small, bounded set of public read requests that produces a batch of rows for
// brian_record_prediction_batch_v1 (see 202610190001_brian_prediction_intelligence.sql). The edge
// function owns the lease, operator switch and persistence; this module only decides what to
// request and how to turn the answers into rows, so every rule here is testable offline.
//
// Bounds per cycle: one events page per configured tag (<= 10 tags), at most `maxMarkets`
// markets kept, two identity requests, and at most `maxActivityPages` activity pages. There is no
// high-frequency loop and no websocket here; cadence is the scheduler's business.

import {
  dataActivityUrl,
  fetchPredictionJson,
  flattenGammaEvents,
  gammaEventsKeysetUrl,
  partitionWalletActivity,
  PREDICTION_CONTRACT_VERSION,
  type PredictionFetchOptions,
  type PredictionFetchResult,
  type PredictionMarket,
  type PredictionWalletTrade,
  verifyProfileIdentity,
} from "./prediction_market.ts";

export const PREDICTION_COLLECTOR_VERSION = "brian.prediction-collector.v0";

// Must match the per-call bounds in brian_record_prediction_batch_v1.
export const MAX_BATCH_MARKETS = 300;
export const MAX_BATCH_TRADES = 1000;

export interface PredictionCollectorConfig {
  tagSlugs: string[];
  maxMarkets: number;
  minLiquidity: number;
  eventsPerTag: number;
  walletTracking: boolean;
  trackedName: string;
  backfillHours: number;
  activityPageSize: number;
  maxActivityPages: number;
}

export const DEFAULT_COLLECTOR_CONFIG: PredictionCollectorConfig = Object.freeze({
  tagSlugs: ["fed", "fomc", "cpi", "inflation", "economy", "bitcoin"],
  maxMarkets: 60,
  minLiquidity: 10_000,
  eventsPerTag: 20,
  walletTracking: false,
  trackedName: "NIULAI4",
  backfillHours: 24,
  activityPageSize: 500,
  maxActivityPages: 2,
});

export interface KnownWallet {
  wallet: string;
  watermark: string | null;
}

export interface MarketRow {
  market_key: string;
  source: string;
  market_id: string;
  condition_id: string | null;
  event_id: string | null;
  event_title: string | null;
  question: string;
  slug: string | null;
  end_date: string | null;
  outcomes: { label: string; clob_token_id: string | null }[];
  first_seen_at: string;
}

export interface SnapshotRow {
  snapshot_id: string;
  market_key: string;
  source_updated_at: string | null;
  received_at: string;
  outcomes: { label: string; implied_probability: number | null; clob_token_id: string | null }[];
  best_bid: number | null;
  best_ask: number | null;
  spread: number | null;
  liquidity: number | null;
  volume_24h: number | null;
  active: boolean;
  closed: boolean;
  source_freshness_seconds: number | null;
  model_version: string;
}

export interface TradeRow {
  idempotency_key: string;
  wallet: string;
  condition_id: string;
  asset_id: string;
  market_slug: string | null;
  title: string | null;
  outcome: string | null;
  side: "BUY" | "SELL";
  price: number;
  size: number;
  usdc_size: number | null;
  transaction_hash: string;
  traded_at: string;
  received_at: string;
  model_version: string;
}

export type HealthStatus = "OK" | "DEGRADED" | "BLOCKED" | "RATE_LIMITED" | "ERROR" | "SKIPPED";

export interface HealthRow {
  health_id: string;
  source: string;
  status: HealthStatus;
  http_status: number | null;
  attempts: number | null;
  latency_ms: number | null;
  records_seen: number;
  records_stored: number;
  detail: string | null;
  observed_at: string;
  metadata: Record<string, unknown>;
}

export interface WalletUpdate {
  wallet: string;
  displayName: string;
  status: "VERIFIED" | "UNVERIFIED" | "AMBIGUOUS";
  evidence: string[];
  // Only advanced after the batch carrying these trades has been persisted (edge function rule).
  watermark: string | null;
}

export interface PredictionCycleBatch {
  version: typeof PREDICTION_COLLECTOR_VERSION;
  markets: MarketRow[];
  snapshots: SnapshotRow[];
  trades: TradeRow[];
  health: HealthRow[];
  wallet: WalletUpdate | null;
  summary: {
    tagsRequested: number;
    marketsSeen: number;
    marketsKept: number;
    marketsRejected: number;
    requests: number;
    tradesSeen: number;
    activityTruncated: boolean;
    identity: string;
  };
}

export function marketKey(market: Pick<PredictionMarket, "marketId">): string {
  return `polymarket:${market.marketId}`;
}

export function toMarketRow(m: PredictionMarket): MarketRow {
  return {
    market_key: marketKey(m),
    source: m.source,
    market_id: m.marketId,
    condition_id: m.conditionId,
    event_id: m.eventId,
    event_title: m.eventTitle,
    question: m.question,
    slug: m.slug,
    end_date: m.endDate,
    outcomes: m.outcomes.map((o) => ({ label: o.label, clob_token_id: o.clobTokenId })),
    first_seen_at: m.receivedAt,
  };
}

// The snapshot key pins the source's own update time, so polling an unchanged market does not
// create duplicate rows. When the source gives no update time, the key falls back to the minute
// of receipt -- explicitly marked so the two are never confused.
export function toSnapshotRow(m: PredictionMarket): SnapshotRow {
  const key = marketKey(m);
  const sourceMs = m.sourceUpdatedAt === null ? null : Date.parse(m.sourceUpdatedAt);
  const receivedMs = Date.parse(m.receivedAt);
  return {
    snapshot_id: m.sourceUpdatedAt !== null ? `${key}|src:${m.sourceUpdatedAt}` : `${key}|rx:${m.receivedAt.slice(0, 16)}`,
    market_key: key,
    source_updated_at: m.sourceUpdatedAt,
    received_at: m.receivedAt,
    outcomes: m.outcomes.map((o) => ({
      label: o.label,
      implied_probability: o.impliedProbability,
      clob_token_id: o.clobTokenId,
    })),
    best_bid: m.bestBid,
    best_ask: m.bestAsk,
    spread: m.spread !== null && m.spread >= 0 ? m.spread : null,
    liquidity: m.liquidity !== null && m.liquidity >= 0 ? m.liquidity : null,
    volume_24h: m.volume24h !== null && m.volume24h >= 0 ? m.volume24h : null,
    active: m.active,
    closed: m.closed,
    source_freshness_seconds: sourceMs === null || !Number.isFinite(receivedMs)
      ? null
      : Math.round((receivedMs - sourceMs) / 1000),
    model_version: PREDICTION_CONTRACT_VERSION,
  };
}

export function toTradeRow(t: PredictionWalletTrade): TradeRow {
  return {
    idempotency_key: t.idempotencyKey,
    wallet: t.wallet,
    condition_id: t.conditionId,
    asset_id: t.assetId,
    market_slug: t.marketSlug,
    title: t.title,
    outcome: t.outcome,
    side: t.side,
    price: t.price,
    size: t.size,
    usdc_size: t.usdcSize !== null && t.usdcSize >= 0 ? t.usdcSize : null,
    transaction_hash: t.transactionHash,
    traded_at: t.tradedAt,
    received_at: t.receivedAt,
    model_version: PREDICTION_CONTRACT_VERSION,
  };
}

// Selection is deterministic and conservative: open, order-book-enabled markets whose liquidity is
// KNOWN and above the floor (unknown liquidity never passes), de-duplicated, highest 24h volume
// first (unknown volume last), then market id as a stable tie-breaker.
export function selectMarkets(markets: PredictionMarket[], maxMarkets: number, minLiquidity: number): PredictionMarket[] {
  const byId = new Map<string, PredictionMarket>();
  for (const m of markets) {
    if (!m.active || m.closed || !m.orderBookEnabled) continue;
    if (m.liquidity === null || m.liquidity < minLiquidity) continue;
    if (!byId.has(m.marketId)) byId.set(m.marketId, m);
  }
  return [...byId.values()]
    .sort((a, b) => {
      const va = a.volume24h ?? -1;
      const vb = b.volume24h ?? -1;
      if (vb !== va) return vb - va;
      return a.marketId < b.marketId ? -1 : a.marketId > b.marketId ? 1 : 0;
    })
    .slice(0, Math.max(0, Math.trunc(maxMarkets)));
}

function healthFrom(
  source: string,
  result: PredictionFetchResult,
  startedMs: number,
  seen: number,
  extra: { status?: HealthStatus; detail?: string | null; metadata?: Record<string, unknown> } = {},
): HealthRow {
  const status: HealthStatus = extra.status ?? (result.status === "OK" ? "OK" : result.status);
  return {
    health_id: `${source}|${result.receivedAt}`,
    source,
    status,
    http_status: result.httpStatus,
    attempts: result.attempts,
    latency_ms: Math.max(0, Date.parse(result.receivedAt) - startedMs),
    records_seen: seen,
    // Storage counts are only known after persistence; the RPC reports them separately.
    records_stored: 0,
    detail: extra.detail ?? result.detail,
    observed_at: result.receivedAt,
    metadata: { collector_version: PREDICTION_COLLECTOR_VERSION, ...(extra.metadata ?? {}) },
  };
}

export async function collectPredictionCycle(
  config: PredictionCollectorConfig,
  known: KnownWallet | null,
  options: PredictionFetchOptions = {},
): Promise<PredictionCycleBatch> {
  const now = options.now ?? (() => Date.now());
  const tags = [...new Set(config.tagSlugs)].slice(0, 10);
  const health: HealthRow[] = [];
  const discovered: PredictionMarket[] = [];
  let rejectedMarkets = 0;
  let requests = 0;

  for (const tag of tags) {
    const started = now();
    requests++;
    const result = await fetchPredictionJson(
      gammaEventsKeysetUrl({ tagSlug: tag, limit: config.eventsPerTag, active: true, closed: false }),
      options,
    );
    if (result.status !== "OK") {
      health.push(healthFrom(`polymarket-gamma-events:${tag}`, result, started, 0));
      continue;
    }
    const flat = flattenGammaEvents(result.body, result.receivedAt);
    discovered.push(...flat.markets);
    rejectedMarkets += flat.rejected.length;
    health.push(
      healthFrom(`polymarket-gamma-events:${tag}`, result, started, flat.markets.length, {
        status: flat.rejected.length > 0 ? "DEGRADED" : "OK",
        detail: flat.rejected.length > 0 ? `${flat.rejected.length} markets failed normalization` : null,
        metadata: { rejected: flat.rejected.slice(0, 10), has_more: flat.nextCursor !== null },
      }),
    );
  }

  const kept = selectMarkets(discovered, Math.min(config.maxMarkets, MAX_BATCH_MARKETS), config.minLiquidity);

  let trades: TradeRow[] = [];
  let wallet: WalletUpdate | null = null;
  let identity = "SKIPPED";
  let truncated = false;
  let tradesSeen = 0;

  if (config.walletTracking) {
    const verdict = await verifyIdentity(config.trackedName, options, health, now);
    requests += verdict.requests;
    identity = verdict.status;
    if (verdict.status === "VERIFIED" && verdict.wallet) {
      const activity = await collectActivity(config, verdict.wallet, known, options, health, now);
      requests += activity.requests;
      trades = activity.trades;
      truncated = activity.truncated;
      tradesSeen = activity.seen;
      wallet = {
        wallet: verdict.wallet,
        displayName: config.trackedName,
        status: "VERIFIED",
        evidence: verdict.evidence,
        watermark: activity.watermark,
      };
    } else if (verdict.status !== "UNREACHABLE" && known !== null) {
      // The previously known wallet no longer verifies: downgrade it so its trades stop being
      // stored. A source outage (UNREACHABLE) is not evidence against the identity.
      wallet = {
        wallet: known.wallet,
        displayName: config.trackedName,
        status: verdict.status === "AMBIGUOUS" ? "AMBIGUOUS" : "UNVERIFIED",
        evidence: [verdict.reason],
        watermark: null,
      };
    }
  }

  return {
    version: PREDICTION_COLLECTOR_VERSION,
    markets: kept.map(toMarketRow),
    snapshots: kept.map(toSnapshotRow),
    trades,
    // Two probes can share a millisecond; the cycle index keeps every health row distinct while a
    // retried persist of the same batch still collapses onto the same keys.
    health: health.map((h, i) => ({ ...h, health_id: `${h.source}|${h.observed_at}|${i}` })),
    wallet,
    summary: {
      tagsRequested: tags.length,
      marketsSeen: discovered.length,
      marketsKept: kept.length,
      marketsRejected: rejectedMarkets,
      requests,
      tradesSeen,
      activityTruncated: truncated,
      identity,
    },
  };
}

async function verifyIdentity(
  name: string,
  options: PredictionFetchOptions,
  health: HealthRow[],
  now: () => number,
): Promise<{
  status: "VERIFIED" | "UNVERIFIED" | "AMBIGUOUS" | "UNREACHABLE";
  wallet: string | null;
  evidence: string[];
  reason: string;
  requests: number;
}> {
  const startedSearch = now();
  const searchUrl = new URL("https://gamma-api.polymarket.com/public-search");
  searchUrl.searchParams.set("q", name);
  searchUrl.searchParams.set("search_profiles", "true");
  const search = await fetchPredictionJson<{ profiles?: unknown }>(searchUrl.toString(), options);
  if (search.status !== "OK") {
    health.push(healthFrom("polymarket-gamma-profile-search", search, startedSearch, 0));
    return { status: "UNREACHABLE", wallet: null, evidence: [], reason: "profile search unavailable", requests: 1 };
  }
  const profiles = Array.isArray(search.body?.profiles) ? search.body!.profiles as unknown[] : null;
  health.push(healthFrom("polymarket-gamma-profile-search", search, startedSearch, profiles?.length ?? 0));

  const exact = (profiles ?? []).find((p) =>
    typeof p === "object" && p !== null && (p as Record<string, unknown>).name === name
  ) as Record<string, unknown> | undefined;
  const candidate = typeof exact?.proxyWallet === "string" ? exact.proxyWallet : null;
  let activityRows: unknown = null;
  let requests = 1;
  if (candidate !== null && /^0x[0-9a-fA-F]{40}$/.test(candidate)) {
    const startedActivity = now();
    requests++;
    const probe = await fetchPredictionJson<unknown[]>(
      `https://data-api.polymarket.com/activity?user=${candidate.toLowerCase()}&limit=5`,
      options,
    );
    health.push(healthFrom("polymarket-data-identity-probe", probe, startedActivity, Array.isArray(probe.body) ? probe.body.length : 0));
    if (probe.status !== "OK") {
      return { status: "UNREACHABLE", wallet: null, evidence: [], reason: "identity probe unavailable", requests };
    }
    activityRows = probe.body;
  }
  const verdict = verifyProfileIdentity(name, profiles, activityRows);
  if (verdict.status === "VERIFIED") {
    return { status: "VERIFIED", wallet: verdict.wallet, evidence: verdict.evidence, reason: "", requests };
  }
  return { status: verdict.status, wallet: null, evidence: [], reason: verdict.reason, requests };
}

async function collectActivity(
  config: PredictionCollectorConfig,
  wallet: string,
  known: KnownWallet | null,
  options: PredictionFetchOptions,
  health: HealthRow[],
  now: () => number,
): Promise<{ trades: TradeRow[]; watermark: string | null; truncated: boolean; seen: number; requests: number }> {
  const sameWallet = known !== null && known.wallet.toLowerCase() === wallet;
  const priorMs = sameWallet && known!.watermark ? Date.parse(known!.watermark) : NaN;
  const floorMs = now() - config.backfillHours * 3_600_000;
  // Inclusive start at the watermark second: same-second rows are re-read and de-duplicated by
  // their idempotency key, so nothing at the boundary is lost.
  const startSeconds = Math.floor((Number.isFinite(priorMs) ? priorMs : floorMs) / 1000);
  const pageSize = Math.max(1, Math.min(500, config.activityPageSize));
  // A cycle never reads more trades than one batch RPC accepts.
  const maxPages = Math.max(1, Math.min(config.maxActivityPages, Math.floor(MAX_BATCH_TRADES / pageSize)));
  const rows: unknown[] = [];
  let requests = 0;
  let truncated = false;
  let failed = false;

  for (let page = 0; page < maxPages; page++) {
    const offset = page * pageSize;
    if (offset + pageSize > 5000) {
      truncated = true;
      break;
    }
    const started = now();
    requests++;
    const result = await fetchPredictionJson<unknown[]>(
      dataActivityUrl({ wallet, startSeconds, limit: pageSize, offset }),
      options,
    );
    const got = Array.isArray(result.body) ? result.body : [];
    health.push(healthFrom("polymarket-data-activity", result, started, got.length, { metadata: { page, start: startSeconds } }));
    if (result.status !== "OK") {
      failed = true;
      break;
    }
    rows.push(...got);
    if (got.length < pageSize) break;
    if (page === maxPages - 1) truncated = true;
  }

  const receivedAt = new Date(now()).toISOString();
  const parts = partitionWalletActivity(rows, receivedAt);
  const tradeRows = parts.trades.map(toTradeRow);
  // The watermark only advances to trades actually read. If a page failed, it does not advance
  // at all, so the next cycle re-reads the whole window instead of skipping a gap.
  let watermark: string | null = null;
  if (!failed && tradeRows.length > 0) {
    watermark = tradeRows.reduce((max, t) => (t.traded_at > max ? t.traded_at : max), tradeRows[0].traded_at);
  }
  if (truncated || parts.rejected.length > 0) {
    health.push({
      health_id: `polymarket-data-activity-coverage|${receivedAt}`,
      source: "polymarket-data-activity-coverage",
      status: "DEGRADED",
      http_status: null,
      attempts: null,
      latency_ms: null,
      records_seen: rows.length,
      records_stored: 0,
      detail: truncated
        ? "activity window larger than this cycle's page bound; remaining trades resume next cycle"
        : `${parts.rejected.length} activity rows failed normalization`,
      observed_at: receivedAt,
      metadata: { collector_version: PREDICTION_COLLECTOR_VERSION, rejected: parts.rejected.slice(0, 10), truncated },
    });
  }
  return { trades: tradeRows, watermark, truncated, seen: rows.length, requests };
}

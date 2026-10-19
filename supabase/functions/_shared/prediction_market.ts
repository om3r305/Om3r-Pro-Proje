// Prediction Intelligence -- Phase 1 public-data connector (pure, injectable, read-only).
//
// OBSERVATION ONLY. This module reads Polymarket's public, unauthenticated market-data
// endpoints and normalizes what they return. It never places orders, never signs anything,
// never touches a wallet key, and never feeds a live decision. Every record it produces carries
// the shadow-only safety envelope below. Nothing here is wired to a collector, a database, or
// brian-live-shadow yet; that is explicit follow-up work.
//
// Evidence classes are kept separate on purpose (see PredictionEvidenceClass):
//   OBSERVED_VERIFIED -- something the public API actually returned (a market, a trade row);
//   HYPOTHESIS        -- an inference about strategy or cause (never produced by this module);
//   MEASURED_RESULT   -- a finding that survived an independent forward test (never produced here).

export const PREDICTION_CONTRACT_VERSION = "brian.prediction-market.v0";

export type PredictionEvidenceClass = "OBSERVED_VERIFIED" | "HYPOTHESIS" | "MEASURED_RESULT";

export interface PredictionSafetyEnvelope {
  readOnly: true;
  shadowOnly: true;
  liveExecution: false;
  canChangeLiveDecisions: false;
}

export const PREDICTION_SAFETY: PredictionSafetyEnvelope = Object.freeze({
  readOnly: true,
  shadowOnly: true,
  liveExecution: false,
  canChangeLiveDecisions: false,
});

// Only these official public read hosts may be contacted. Anything else is refused before a
// request is made, so a malformed or injected URL can never send traffic elsewhere.
export const PREDICTION_ALLOWED_HOSTS = Object.freeze([
  "gamma-api.polymarket.com",
  "data-api.polymarket.com",
  "clob.polymarket.com",
]);

export function assertAllowedPredictionUrl(raw: string): URL {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error(`prediction url is not parseable: ${raw}`);
  }
  if (url.protocol !== "https:") throw new Error(`prediction url must be https: ${url.origin}`);
  if (url.username || url.password) throw new Error("prediction url must not carry credentials");
  if (!PREDICTION_ALLOWED_HOSTS.includes(url.hostname)) {
    throw new Error(`prediction host is not allow-listed: ${url.hostname}`);
  }
  return url;
}

// ---------------------------------------------------------------------------------------------
// HTTP with bounded retry. 429 / 5xx / network errors are retried with Retry-After honoured and
// exponential backoff capped. 401/403/451 mean the source refuses us (access or geography): the
// source is reported BLOCKED and is never retried or circumvented.
// ---------------------------------------------------------------------------------------------

export type PredictionFetchStatus = "OK" | "BLOCKED" | "RATE_LIMITED" | "ERROR";

export interface PredictionFetchResult<T = unknown> {
  status: PredictionFetchStatus;
  httpStatus: number | null;
  attempts: number;
  receivedAt: string;
  body: T | null;
  detail: string | null;
}

export interface PredictionFetchOptions {
  fetchImpl?: typeof fetch;
  sleep?: (ms: number) => Promise<void>;
  now?: () => number;
  maxAttempts?: number;
  baseDelayMs?: number;
  maxDelayMs?: number;
  timeoutMs?: number;
}

export function parseRetryAfterMs(header: string | null, nowMs: number): number | null {
  if (header === null) return null;
  const trimmed = header.trim();
  if (trimmed === "") return null;
  if (/^\d+(\.\d+)?$/.test(trimmed)) return Math.round(Number(trimmed) * 1000);
  const at = Date.parse(trimmed);
  if (!Number.isFinite(at)) return null;
  return Math.max(0, at - nowMs);
}

export async function fetchPredictionJson<T = unknown>(
  rawUrl: string,
  options: PredictionFetchOptions = {},
): Promise<PredictionFetchResult<T>> {
  const url = assertAllowedPredictionUrl(rawUrl);
  const fetchImpl = options.fetchImpl ?? fetch;
  const sleep = options.sleep ?? ((ms: number) => new Promise<void>((r) => setTimeout(r, ms)));
  const now = options.now ?? (() => Date.now());
  const maxAttempts = Math.max(1, options.maxAttempts ?? 4);
  const baseDelayMs = options.baseDelayMs ?? 500;
  const maxDelayMs = options.maxDelayMs ?? 30_000;
  const timeoutMs = options.timeoutMs ?? 15_000;

  let lastStatus: PredictionFetchStatus = "ERROR";
  let lastHttp: number | null = null;
  let lastDetail: string | null = null;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    let retryAfterMs: number | null = null;
    try {
      const response = await fetchImpl(url.toString(), {
        method: "GET",
        redirect: "error",
        headers: { accept: "application/json" },
        signal: AbortSignal.timeout(timeoutMs),
      });
      lastHttp = response.status;
      if (response.status === 401 || response.status === 403 || response.status === 451) {
        return {
          status: "BLOCKED",
          httpStatus: response.status,
          attempts: attempt,
          receivedAt: new Date(now()).toISOString(),
          body: null,
          detail: `source refused access (HTTP ${response.status}); not retried, not circumvented`,
        };
      }
      if (response.status === 429 || response.status >= 500) {
        lastStatus = response.status === 429 ? "RATE_LIMITED" : "ERROR";
        lastDetail = `HTTP ${response.status}`;
        retryAfterMs = parseRetryAfterMs(response.headers.get("retry-after"), now());
        await response.body?.cancel();
      } else if (!response.ok) {
        await response.body?.cancel();
        return {
          status: "ERROR",
          httpStatus: response.status,
          attempts: attempt,
          receivedAt: new Date(now()).toISOString(),
          body: null,
          detail: `HTTP ${response.status}`,
        };
      } else {
        const receivedAt = new Date(now()).toISOString();
        try {
          const body = (await response.json()) as T;
          return { status: "OK", httpStatus: response.status, attempts: attempt, receivedAt, body, detail: null };
        } catch {
          return {
            status: "ERROR",
            httpStatus: response.status,
            attempts: attempt,
            receivedAt,
            body: null,
            detail: "response body was not valid JSON",
          };
        }
      }
    } catch (error) {
      lastStatus = "ERROR";
      lastDetail = error instanceof Error ? error.message : String(error);
    }
    if (attempt < maxAttempts) {
      const backoff = Math.min(maxDelayMs, baseDelayMs * 2 ** (attempt - 1));
      await sleep(Math.min(maxDelayMs, Math.max(backoff, retryAfterMs ?? 0)));
    }
  }
  return {
    status: lastStatus,
    httpStatus: lastHttp,
    attempts: maxAttempts,
    receivedAt: new Date(now()).toISOString(),
    body: null,
    detail: lastDetail,
  };
}

// ---------------------------------------------------------------------------------------------
// Market normalization (Gamma). Missing or malformed values stay null -- never coerced to 0.
// ---------------------------------------------------------------------------------------------

export interface PredictionOutcome {
  label: string;
  impliedProbability: number | null;
  clobTokenId: string | null;
}

export interface PredictionMarket {
  contractVersion: typeof PREDICTION_CONTRACT_VERSION;
  evidenceClass: "OBSERVED_VERIFIED";
  source: "polymarket-gamma";
  marketId: string;
  conditionId: string | null;
  eventId: string | null;
  eventTitle: string | null;
  question: string;
  slug: string | null;
  endDate: string | null;
  active: boolean;
  closed: boolean;
  orderBookEnabled: boolean;
  liquidity: number | null;
  volume24h: number | null;
  bestBid: number | null;
  bestAsk: number | null;
  spread: number | null;
  outcomes: PredictionOutcome[];
  sourceUpdatedAt: string | null;
  receivedAt: string;
  safety: PredictionSafetyEnvelope;
}

export type NormalizeResult<T> = { ok: true; value: T } | { ok: false; reason: string };

function finiteOrNull(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

function probabilityOrNull(value: unknown): number | null {
  const n = finiteOrNull(value);
  return n !== null && n >= 0 && n <= 1 ? n : null;
}

function isoOrNull(value: unknown): string | null {
  if (typeof value !== "string" || value === "") return null;
  const t = Date.parse(value);
  return Number.isFinite(t) ? new Date(t).toISOString() : null;
}

// Gamma encodes outcomes / outcomePrices / clobTokenIds as JSON *strings* of arrays.
function parseStringArray(value: unknown): string[] | null {
  let parsed: unknown = value;
  if (typeof value === "string") {
    try {
      parsed = JSON.parse(value);
    } catch {
      return null;
    }
  }
  if (!Array.isArray(parsed)) return null;
  return parsed.map((v) => (typeof v === "string" ? v : v === null || v === undefined ? "" : String(v)));
}

export interface GammaEventContext {
  id: string | null;
  title: string | null;
}

export function normalizeGammaMarket(
  raw: unknown,
  receivedAt: string,
  event: GammaEventContext | null = null,
): NormalizeResult<PredictionMarket> {
  if (typeof raw !== "object" || raw === null) return { ok: false, reason: "market is not an object" };
  const m = raw as Record<string, unknown>;
  const marketId = m.id === undefined || m.id === null ? "" : String(m.id);
  if (marketId === "") return { ok: false, reason: "market has no id" };
  const question = typeof m.question === "string" ? m.question.trim() : "";
  if (question === "") return { ok: false, reason: "market has no question" };

  const labels = parseStringArray(m.outcomes);
  if (labels === null || labels.length === 0) return { ok: false, reason: "outcomes missing or not a JSON array" };
  const prices = m.outcomePrices === undefined || m.outcomePrices === null ? null : parseStringArray(m.outcomePrices);
  if (prices !== null && prices.length !== labels.length) {
    return { ok: false, reason: "outcomePrices length does not match outcomes" };
  }
  const tokens = m.clobTokenIds === undefined || m.clobTokenIds === null ? null : parseStringArray(m.clobTokenIds);
  if (tokens !== null && tokens.length !== labels.length) {
    return { ok: false, reason: "clobTokenIds length does not match outcomes" };
  }

  return {
    ok: true,
    value: {
      contractVersion: PREDICTION_CONTRACT_VERSION,
      evidenceClass: "OBSERVED_VERIFIED",
      source: "polymarket-gamma",
      marketId,
      conditionId: typeof m.conditionId === "string" && m.conditionId !== "" ? m.conditionId : null,
      eventId: event?.id ?? null,
      eventTitle: event?.title ?? null,
      question,
      slug: typeof m.slug === "string" && m.slug !== "" ? m.slug : null,
      endDate: isoOrNull(m.endDate),
      active: m.active === true,
      closed: m.closed === true,
      orderBookEnabled: m.enableOrderBook === true,
      liquidity: finiteOrNull(m.liquidityNum ?? m.liquidity),
      volume24h: finiteOrNull(m.volume24hr),
      bestBid: probabilityOrNull(m.bestBid),
      bestAsk: probabilityOrNull(m.bestAsk),
      spread: finiteOrNull(m.spread),
      outcomes: labels.map((label, i) => ({
        label,
        impliedProbability: prices === null ? null : probabilityOrNull(prices[i]),
        clobTokenId: tokens === null || tokens[i] === "" ? null : tokens[i],
      })),
      sourceUpdatedAt: isoOrNull(m.updatedAt),
      receivedAt,
      safety: PREDICTION_SAFETY,
    },
  };
}

// ---------------------------------------------------------------------------------------------
// Keyset pagination for Gamma. The legacy offset endpoint is deprecated (sunset header); the
// request cursor parameter is `after_cursor` and the response field is `next_cursor`. A wrong
// parameter name (e.g. `cursor`) is silently ignored by the API and would return page 1 forever,
// so the iterator also refuses a cursor it has already seen.
// ---------------------------------------------------------------------------------------------

export interface GammaKeysetPage {
  markets: unknown[];
  nextCursor: string | null;
  fetch: PredictionFetchResult;
}

export function gammaMarketsKeysetUrl(params: { limit: number; afterCursor?: string | null; active?: boolean; closed?: boolean }): string {
  const url = new URL("https://gamma-api.polymarket.com/markets/keyset");
  url.searchParams.set("limit", String(Math.max(1, Math.min(500, Math.trunc(params.limit)))));
  if (params.active !== undefined) url.searchParams.set("active", String(params.active));
  if (params.closed !== undefined) url.searchParams.set("closed", String(params.closed));
  if (params.afterCursor) url.searchParams.set("after_cursor", params.afterCursor);
  return url.toString();
}

export async function* iterateGammaMarkets(
  params: { limit: number; maxPages: number; active?: boolean; closed?: boolean },
  options: PredictionFetchOptions = {},
): AsyncGenerator<GammaKeysetPage> {
  const seen = new Set<string>();
  let cursor: string | null = null;
  for (let page = 0; page < Math.max(1, params.maxPages); page++) {
    const result: PredictionFetchResult<{ markets?: unknown; next_cursor?: unknown }> = await fetchPredictionJson(
      gammaMarketsKeysetUrl({ ...params, afterCursor: cursor }),
      options,
    );
    const markets = result.body && Array.isArray(result.body.markets) ? result.body.markets : [];
    const next = result.body && typeof result.body.next_cursor === "string" && result.body.next_cursor !== ""
      ? result.body.next_cursor
      : null;
    yield { markets, nextCursor: next, fetch: result };
    if (result.status !== "OK" || next === null || seen.has(next)) return;
    seen.add(next);
    cursor = next;
  }
}

// Events carry their markets inline, and -- unlike markets/keyset, which silently ignores it --
// events/keyset honours `tag_slug`, so topic discovery goes through events.
export function gammaEventsKeysetUrl(params: {
  tagSlug: string;
  limit: number;
  afterCursor?: string | null;
  active?: boolean;
  closed?: boolean;
}): string {
  if (!/^[a-z0-9-]{1,64}$/.test(params.tagSlug)) throw new Error(`invalid tag slug: ${params.tagSlug}`);
  const url = new URL("https://gamma-api.polymarket.com/events/keyset");
  url.searchParams.set("limit", String(Math.max(1, Math.min(100, Math.trunc(params.limit)))));
  url.searchParams.set("tag_slug", params.tagSlug);
  if (params.active !== undefined) url.searchParams.set("active", String(params.active));
  if (params.closed !== undefined) url.searchParams.set("closed", String(params.closed));
  if (params.afterCursor) url.searchParams.set("after_cursor", params.afterCursor);
  return url.toString();
}

export function flattenGammaEvents(
  body: unknown,
  receivedAt: string,
): { markets: PredictionMarket[]; rejected: { marketId: string | null; reason: string }[]; nextCursor: string | null } {
  const markets: PredictionMarket[] = [];
  const rejected: { marketId: string | null; reason: string }[] = [];
  const b = typeof body === "object" && body !== null ? body as Record<string, unknown> : {};
  const events = Array.isArray(b.events) ? b.events : [];
  for (const rawEvent of events) {
    if (typeof rawEvent !== "object" || rawEvent === null) continue;
    const e = rawEvent as Record<string, unknown>;
    const context: GammaEventContext = {
      id: e.id === undefined || e.id === null ? null : String(e.id),
      title: typeof e.title === "string" ? e.title : null,
    };
    for (const rawMarket of Array.isArray(e.markets) ? e.markets : []) {
      const result = normalizeGammaMarket(rawMarket, receivedAt, context);
      if (result.ok) markets.push(result.value);
      else {
        const id = typeof rawMarket === "object" && rawMarket !== null ? (rawMarket as Record<string, unknown>).id : null;
        rejected.push({ marketId: id === undefined || id === null ? null : String(id), reason: result.reason });
      }
    }
  }
  const nextCursor = typeof b.next_cursor === "string" && b.next_cursor !== "" ? b.next_cursor : null;
  return { markets, rejected, nextCursor };
}

// Data API activity, oldest first from an inclusive `start` (epoch seconds). Historical offsets are
// capped at 5000 by the API, so an active trader must be resumed by time, not by offset.
export const DATA_ACTIVITY_MAX_OFFSET = 5000;

export function dataActivityUrl(params: { wallet: string; startSeconds: number; limit: number; offset: number }): string {
  if (!WALLET_RE.test(params.wallet)) throw new Error("a full wallet address is required");
  if (!Number.isInteger(params.startSeconds) || params.startSeconds < 0) throw new Error("startSeconds must be a non-negative integer");
  const limit = Math.max(1, Math.min(500, Math.trunc(params.limit)));
  const offset = Math.max(0, Math.trunc(params.offset));
  if (offset + limit > DATA_ACTIVITY_MAX_OFFSET) throw new Error("activity offset would exceed the API's historical cap");
  const url = new URL("https://data-api.polymarket.com/activity");
  url.searchParams.set("user", params.wallet.toLowerCase());
  url.searchParams.set("type", "TRADE");
  url.searchParams.set("start", String(params.startSeconds));
  url.searchParams.set("sortBy", "TIMESTAMP");
  url.searchParams.set("sortDirection", "ASC");
  url.searchParams.set("limit", String(limit));
  url.searchParams.set("offset", String(offset));
  return url.toString();
}

// ---------------------------------------------------------------------------------------------
// Public wallet trades (Data API /activity). Only rows of type TRADE are trades; REDEEM etc. are
// settlements and are reported separately so realized and unrealized outcomes are never mixed.
// ---------------------------------------------------------------------------------------------

export interface PredictionWalletTrade {
  contractVersion: typeof PREDICTION_CONTRACT_VERSION;
  evidenceClass: "OBSERVED_VERIFIED";
  source: "polymarket-data-activity";
  idempotencyKey: string;
  wallet: string;
  conditionId: string;
  assetId: string;
  marketSlug: string | null;
  title: string | null;
  outcome: string | null;
  side: "BUY" | "SELL";
  price: number;
  size: number;
  usdcSize: number | null;
  transactionHash: string;
  tradedAt: string;
  receivedAt: string;
  safety: PredictionSafetyEnvelope;
}

const WALLET_RE = /^0x[0-9a-fA-F]{40}$/;
const TX_RE = /^0x[0-9a-fA-F]{64}$/;

export function normalizeWalletActivityRow(raw: unknown, receivedAt: string): NormalizeResult<PredictionWalletTrade> {
  if (typeof raw !== "object" || raw === null) return { ok: false, reason: "row is not an object" };
  const r = raw as Record<string, unknown>;
  if (r.type !== "TRADE") return { ok: false, reason: `row type ${String(r.type)} is not a TRADE` };
  const wallet = typeof r.proxyWallet === "string" && WALLET_RE.test(r.proxyWallet) ? r.proxyWallet.toLowerCase() : null;
  if (wallet === null) return { ok: false, reason: "row has no valid proxyWallet" };
  const tx = typeof r.transactionHash === "string" && TX_RE.test(r.transactionHash) ? r.transactionHash.toLowerCase() : null;
  if (tx === null) return { ok: false, reason: "row has no valid transactionHash" };
  const side = r.side === "BUY" || r.side === "SELL" ? r.side : null;
  if (side === null) return { ok: false, reason: "row side is not BUY or SELL" };
  const price = probabilityOrNull(r.price);
  const size = finiteOrNull(r.size);
  const ts = finiteOrNull(r.timestamp);
  if (price === null) return { ok: false, reason: "row price missing or outside [0,1]" };
  if (size === null || size <= 0) return { ok: false, reason: "row size missing or non-positive" };
  if (ts === null || ts <= 0) return { ok: false, reason: "row timestamp missing" };
  const conditionId = typeof r.conditionId === "string" && r.conditionId !== "" ? r.conditionId : null;
  const assetId = r.asset === undefined || r.asset === null || r.asset === "" ? null : String(r.asset);
  if (conditionId === null || assetId === null) return { ok: false, reason: "row has no conditionId or asset" };

  return {
    ok: true,
    value: {
      contractVersion: PREDICTION_CONTRACT_VERSION,
      evidenceClass: "OBSERVED_VERIFIED",
      source: "polymarket-data-activity",
      // One transaction can carry several fills, so the key also pins asset, side, price, size
      // and time. Re-ingesting the same row yields the same key; a different fill never collides.
      idempotencyKey: [wallet, tx, assetId, side, price, size, ts].join(":"),
      wallet,
      conditionId,
      assetId,
      marketSlug: typeof r.slug === "string" ? r.slug : null,
      title: typeof r.title === "string" ? r.title : null,
      outcome: typeof r.outcome === "string" ? r.outcome : null,
      side,
      price,
      size,
      usdcSize: finiteOrNull(r.usdcSize),
      transactionHash: tx,
      tradedAt: new Date(ts * 1000).toISOString(),
      receivedAt,
      safety: PREDICTION_SAFETY,
    },
  };
}

export function partitionWalletActivity(rows: unknown[], receivedAt: string): {
  trades: PredictionWalletTrade[];
  settlements: number;
  rejected: { index: number; reason: string }[];
} {
  const trades: PredictionWalletTrade[] = [];
  const seen = new Set<string>();
  const rejected: { index: number; reason: string }[] = [];
  let settlements = 0;
  rows.forEach((row, index) => {
    const type = typeof row === "object" && row !== null ? (row as Record<string, unknown>).type : undefined;
    if (type === "REDEEM" || type === "MERGE" || type === "SPLIT" || type === "REWARD" || type === "CONVERSION") {
      settlements++;
      return;
    }
    const result = normalizeWalletActivityRow(row, receivedAt);
    if (!result.ok) {
      rejected.push({ index, reason: result.reason });
      return;
    }
    if (seen.has(result.value.idempotencyKey)) return;
    seen.add(result.value.idempotencyKey);
    trades.push(result.value);
  });
  return { trades, settlements, rejected };
}

// ---------------------------------------------------------------------------------------------
// Identity verification. A display name alone is not an identity: names are not unique across
// case/lookalikes and a wallet must never be guessed or completed from a fragment. A profile is
// VERIFIED only when (1) exactly one search candidate matches the expected name exactly, (2) its
// proxyWallet is a full valid address, and (3) the wallet's own public activity independently
// reports the same wallet and the same name. Anything else stays UNVERIFIED and tracking waits.
// ---------------------------------------------------------------------------------------------

export type IdentityVerdict =
  | { status: "VERIFIED"; name: string; wallet: string; evidence: string[] }
  | { status: "UNVERIFIED" | "AMBIGUOUS"; reason: string };

export function verifyProfileIdentity(
  expectedName: string,
  searchProfiles: unknown,
  activityRows: unknown,
): IdentityVerdict {
  if (!Array.isArray(searchProfiles)) return { status: "UNVERIFIED", reason: "profile search returned no list" };
  const exact = searchProfiles.filter((p) =>
    typeof p === "object" && p !== null && (p as Record<string, unknown>).name === expectedName
  ) as Record<string, unknown>[];
  if (exact.length === 0) return { status: "UNVERIFIED", reason: `no profile named exactly ${expectedName}` };
  const wallets = new Set(
    exact.map((p) => (typeof p.proxyWallet === "string" ? p.proxyWallet.toLowerCase() : "")),
  );
  if (exact.length > 1 && wallets.size > 1) {
    return { status: "AMBIGUOUS", reason: `${exact.length} profiles share the exact name with different wallets` };
  }
  const candidate = exact[0];
  const wallet = typeof candidate.proxyWallet === "string" ? candidate.proxyWallet.toLowerCase() : "";
  if (!WALLET_RE.test(wallet)) return { status: "UNVERIFIED", reason: "candidate has no full valid wallet address" };

  if (!Array.isArray(activityRows) || activityRows.length === 0) {
    return { status: "UNVERIFIED", reason: "no public activity to cross-check the wallet against" };
  }
  const mismatched = activityRows.filter((row) => {
    if (typeof row !== "object" || row === null) return true;
    const r = row as Record<string, unknown>;
    return typeof r.proxyWallet !== "string" || r.proxyWallet.toLowerCase() !== wallet ||
      (r.name !== undefined && r.name !== expectedName);
  });
  if (mismatched.length > 0) {
    return { status: "UNVERIFIED", reason: `${mismatched.length} activity rows disagree with the candidate wallet or name` };
  }
  const named = activityRows.some((row) => (row as Record<string, unknown>).name === expectedName);
  if (!named) return { status: "UNVERIFIED", reason: "activity rows never repeat the expected name" };
  return {
    status: "VERIFIED",
    name: expectedName,
    wallet,
    evidence: ["gamma public-search exact name match", "data-api activity wallet match", "data-api activity name match"],
  };
}

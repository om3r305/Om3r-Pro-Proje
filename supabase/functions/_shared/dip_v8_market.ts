import {
  type Bar,
  type Book,
  type Flow,
  parseBars,
  readBook,
  readFlow,
  type Rules,
  type Segment,
  SYMBOL,
  TF_MS,
} from "./dip_v8.ts";
const HOSTS = [
  "https://api.binance.com",
  "https://api1.binance.com",
  "https://api3.binance.com",
];
const SAFE_ROUTES = new Set([
  "/api/v3/klines",
  "/api/v3/depth",
  "/api/v3/aggTrades",
  "/api/v3/exchangeInfo",
]);
type Fetcher = typeof fetch;
export async function marketJson(
  path: string,
  fetcher: Fetcher = fetch,
): Promise<unknown> {
  if (!SAFE_ROUTES.has(path.split("?")[0])) {
    throw Error("FORBIDDEN_MARKET_ROUTE");
  }
  let error = "MARKET_UNAVAILABLE";
  for (const host of HOSTS) {
    try {
      const r = await fetcher(host + path, {
        method: "GET",
        headers: {
          accept: "application/json",
          "user-agent": "Brian-DIP-V8-Shadow/8.1",
        },
        signal: AbortSignal.timeout(5500),
      });
      if (!r.ok) {
        error = "HTTP_" + r.status;
        if ([418, 429].includes(r.status)) throw Error("RATE_LIMIT");
        continue;
      }
      return await r.json();
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
      if (error === "RATE_LIMIT") break;
    }
  }
  throw Error("BINANCE:" + error);
}
export type Market = {
  bars: Record<string, Bar[]>;
  book: Book;
  flow: Flow;
  rules: Rules;
  availableAt: number;
};
let rulesCache: { at: number; rules: Rules } | null = null;
export async function exchangeRules(fetcher: Fetcher = fetch): Promise<Rules> {
  if (rulesCache && Date.now() - rulesCache.at < 10 * 60_000) {
    return rulesCache.rules;
  }
  const raw = await marketJson(
    "/api/v3/exchangeInfo?symbol=" + SYMBOL,
    fetcher,
  ) as {
    symbols?: {
      symbol: string;
      status: string;
      filters: Record<string, unknown>[];
    }[];
  };
  const s = raw.symbols?.find((s) => s.symbol === SYMBOL);
  if (!s || s.status !== "TRADING") throw Error("SYMBOL_NOT_TRADING");
  const lot = s.filters.find((x) => x.filterType === "LOT_SIZE"),
    notional = s.filters.find((x) => x.filterType === "NOTIONAL") ||
      s.filters.find((x) => x.filterType === "MIN_NOTIONAL");
  const rules = {
    minQty: Number(lot?.minQty),
    maxQty: Number(lot?.maxQty),
    stepSize: Number(lot?.stepSize),
    minNotional: Number(notional?.minNotional),
  };
  if (!Object.values(rules).every((x) => Number.isFinite(x) && x > 0)) {
    throw Error("INVALID_MARKET_FILTERS");
  }
  rulesCache = { at: Date.now(), rules };
  return rules;
}
export async function getMarket(fetcher: Fetcher = fetch): Promise<Market> {
  const tfs = Object.keys(TF_MS);
  const [candles, depth, trades, rules] = await Promise.all([
    Promise.all(
      tfs.map(async (tf) =>
        [
          tf,
          await marketJson(
            `/api/v3/klines?symbol=${SYMBOL}&interval=${tf}&limit=${
              tf === "4h" ? 180 : 240
            }`,
            fetcher,
          ),
        ] as const
      ),
    ),
    marketJson(`/api/v3/depth?symbol=${SYMBOL}&limit=20`, fetcher).then((raw) =>
      readBook(raw, Date.now())
    ),
    marketJson(`/api/v3/aggTrades?symbol=${SYMBOL}&limit=150`, fetcher),
    exchangeRules(fetcher),
  ]);
  const availableAt = Date.now(), bars: Record<string, Bar[]> = {};
  for (const [tf, raw] of candles) bars[tf] = parseBars(raw, tf, availableAt);
  if (availableAt - depth.receivedAt > 15_000 || depth.spreadBps > 30) {
    throw Error("STALE_OR_WIDE_BOOK");
  }
  return {
    bars,
    book: depth,
    flow: readFlow(trades, availableAt),
    rules,
    availableAt,
  };
}
async function boundary(
  start: number,
  end: number,
  fallback: number,
  fetcher: Fetcher,
): Promise<Segment> {
  const points: { t: number; id: number; p: number }[] = [];
  let nextId: number | null = null, done = false;
  for (let page = 0; page < 8; page++) {
    const params = nextId === null
      ? `startTime=${start}&endTime=${end - 1}`
      : `fromId=${nextId}`;
    const rows = await marketJson(
      `/api/v3/aggTrades?symbol=${SYMBOL}&${params}&limit=1000`,
      fetcher,
    );
    if (!Array.isArray(rows)) throw Error("INVALID_BOUNDARY_TRADES");
    for (const row of rows) {
      const id = Number(row.a), t = Number(row.T), p = Number(row.p);
      if (!Number.isSafeInteger(id) || !Number.isFinite(t) || !(p > 0)) {
        throw Error("INVALID_BOUNDARY_TRADES");
      }
      if (nextId !== null && id !== nextId) throw Error("TRADE_ID_GAP");
      nextId = id + 1;
      if (t < start) throw Error("BOUNDARY_BEFORE_DECISION");
      if (t >= end) {
        done = true;
        break;
      }
      if (points.length && t < points.at(-1)!.t) {
        throw Error("TRADE_TIME_ORDER");
      }
      points.push({ t, id, p });
    }
    if (done || rows.length < 1000) {
      done = true;
      break;
    }
  }
  if (!done) throw Error("BOUNDARY_PAGE_LIMIT");
  const prices = points.map((x) => x.p);
  return {
    start,
    end,
    kind: "TRADES",
    points,
    o: prices[0] ?? fallback,
    c: prices.at(-1) ?? fallback,
    h: prices.length ? Math.max(...prices) : fallback,
    l: prices.length ? Math.min(...prices) : fallback,
  };
}
// One-minute sealed bars plus exact timestamped trades for the partial boundary minutes.
// Raw bars never extend across the activation/deadline boundary.
export async function pricePath(
  start: number,
  due: number,
  nowMs: number,
  entry: number,
  fetcher: Fetcher = fetch,
): Promise<{ end: number; segments: Segment[] }> {
  const minute = 60_000, sealed = Math.floor((nowMs - 1500) / minute) * minute;
  const end = due <= sealed ? due : sealed;
  if (end <= start) return { end: start, segments: [] };
  if (end - start > 181 * minute) throw Error("PATH_WINDOW_TOO_LARGE");
  const segments: Segment[] = [],
    firstFull = Math.min(end, Math.ceil(start / minute) * minute),
    lastFull = Math.floor(end / minute) * minute;
  if (start < firstFull) {
    segments.push(await boundary(start, firstFull, entry, fetcher));
  }
  const fullStart = firstFull;
  if (lastFull > fullStart) {
    const raw = await marketJson(
      `/api/v3/klines?symbol=${SYMBOL}&interval=1m&startTime=${fullStart}&endTime=${
        lastFull - 1
      }&limit=200`,
      fetcher,
    );
    if (!Array.isArray(raw)) throw Error("NO_PRICE_PATH");
    let cursor = fullStart;
    for (const x of raw) {
      if (!Array.isArray(x)) throw Error("INVALID_PRICE_PATH");
      const t = Number(x[0]),
        ct = Number(x[6]),
        o = Number(x[1]),
        h = Number(x[2]),
        l = Number(x[3]),
        c = Number(x[4]);
      if (
        t !== cursor || ct !== t + minute - 1 || ct >= end ||
        ![o, h, l, c].every(Number.isFinite) || l <= 0 || h < Math.max(o, c) ||
        l > Math.min(o, c)
      ) throw Error("PRICE_PATH_GAP_OR_ORDER");
      segments.push({ start: t, end: ct + 1, o, h, l, c, kind: "BAR" });
      cursor = ct + 1;
    }
    if (cursor !== lastFull) throw Error("PRICE_PATH_INCOMPLETE");
  }
  const tailStart = Math.max(firstFull, lastFull);
  if (tailStart < end) {
    segments.push(
      await boundary(tailStart, end, segments.at(-1)?.c ?? entry, fetcher),
    );
  }
  return { end, segments };
}

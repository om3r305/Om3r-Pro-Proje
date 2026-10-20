// Polymarket CLOB order book reducer for the public market websocket (pure).
//
// Message shapes observed 2026-10-20 on wss://ws-subscriptions-clob.polymarket.com/ws/market:
//   book:          { event_type:"book", asset_id, bids:[{price,size}], asks:[{price,size}], timestamp, hash }
//   price_change:  { event_type:"price_change", price_changes:[{asset_id, price, size, side, best_bid, best_ask}], timestamp }
// `size` in price_change is the ABSOLUTE size now resting at that price ("0" removes the level);
// side BUY = bid, SELL = ask. A book message replaces the whole side state for that asset.
// Prices/sizes stay decimal strings on the wire; they are parsed once here and kept as numbers
// rounded to the tick, because every downstream computation is in cents and shares.

export interface Level {
  price: number;
  size: number;
}

export interface AssetBook {
  assetId: string;
  bids: Map<number, number>;
  asks: Map<number, number>;
  lastUpdateMs: number | null;
  synced: boolean; // true only after a full `book` snapshot has been applied
}

export type BookState = Map<string, AssetBook>;

function px(v: unknown): number | null {
  const n = typeof v === "string" ? Number(v) : typeof v === "number" ? v : NaN;
  return Number.isFinite(n) && n >= 0 && n <= 1 ? Math.round(n * 1e4) / 1e4 : null;
}
function sz(v: unknown): number | null {
  const n = typeof v === "string" ? Number(v) : typeof v === "number" ? v : NaN;
  return Number.isFinite(n) && n >= 0 ? n : null;
}

export function emptyBook(assetId: string): AssetBook {
  return { assetId, bids: new Map(), asks: new Map(), lastUpdateMs: null, synced: false };
}

export interface ApplyResult {
  applied: number;
  ignored: number;
  unsyncedDeltas: number;
}

// Apply one decoded websocket payload (object or array). Unknown event types are ignored; deltas
// for an asset that has not received a snapshot yet are counted and NOT applied (the book would
// otherwise be built from a partial view).
export function applyMessage(state: BookState, payload: unknown, tracked: Set<string>): ApplyResult {
  const res: ApplyResult = { applied: 0, ignored: 0, unsyncedDeltas: 0 };
  const items = Array.isArray(payload) ? payload : [payload];
  for (const raw of items) {
    if (typeof raw !== "object" || raw === null) {
      res.ignored++;
      continue;
    }
    const m = raw as Record<string, unknown>;
    const ts = Number(m.timestamp);
    const tsMs = Number.isFinite(ts) ? ts : null;
    if (m.event_type === "book") {
      const assetId = String(m.asset_id ?? "");
      if (!tracked.has(assetId)) {
        res.ignored++;
        continue;
      }
      const b = emptyBook(assetId);
      for (const [side, target] of [[m.bids, b.bids], [m.asks, b.asks]] as const) {
        for (const lvl of Array.isArray(side) ? side : []) {
          const p = px((lvl as Record<string, unknown>).price);
          const s = sz((lvl as Record<string, unknown>).size);
          if (p !== null && s !== null && s > 0) target.set(p, s);
        }
      }
      b.lastUpdateMs = tsMs;
      b.synced = true;
      state.set(assetId, b);
      res.applied++;
    } else if (m.event_type === "price_change") {
      for (const ch of Array.isArray(m.price_changes) ? m.price_changes : []) {
        const c = ch as Record<string, unknown>;
        const assetId = String(c.asset_id ?? "");
        if (!tracked.has(assetId)) {
          res.ignored++;
          continue;
        }
        const book = state.get(assetId);
        if (!book || !book.synced) {
          res.unsyncedDeltas++;
          continue;
        }
        const p = px(c.price);
        const s = sz(c.size);
        const sideMap = c.side === "BUY" ? book.bids : c.side === "SELL" ? book.asks : null;
        if (p === null || s === null || sideMap === null) {
          res.ignored++;
          continue;
        }
        if (s === 0) sideMap.delete(p);
        else sideMap.set(p, s);
        book.lastUpdateMs = tsMs ?? book.lastUpdateMs;
        res.applied++;
      }
    } else {
      res.ignored++;
    }
  }
  return res;
}

export function bestBid(b: AssetBook): number | null {
  let best: number | null = null;
  for (const p of b.bids.keys()) if (best === null || p > best) best = p;
  return best;
}
export function bestAsk(b: AssetBook): number | null {
  let best: number | null = null;
  for (const p of b.asks.keys()) if (best === null || p < best) best = p;
  return best;
}

export function topLevels(b: AssetBook, n: number): { bids: Level[]; asks: Level[] } {
  const bids = [...b.bids.entries()].sort((x, y) => y[0] - x[0]).slice(0, n).map(([price, size]) => ({ price, size }));
  const asks = [...b.asks.entries()].sort((x, y) => x[0] - y[0]).slice(0, n).map(([price, size]) => ({ price, size }));
  return { bids, asks };
}

export type FillResult =
  | { status: "FILLED"; shares: number; avgPrice: number; worstPrice: number; bestAsk: number; slippage: number }
  | { status: "UNFILLABLE"; reason: "NO_BOOK" | "NO_ASKS" | "INSUFFICIENT_DEPTH_WITHIN_TOLERANCE" };

// Taker buy of `shares` walking the ask side, never paying more than bestAsk + tolerance.
// Partial fills are reported as UNFILLABLE: the pre-registered rule is a fixed size.
export function walkAsks(asks: Level[], shares: number, tolerance: number): FillResult {
  if (!asks.length) return { status: "UNFILLABLE", reason: "NO_ASKS" };
  const sorted = [...asks].sort((a, b) => a.price - b.price);
  const best = sorted[0].price;
  const limit = Math.round((best + tolerance) * 1e4) / 1e4;
  let remaining = shares;
  let cost = 0;
  let worst = best;
  for (const lvl of sorted) {
    if (lvl.price > limit + 1e-9) break;
    const take = Math.min(remaining, lvl.size);
    cost += take * lvl.price;
    remaining -= take;
    worst = lvl.price;
    if (remaining <= 1e-9) break;
  }
  if (remaining > 1e-9) return { status: "UNFILLABLE", reason: "INSUFFICIENT_DEPTH_WITHIN_TOLERANCE" };
  const avg = cost / shares;
  return { status: "FILLED", shares, avgPrice: Math.round(avg * 1e6) / 1e6, worstPrice: worst, bestAsk: best, slippage: Math.round((avg - best) * 1e6) / 1e6 };
}

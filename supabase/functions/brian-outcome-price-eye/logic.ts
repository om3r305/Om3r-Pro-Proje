export const PRICE_CAPTURE_VERSION = 'brian.outcome-price-eye.v1';
export const BOOK_URI = 'https://api.binance.com/api/v3/ticker/bookTicker';
export type Quote = { asset_id: string; bid_price: string; ask_price: string; observed_mid_price: number; bid_qty: number; ask_qty: number };

const positive = (value: unknown): number | null => {
  if (typeof value !== 'number' && (typeof value !== 'string' || !value.trim())) return null;
  const n = Number(value); return Number.isFinite(n) && n > 0 ? n : null;
};
export function parseOutcomeQuotes(payload: unknown, assets: string[]) {
  if (!Array.isArray(payload)) throw Error('BOOK_SCHEMA_INVALID');
  if (assets.length > 500 || new Set(assets).size !== assets.length || assets.some(a => !/^crypto:[A-Z0-9]{1,30}USDT$/.test(a))) throw Error('WATCH_ASSETS_INVALID');
  const wanted = new Set(assets), seen = new Set<string>(), duplicate = new Set<string>();
  const quotes = new Map<string, Quote>();
  for (const item of payload) {
    if (!item || typeof item !== 'object') continue;
    const row = item as Record<string, unknown>, asset = `crypto:${String(row.symbol ?? '')}`;
    if (!wanted.has(asset)) continue;
    if (seen.has(asset)) { duplicate.add(asset); quotes.delete(asset); continue; }
    seen.add(asset);
    const bid = positive(row.bidPrice), ask = positive(row.askPrice), bidQty = positive(row.bidQty), askQty = positive(row.askQty);
    if (bid === null || ask === null || bidQty === null || askQty === null || ask < bid) continue;
    const mid = bid / 2 + ask / 2;
    if (!Number.isFinite(mid) || !(mid > 0)) continue;
    quotes.set(asset, { asset_id: asset, bid_price: String(row.bidPrice), ask_price: String(row.askPrice), observed_mid_price: mid, bid_qty: bidQty, ask_qty: askQty });
  }
  for (const asset of duplicate) quotes.delete(asset);
  return { quotes: assets.flatMap(a => quotes.has(a) ? [quotes.get(a)!] : []), missing: assets.filter(a => !quotes.has(a)) };
}
export async function readBookSnapshot(fetcher: typeof fetch = fetch): Promise<{ payload: unknown; receivedAt: string; responseHash: string; requestStartedAt: string }> {
  const requestStartedAt = new Date().toISOString();
  const r = await fetcher(BOOK_URI, { redirect: 'error', headers: { accept: 'application/json', 'user-agent': 'BrianOutcomePrice/1.0' }, signal: AbortSignal.timeout(6500) });
  if (!r.ok) {
    await r.body?.cancel();
    const error = new Error(`BOOK_HTTP_${r.status}`) as Error & { retryAfterSeconds?: number };
    const raw = r.headers.get('retry-after'), seconds = raw && /^\d+$/.test(raw) ? Number(raw) : 60;
    error.retryAfterSeconds = r.status === 429 || r.status === 418 ? Math.min(3600, Math.max(60, seconds)) : 60;
    throw error;
  }
  if (Number(r.headers.get('content-length')) > 2_000_000) { await r.body?.cancel(); throw Error('BOOK_TOO_LARGE'); }
  const reader = r.body?.getReader(); if (!reader) throw Error('BOOK_EMPTY');
  const chunks: Uint8Array[] = []; let length = 0;
  try { while (true) { const part = await reader.read(); if (part.done) break; length += part.value.byteLength; if (length > 2_000_000) throw Error('BOOK_TOO_LARGE'); chunks.push(part.value); } }
  finally { await reader.cancel().catch(() => {}); }
  const receivedAt = new Date().toISOString();
  const bytes = new Uint8Array(length); let offset = 0; for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  const responseHash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(b => b.toString(16).padStart(2, '0')).join('');
  return { payload: JSON.parse(new TextDecoder().decode(bytes)), receivedAt, responseHash, requestStartedAt };
}

// NIULAI4 pilot runtime (I/O). Local, read-only, shadow-only. Writes ONLY to the local output
// directory; never to CORE; never places an order; uses no key.

import { fetchPredictionJson, normalizeGammaResolution, verifyProfileIdentity } from "../../supabase/functions/_shared/prediction_market.ts";
import { applyMessage, type BookState, bestAsk, bestBid, topLevels } from "./book.ts";
import type { Capture, Observation, Outcome } from "./evaluate.ts";
import { reconnectDelayMs, StopGuard } from "./guards.ts";
import { Recorder } from "./recorder.ts";
import { assertPilotHost, RULES } from "./rules.ts";
import { SpotSeries, spotDirection } from "./spot.ts";
import { ActivityWatcher, captureTimes } from "./watcher.ts";
import { activeAndNext, inSamplingTail, type WindowRef, windowMatchesMarket, withinActiveHours } from "./windows.ts";

interface MarketMeta {
  window: WindowRef;
  marketId: string;
  tokens: Record<Outcome, string>;
  feeRate: number | null;
}

export async function verifyIdentityNow(): Promise<string> {
  const search = await fetchPredictionJson<{ profiles?: unknown[] }>(
    `https://gamma-api.polymarket.com/public-search?q=${RULES.displayName}&search_profiles=true`,
  );
  if (search.status !== "OK") return "UNREACHABLE";
  const probe = await fetchPredictionJson<unknown[]>(`https://data-api.polymarket.com/activity?user=${RULES.wallet}&limit=5`);
  if (probe.status !== "OK") return "UNREACHABLE";
  const v = verifyProfileIdentity(RULES.displayName, search.body?.profiles ?? null, probe.body);
  return v.status === "VERIFIED" && v.wallet === RULES.wallet ? "VERIFIED" : v.status === "VERIFIED" ? "WALLET_CHANGED" : v.status;
}

async function loadMarket(w: WindowRef): Promise<MarketMeta | null> {
  const r = await fetchPredictionJson<Record<string, unknown>>(`https://gamma-api.polymarket.com/markets/slug/${w.slug}`);
  if (r.status !== "OK" || !r.body) return null;
  const m = r.body;
  if (!windowMatchesMarket(w, m.eventStartTime, m.endDate)) return null;
  const outcomes = JSON.parse(String(m.outcomes ?? "[]")) as string[];
  const tokens = JSON.parse(String(m.clobTokenIds ?? "[]")) as string[];
  const up = outcomes.indexOf("Up");
  const down = outcomes.indexOf("Down");
  if (up < 0 || down < 0 || !tokens[up] || !tokens[down]) return null;
  const schedule = m.feeSchedule as { rate?: unknown } | undefined;
  const rate = Number(schedule?.rate);
  return {
    window: w,
    marketId: String(m.id),
    tokens: { Up: tokens[up], Down: tokens[down] },
    feeRate: m.feesEnabled === true && Number.isFinite(rate) ? rate : m.feesEnabled === false ? 0 : null,
  };
}

export async function runPilot(opts: { outDir: string; rulesFingerprint: string; clockOffsetMs: number; guard: StopGuard }) {
  const clock = () => Date.now() + opts.clockOffsetMs;
  const rec = new Recorder({ dir: opts.outDir, capBytes: RULES.diskCapBytes, rulesVersion: "niulai4-pilot.v1", rulesFingerprint: opts.rulesFingerprint, now: clock });
  const guard = opts.guard;
  const books: BookState = new Map();
  const spot = new SpotSeries();
  const markets = new Map<string, MarketMeta>();
  const pendingCaptures: { obs: Observation; delaySeconds: number; atMs: number }[] = [];
  const resolutionsDue = new Map<string, { marketId: string; dueMs: number }>();
  const stats = { wsMessages: 0, wsReconnects: 0, spotReconnects: 0, coinbaseErrors: 0, unsyncedDeltas: 0, polls: 0, pollErrors: 0, observations: 0, captures: 0, tailSamples: 0 };
  const wsRef: { current: WebSocket | null } = { current: null };
  let wsAttempt = 0;
  let subscribed = "";
  let lastIdentityMs = clock();

  const write = async (kind: Parameters<Recorder["write"]>[0], r: Record<string, unknown>) => {
    if (!(await rec.write(kind, r))) guard.stop("DISK_CAP");
  };

  const connectBook = (assetIds: string[]) => {
    try { wsRef.current?.close(); } catch { /* ignore */ }
    books.clear();
    subscribed = assetIds.join(",");
    const url = assertPilotHost("wss://ws-subscriptions-clob.polymarket.com/ws/market").toString();
    const s = new WebSocket(url);
    wsRef.current = s;
    s.onopen = () => {
      wsAttempt = 0;
      s.send(JSON.stringify({ assets_ids: assetIds, type: "market" }));
    };
    s.onmessage = (ev) => {
      stats.wsMessages++;
      try {
        const r = applyMessage(books, JSON.parse(String(ev.data)), new Set(assetIds));
        stats.unsyncedDeltas += r.unsyncedDeltas;
      } catch { /* malformed frame: counted implicitly by no-apply */ }
    };
    s.onclose = () => {
      if (guard.stopped || wsRef.current !== s) return;
      stats.wsReconnects++;
      guard.onWsReconnect();
      void write("health", { event: "WS_GAP_START", atMs: clock() });
      setTimeout(() => { if (!guard.stopped && wsRef.current === s) connectBook(assetIds); }, reconnectDelayMs(wsAttempt++));
    };
  };

  // Spot: Binance public bookTicker stream (reconnecting) + Coinbase REST cross-check every 5 s.
  const spotRef: { current: WebSocket | null; attempt: number } = { current: null, attempt: 0 };
  const connectSpot = () => {
    const s = new WebSocket(assertPilotHost("wss://data-stream.binance.vision/ws/btcusdt@bookTicker").toString());
    spotRef.current = s;
    s.onopen = () => { spotRef.attempt = 0; };
    s.onmessage = (ev) => {
      try {
        const d = JSON.parse(String(ev.data));
        const mid = (Number(d.b) + Number(d.a)) / 2;
        spot.add({ tMs: clock(), mid, source: "binance" });
      } catch { /* ignore */ }
    };
    s.onclose = () => {
      if (guard.stopped || spotRef.current !== s) return;
      stats.spotReconnects++;
      void write("health", { event: "SPOT_GAP_START", atMs: clock() });
      setTimeout(() => { if (!guard.stopped && spotRef.current === s) connectSpot(); }, reconnectDelayMs(spotRef.attempt++));
    };
  };
  connectSpot();
  let lastCoinbase = 0;

  const snapshotFor = (meta: MarketMeta | undefined) => {
    if (!meta) return null;
    const up = books.get(meta.tokens.Up);
    const dn = books.get(meta.tokens.Down);
    if (!up?.synced || !dn?.synced) return null;
    const mid = (b: typeof up) => {
      const bb = bestBid(b);
      const ba = bestAsk(b);
      return bb !== null && ba !== null ? Math.round(((bb + ba) / 2) * 1e4) / 1e4 : null;
    };
    return {
      asks: { Up: topLevels(up, 20).asks, Down: topLevels(dn, 20).asks },
      bids: { Up: topLevels(up, RULES.bookSampleLevels).bids, Down: topLevels(dn, RULES.bookSampleLevels).bids },
      mids: { Up: mid(up), Down: mid(dn) },
    };
  };

  const watcher = new ActivityWatcher(RULES.wallet, clock);
  await write("health", { event: "START", stats });

  let lastPoll = 0;
  let lastHealth = clock();
  while (!guard.stopped) {
    const now = clock();
    guard.checkDuration();
    guard.checkDisk(rec.bytesWritten);
    if (now - lastIdentityMs > 86_400_000) {
      const id = await verifyIdentityNow();
      guard.onIdentity(id);
      lastIdentityMs = now;
      await write("health", { event: "IDENTITY", status: id });
    }

    // Keep the active and next window's markets loaded and subscribed.
    const [a, b] = activeAndNext(now);
    for (const w of [a, b]) {
      if (!markets.has(w.slug)) {
        const meta = await loadMarket(w);
        if (meta) {
          markets.set(w.slug, meta);
          resolutionsDue.set(w.slug, { marketId: meta.marketId, dueMs: (w.endSec + 3600) * 1000 });
          await write("window", { slug: w.slug, marketId: meta.marketId, tokens: meta.tokens, feeRate: meta.feeRate, startSec: w.startSec, endSec: w.endSec });
        }
      }
    }
    const wanted = [a, b].flatMap((w) => (markets.get(w.slug) ? [markets.get(w.slug)!.tokens.Up, markets.get(w.slug)!.tokens.Down] : []));
    if (wanted.length && wanted.join(",") !== subscribed) connectBook(wanted);

    if (now - lastCoinbase >= RULES.spotCrossCheckMs) {
      lastCoinbase = now;
      try {
        const r = await fetch(assertPilotHost("https://api.exchange.coinbase.com/products/BTC-USD/ticker").toString(), { signal: AbortSignal.timeout(5000) });
        guard.onHttpStatus(r.status);
        if (r.ok) {
          const d = await r.json();
          spot.add({ tMs: clock(), mid: (Number(d.bid) + Number(d.ask)) / 2, source: "coinbase" });
        } else stats.coinbaseErrors++;
      } catch { stats.coinbaseErrors++; }
    }

    if (withinActiveHours(now)) {
      if (now - lastPoll >= RULES.activityPollMs) {
        lastPoll = now;
        stats.polls++;
        const t = await watcher.tick();
        guard.onHttpStatus(t.httpStatus);
        if (t.httpStatus === null || t.httpStatus >= 400) stats.pollErrors++;
        for (const o of t.newObservations) {
          stats.observations++;
          await write("observation", { ...o });
          for (const c of captureTimes(o.firstSeenMs)) pendingCaptures.push({ obs: o, ...c });
        }
      }
      // Tail sampling: once per second during the last 120 s of the active window.
      const meta = markets.get(a.slug);
      if (inSamplingTail(now, a)) {
        const snap = snapshotFor(meta);
        stats.tailSamples++;
        await write("tail_sample", { slug: a.slug, atMs: now, synced: snap !== null, bids: snap?.bids ?? null, asks: snap ? { Up: snap.asks.Up.slice(0, RULES.bookSampleLevels), Down: snap.asks.Down.slice(0, RULES.bookSampleLevels) } : null, mids: snap?.mids ?? null, spot: spot.at(now), spotTwap60: spot.twap(now), coinbase: spot.at(now, "coinbase") });
      }
    }

    // Due captures (also outside active hours, so a late-hour observation is still completed).
    for (let i = pendingCaptures.length - 1; i >= 0; i--) {
      const p = pendingCaptures[i];
      if (p.atMs > now) continue;
      pendingCaptures.splice(i, 1);
      const snap = snapshotFor(markets.get(p.obs.windowSlug));
      const cap: Capture & { lateMs: number } = {
        obsId: p.obs.obsId, delaySeconds: p.delaySeconds, capturedMs: now, lateMs: now - p.atMs,
        asks: snap?.asks ?? null, mids: snap?.mids ?? null,
        spotDirection: spotDirection(spot, now, p.obs.windowStartSec * 1000),
      };
      stats.captures++;
      await write("capture", cap as unknown as Record<string, unknown>);
    }

    // Resolutions, one hour after a window ends.
    for (const [slug, r] of resolutionsDue) {
      if (r.dueMs > now) continue;
      const res = await fetchPredictionJson(`https://gamma-api.polymarket.com/markets/${r.marketId}`);
      guard.onHttpStatus(res.httpStatus);
      const n = res.status === "OK" ? normalizeGammaResolution(res.body, new Date(now).toISOString()) : null;
      if (n?.ok) {
        await write("resolution", { slug, marketId: r.marketId, winner: n.value.winningLabel, closedTime: n.value.closedTime });
        resolutionsDue.delete(slug);
      } else r.dueMs = now + 15 * 60_000; // retry later; never guessed
    }

    // Drop markets more than two windows old once their captures are done.
    for (const slug of markets.keys()) {
      const m = markets.get(slug)!;
      if (m.window.endSec * 1000 < now - 2 * RULES.windowSeconds * 1000 && !pendingCaptures.some((p) => p.obs.windowSlug === slug)) markets.delete(slug);
    }

    if (now - lastHealth >= 60_000) {
      lastHealth = now;
      await write("health", { event: "HEALTH", stats: { ...stats }, bytes: rec.bytesWritten, spotFresh: spot.at(now) !== null });
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  try { wsRef.current?.close(); } catch { /* ignore */ }
  try { spotRef.current?.close(); } catch { /* ignore */ }
  await rec.write("health", { event: "STOP", reason: guard.stopped, stats });
  return { reason: guard.stopped, stats };
}

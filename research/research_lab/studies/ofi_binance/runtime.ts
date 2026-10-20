// Live collection loop: one combined Binance bookTicker websocket for all symbols, a 1 s flush
// timer, reconnect with exponential backoff, and hard stop rules. Public data only, no orders.

import { reconnectDelayMs } from "../../../niulai4_pilot/guards.ts";
import { parseBookTicker, SecondAggregator } from "./ofi.ts";
import { Recorder } from "./recorder.ts";
import { assertAllowed, RULES, RULES_VERSION } from "./rules.ts";

export type StopReason = "DURATION_ELAPSED" | "DISK_CAP" | "WS_UNSTABLE" | "OPERATOR";

export async function runCollection(opts: { outDir: string; rulesFingerprint: string; clockOffsetMs: number; stopSignal: { reason: StopReason | null } }) {
  const clock = () => Date.now() + opts.clockOffsetMs;
  const startedMs = clock();
  const rec = new Recorder({ dir: opts.outDir, capBytes: RULES.diskCapBytes, rulesVersion: RULES_VERSION, rulesFingerprint: opts.rulesFingerprint });
  const aggs = new Map(RULES.symbols.map((s) => [s, new SecondAggregator(s, RULES.maxSilenceMs)]));
  const stats = { messages: 0, invalid: 0, buckets: 0, reconnects: 0, gapsStarted: 0 };
  const reconnectTimes: number[] = [];
  const stop = (r: StopReason) => { opts.stopSignal.reason ??= r; };

  // Tick size per symbol (paper measures Delta P in ticks).
  const meta: Record<string, unknown>[] = [];
  try {
    const url = assertAllowed(`${RULES.restBase}/api/v3/exchangeInfo?symbols=${encodeURIComponent(JSON.stringify(RULES.symbols))}`);
    const info = await (await fetch(url)).json();
    for (const s of info.symbols ?? []) {
      const pf = (s.filters ?? []).find((f: { filterType: string }) => f.filterType === "PRICE_FILTER");
      meta.push({ symbol: s.symbol, tickSize: Number(pf?.tickSize), at: new Date(clock()).toISOString() });
    }
  } catch (e) {
    meta.push({ error: "EXCHANGE_INFO_UNAVAILABLE", detail: String(e).slice(0, 120) });
  }
  await rec.write("meta", "all", meta, clock());

  let ws: WebSocket | null = null;
  let attempt = 0;
  const streams = RULES.symbols.map((s) => `${s.toLowerCase()}@bookTicker`).join("/");
  const connect = () => {
    if (opts.stopSignal.reason) return;
    const url = assertAllowed(`${RULES.wsBase}?streams=${streams}`);
    const s = new WebSocket(url.toString());
    ws = s;
    s.onopen = () => { attempt = 0; };
    s.onmessage = (ev) => {
      stats.messages++;
      let msg: unknown;
      try { msg = JSON.parse(String(ev.data)); } catch { stats.invalid++; return; }
      const t = parseBookTicker(msg);
      if (!t) { stats.invalid++; return; }
      const agg = aggs.get(t.symbol);
      if (!agg) return;
      pending.push(...agg.push(t, clock()));
    };
    s.onclose = () => {
      if (ws !== s) return;
      for (const a of aggs.values()) a.reset(); // no e_n across a reconnect
      stats.reconnects++;
      stats.gapsStarted++;
      const now = clock();
      reconnectTimes.push(now);
      while (reconnectTimes.length && reconnectTimes[0] < now - 600_000) reconnectTimes.shift();
      if (reconnectTimes.length > RULES.maxReconnectsPer10Min) { stop("WS_UNSTABLE"); return; }
      setTimeout(connect, reconnectDelayMs(attempt++));
    };
    s.onerror = () => { /* onclose follows */ };
  };
  const pending: ReturnType<SecondAggregator["push"]> = [];
  connect();

  let lastHealth = clock();
  while (!opts.stopSignal.reason) {
    await new Promise((r) => setTimeout(r, 250));
    const now = clock();
    if (now - startedMs >= RULES.collectionDays * 86_400_000) stop("DURATION_ELAPSED");
    for (const a of aggs.values()) pending.push(...a.flush(now));
    if (pending.length) {
      const bySymbol = new Map<string, Record<string, unknown>[]>();
      for (const b of pending.splice(0)) {
        const list = bySymbol.get(b.symbol) ?? [];
        list.push(b as unknown as Record<string, unknown>);
        bySymbol.set(b.symbol, list);
      }
      for (const [sym, list] of bySymbol) {
        if (!(await rec.write("bucket", sym, list, now))) { stop("DISK_CAP"); break; }
        stats.buckets += list.length;
      }
    }
    if (now - lastHealth >= RULES.healthEveryMs) {
      lastHealth = now;
      await rec.write("health", "all", [{ at: new Date(now).toISOString(), ...stats, bytes: rec.bytesWritten }], now);
    }
  }
  try { (ws as WebSocket | null)?.close(); } catch { /* ignore */ }
  await rec.write("health", "all", [{ at: new Date(clock()).toISOString(), event: "STOP", reason: opts.stopSignal.reason, ...stats }], clock());
  return { reason: opts.stopSignal.reason, stats };
}

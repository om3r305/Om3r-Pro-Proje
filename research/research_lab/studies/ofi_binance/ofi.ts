// Pure OFI core (no I/O). Event contribution e_n exactly as defined in Cont, Kukanov & Stoikov
// (arXiv:1011.6402, section 2.1):
//   e_n = I{Pb_n >= Pb_{n-1}} qb_n - I{Pb_n <= Pb_{n-1}} qb_{n-1}
//       - I{Pa_n <= Pa_{n-1}} qa_n + I{Pa_n >= Pa_{n-1}} qa_{n-1}
// Each Binance bookTicker message is one event. After a (re)connect the first message only sets
// the state: no e_n is ever computed across a gap.

export interface Top {
  bid: number;
  bidQty: number;
  ask: number;
  askQty: number;
  updateId: number;
}

export function eventContribution(prev: Top, cur: Top): number {
  let e = 0;
  if (cur.bid >= prev.bid) e += cur.bidQty;
  if (cur.bid <= prev.bid) e -= prev.bidQty;
  if (cur.ask <= prev.ask) e -= cur.askQty;
  if (cur.ask >= prev.ask) e += prev.askQty;
  return e;
}

export function parseBookTicker(msg: unknown): (Top & { symbol: string }) | null {
  // Combined stream: {stream, data: {u, s, b, B, a, A}}
  const d = (msg as { data?: Record<string, unknown> })?.data ?? (msg as Record<string, unknown>);
  if (!d || typeof d !== "object") return null;
  const symbol = String((d as Record<string, unknown>).s ?? "");
  const top = {
    bid: Number((d as Record<string, unknown>).b),
    bidQty: Number((d as Record<string, unknown>).B),
    ask: Number((d as Record<string, unknown>).a),
    askQty: Number((d as Record<string, unknown>).A),
    updateId: Number((d as Record<string, unknown>).u),
  };
  if (!symbol || !Object.values(top).every(Number.isFinite)) return null;
  if (top.bid <= 0 || top.ask <= 0 || top.bid >= top.ask || top.bidQty < 0 || top.askQty < 0) return null; // crossed or invalid
  return { symbol, ...top };
}

export interface Bucket {
  symbol: string;
  sec: number; // unix second (start of the bucket)
  ofi: number;
  events: number;
  depthSum: number; // sum over events of (qb + qa)
  bid: number;
  ask: number;
  bidQty: number;
  askQty: number;
  mid: number;
  firstUpdateId: number | null;
  lastUpdateId: number | null;
  stateReset: boolean; // the book state was (re)initialised in this second
}

// One aggregator per symbol. push() for each message, flush() on a timer. Seconds are emitted
// only while the stream is alive (last message within maxSilenceMs); silent seconds are gaps.
export class SecondAggregator {
  private prev: Top | null = null;
  private cur: Bucket | null = null;
  private lastMsgMs: number | null = null;
  private resetPending = true;
  constructor(readonly symbol: string, private readonly maxSilenceMs: number) {}

  reset(): void {
    this.prev = null;
    this.resetPending = true;
  }

  private open(sec: number): Bucket {
    const p = this.prev!;
    return {
      symbol: this.symbol, sec, ofi: 0, events: 0, depthSum: 0,
      bid: p.bid, ask: p.ask, bidQty: p.bidQty, askQty: p.askQty, mid: (p.bid + p.ask) / 2,
      firstUpdateId: null, lastUpdateId: null, stateReset: false,
    };
  }

  // Close every whole second before `untilSec`; carry the state forward through quiet seconds
  // only while the stream is alive.
  private closeUntil(untilSec: number, nowMs: number, out: Bucket[]): void {
    if (!this.cur) return;
    while (this.cur.sec < untilSec) {
      const endMs = (this.cur.sec + 1) * 1000;
      if (this.lastMsgMs === null || endMs - this.lastMsgMs > this.maxSilenceMs || nowMs < endMs) {
        if (nowMs < endMs) return; // not finished yet
        // Silence: stop emitting and drop the state, so the next message only re-initialises it
        // (no e_n across a gap, exactly like a reconnect).
        this.cur = null;
        this.prev = null;
        this.resetPending = true;
        return;
      }
      out.push(this.cur);
      this.cur = this.prev ? this.open(this.cur.sec + 1) : null;
      if (!this.cur) return;
    }
  }

  push(t: Top, recvMs: number): Bucket[] {
    const out: Bucket[] = [];
    const sec = Math.floor(recvMs / 1000);
    if (this.cur && sec > this.cur.sec) this.closeUntil(sec, recvMs, out);
    if (this.prev === null) {
      this.prev = t;
      this.lastMsgMs = recvMs;
      this.cur = this.open(sec);
      this.cur.stateReset = this.resetPending;
      this.resetPending = false;
      this.cur.firstUpdateId = t.updateId;
      this.cur.lastUpdateId = t.updateId;
      return out;
    }
    if (!this.cur || this.cur.sec !== sec) {
      this.cur = this.open(sec);
    }
    if (t.updateId < this.prev.updateId) return out; // stale/out-of-order message: ignored
    const e = eventContribution(this.prev, t);
    this.cur.ofi += e;
    this.cur.events += 1;
    this.cur.depthSum += t.bidQty + t.askQty;
    this.cur.bid = t.bid; this.cur.ask = t.ask; this.cur.bidQty = t.bidQty; this.cur.askQty = t.askQty;
    this.cur.mid = (t.bid + t.ask) / 2;
    this.cur.firstUpdateId ??= t.updateId;
    this.cur.lastUpdateId = t.updateId;
    this.prev = t;
    this.lastMsgMs = recvMs;
    return out;
  }

  flush(nowMs: number): Bucket[] {
    const out: Bucket[] = [];
    this.closeUntil(Math.floor(nowMs / 1000), nowMs, out);
    return out;
  }
}

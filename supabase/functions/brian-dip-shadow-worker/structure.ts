import {
  type Bar,
  hash,
  mean,
  type Pivot,
  type Struct,
} from "../_shared/dip_v8.ts";
const LEFT = 3, RIGHT = 3, BREAK_BUFFER_ATR = .15;
// Inputs are validated closed candles; confirmation is never backdated to the pivot.
export function atr(rows: Bar[], p = 14) {
  const x: number[] = [];
  for (let i = Math.max(1, rows.length - p); i < rows.length; i++) {
    const r = rows[i], pc = rows[i - 1].c;
    x.push(Math.max(r.h - r.l, Math.abs(r.h - pc), Math.abs(r.l - pc)));
  }
  return mean(x);
}
export function classifyPivots(rows: Bar[]): Pivot[] {
  const out: Pivot[] = [];
  for (let i = LEFT; i < rows.length - RIGHT; i++) {
    let hi = true, lo = true;
    for (let j = 1; j <= LEFT; j++) {
      if (rows[i].h <= rows[i - j].h) hi = false;
      if (rows[i].l >= rows[i - j].l) lo = false;
    }
    for (let j = 1; j <= RIGHT; j++) {
      if (rows[i].h <= rows[i + j].h) hi = false;
      if (rows[i].l >= rows[i + j].l) lo = false;
    }
    if (hi) out.push({ i, t: rows[i].t, p: rows[i].h, kind: "H" });
    if (lo) out.push({ i, t: rows[i].t, p: rows[i].l, kind: "L" });
  }
  out.sort((a, b) => a.i - b.i);
  let ph: Pivot | null = null, pl: Pivot | null = null;
  for (const p of out) {
    if (p.kind === "H") {
      p.label = ph ? (p.p > ph.p ? "HH" : "LH") : "H";
      ph = p;
    } else {
      p.label = pl ? (p.p > pl.p ? "HL" : "LL") : "L";
      pl = p;
    }
  }
  return out;
}
export async function structure(tf: string, raw: Bar[]): Promise<Struct> {
  const rows = raw,
    a = atr(rows),
    pv = classifyPivots(rows),
    hs = pv.filter((x) => x.kind === "H"),
    ls = pv.filter((x) => x.kind === "L"),
    lh = hs.at(-1) || null,
    ll = ls.at(-1) || null,
    ph = hs.at(-2) || null,
    pl = ls.at(-2) || null,
    last = rows.at(-1)!,
    prev = rows.at(-2)!;
  let trend: "UP" | "DOWN" | "RANGE" = "RANGE";
  if (lh && ph && ll && pl) {
    if (lh.p > ph.p && ll.p > pl.p) trend = "UP";
    else if (lh.p < ph.p && ll.p < pl.p) trend = "DOWN";
  }
  let bos: "UP" | "DOWN" | null = null, choch: "UP" | "DOWN" | null = null;
  if (lh && last.c > lh.p + BREAK_BUFFER_ATR * a) bos = "UP";
  if (ll && last.c < ll.p - BREAK_BUFFER_ATR * a) bos = "DOWN";
  if (trend === "DOWN" && bos === "UP") choch = "UP";
  if (trend === "UP" && bos === "DOWN") choch = "DOWN";
  let sweep: "BULL" | "BEAR" | null = null;
  if (ll && last.l < ll.p && last.c > ll.p) sweep = "BULL";
  if (lh && last.h > lh.p && last.c < lh.p) sweep = "BEAR";
  let failedBreak: "BULL" | "BEAR" | null = null;
  if (ll && prev.c < ll.p && last.c > ll.p) failedBreak = "BULL";
  if (lh && prev.c > lh.p && last.c < lh.p) failedBreak = "BEAR";
  let equalHigh: number | null = null, equalLow: number | null = null;
  for (let i = hs.length - 1; i > 0; i--) {
    if (Math.abs(hs[i].p - hs[i - 1].p) <= a * .15) {
      equalHigh = (hs[i].p + hs[i - 1].p) / 2;
      break;
    }
  }
  for (let i = ls.length - 1; i > 0; i--) {
    if (Math.abs(ls[i].p - ls[i - 1].p) <= a * .15) {
      equalLow = (ls[i].p + ls[i - 1].p) / 2;
      break;
    }
  }
  const fingerprint = await hash(
    [
      tf,
      lh?.t,
      lh?.p,
      ll?.t,
      ll?.p,
      trend,
      bos,
      choch,
      sweep,
      failedBreak,
      equalHigh,
      equalLow,
    ].join("|"),
  );
  return {
    tf,
    lastClose: last.c,
    atr: a,
    pivots: pv.slice(-12),
    lastHigh: lh,
    lastLow: ll,
    trend,
    bos,
    choch,
    sweep,
    failedBreak,
    equalHigh,
    equalLow,
    fingerprint,
  };
}

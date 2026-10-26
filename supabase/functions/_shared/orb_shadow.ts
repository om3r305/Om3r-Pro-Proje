// Opening-range breakout (ORB) shadow study `orb-shadow-v1` (pure, no I/O).
// Idea: the opening-range breakout rules of the MIT-licensed day-trader-skill pack
// (github.com/Jakeschincariol/day-trader-skill, trader-entry/trader-exit); re-implemented here, no code
// copied. Pre-registered in research/research_lab/studies/orb_shadow/PREREGISTRATION.md.
//
// Per symbol and New York session day, from 5-minute bars (bar time = bar start):
//  * opening range = the 09:30, 09:35 and 09:40 bars (all three required): high, low, volume;
//  * trigger = first bar from 09:45 up to the 15:50 bar whose CLOSE is above the range high (long) or
//    below the range low (short); entry at that close; stop = range midpoint;
//  * A ORB_1R: target = entry +- 1R; B ORB_CLOSE: no target; both flatten at the 15:55 bar close;
//    C ORB_IN_PLAY_1R: A, only if the opening-range volume is >= 2x the median of the previous
//    10 sessions' opening-range volume (needs 10 prior sessions);
//  * fills are conservative: a bar that touches both stop and target exits at the stop; a bar that
//    opens beyond the stop exits at its open; a target is never filled better than the target.
// Net = gross - the engine's own round-trip cost for the asset class. Shadow only; nothing trades.

export const ORB_RULES_VERSION = "orb-shadow.v1";
export const ORB_RULES = Object.freeze({
  openingBars: ["09:30", "09:35", "09:40"],
  firstTriggerBar: "09:45",
  lastTriggerBar: "15:50",
  flattenBar: "15:55",
  targetR: 1,
  inPlayVolumeMultiple: 2,
  inPlayLookbackSessions: 10,
  costBps: { equity: 18, etf: 12 } as Record<string, number>,
});
export type OrbVariant = "ORB_1R" | "ORB_CLOSE" | "ORB_IN_PLAY_1R";

export interface Bar { t: number; o: number; h: number; l: number; c: number; v: number } // t = unix seconds (bar start)
export interface OrbResult {
  sessionDate: string;
  variant: OrbVariant;
  status: "TRADE" | "NO_TRIGGER" | "NOT_IN_PLAY" | "INSUFFICIENT_DATA";
  direction: -1 | 1 | null;
  orHigh: number | null; orLow: number | null; orVolume: number | null; relVolume: number | null;
  entryAt: string | null; entryPrice: number | null; stopPrice: number | null; targetPrice: number | null;
  exitAt: string | null; exitPrice: number | null; exitReason: "TARGET" | "STOP" | "TIME" | null;
  grossBps: number | null; costBps: number; netBps: number | null;
}

const fmt = (tz: string) => new Intl.DateTimeFormat("en-CA", { timeZone: tz, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" });
function local(t: number, f: Intl.DateTimeFormat): { date: string; hm: string } {
  const p = Object.fromEntries(f.formatToParts(new Date(t * 1000)).map((x) => [x.type, x.value]));
  return { date: `${p.year}-${p.month}-${p.day}`, hm: `${p.hour}:${p.minute}` };
}
const ok = (b: Bar) => [b.o, b.h, b.l, b.c].every((x) => Number.isFinite(x) && x > 0) && Number.isFinite(b.v) && b.v >= 0;

export function groupSessions(bars: Bar[], tz = "America/New_York"): Map<string, Map<string, Bar>> {
  const f = fmt(tz);
  const out = new Map<string, Map<string, Bar>>();
  for (const b of bars) {
    if (!ok(b)) continue;
    const { date, hm } = local(b.t, f);
    if (hm < "09:30" || hm > "15:55") continue;
    const day = out.get(date) ?? new Map<string, Bar>();
    day.set(hm, b);
    out.set(date, day);
  }
  return out;
}

function median(xs: number[]): number { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; }

function simulate(day: Map<string, Bar>, withTarget: boolean) {
  const or = ORB_RULES.openingBars.map((k) => day.get(k));
  if (or.some((b) => !b) || !day.get(ORB_RULES.flattenBar)) return null;
  const orHigh = Math.max(...or.map((b) => b!.h)), orLow = Math.min(...or.map((b) => b!.l));
  const orVolume = or.reduce((s, b) => s + b!.v, 0);
  const times = [...day.keys()].sort();
  let entry: { dir: -1 | 1; price: number; idx: number; t: number } | null = null;
  for (let i = 0; i < times.length; i++) {
    const hm = times[i];
    if (hm < ORB_RULES.firstTriggerBar || hm > ORB_RULES.lastTriggerBar) continue;
    const b = day.get(hm)!;
    if (b.c > orHigh) { entry = { dir: 1, price: b.c, idx: i, t: b.t + 300 }; break; }
    if (b.c < orLow) { entry = { dir: -1, price: b.c, idx: i, t: b.t + 300 }; break; }
  }
  const base = { orHigh, orLow, orVolume };
  if (!entry) return { ...base, entry: null };
  const stop = (orHigh + orLow) / 2;
  const risk = Math.abs(entry.price - stop);
  if (!(risk > 0)) return { ...base, entry: null };
  const target = withTarget ? entry.price + entry.dir * ORB_RULES.targetR * risk : null;
  let exit: { price: number; t: number; reason: "TARGET" | "STOP" | "TIME" } | null = null;
  for (let i = entry.idx + 1; i < times.length && !exit; i++) {
    const hm = times[i], b = day.get(hm)!;
    const stopHit = entry.dir === 1 ? b.l <= stop : b.h >= stop;
    const gapStop = entry.dir === 1 ? b.o <= stop : b.o >= stop;
    const targetHit = target != null && (entry.dir === 1 ? b.h >= target : b.l <= target);
    if (stopHit) exit = { price: gapStop ? b.o : stop, t: b.t, reason: "STOP" };
    else if (targetHit) exit = { price: target!, t: b.t, reason: "TARGET" };
    else if (hm === ORB_RULES.flattenBar) exit = { price: b.c, t: b.t + 300, reason: "TIME" };
  }
  if (!exit) { const last = day.get(ORB_RULES.flattenBar)!; exit = { price: last.c, t: last.t + 300, reason: "TIME" }; }
  return { ...base, entry, stop, target, exit };
}

/** All variants for every complete session in the bars, oldest first. */
export function orbResults(bars: Bar[], assetClass: string, tz = "America/New_York"): OrbResult[] {
  const sessions = groupSessions(bars, tz);
  const dates = [...sessions.keys()].sort();
  const costBps = ORB_RULES.costBps[assetClass] ?? ORB_RULES.costBps.equity;
  const orVolumes: number[] = [];
  const results: OrbResult[] = [];
  const iso = (t: number) => new Date(t * 1000).toISOString();
  for (const date of dates) {
    const day = sessions.get(date)!;
    const prior = orVolumes.slice(-ORB_RULES.inPlayLookbackSessions);
    for (const variant of ["ORB_1R", "ORB_CLOSE", "ORB_IN_PLAY_1R"] as OrbVariant[]) {
      const sim = simulate(day, variant !== "ORB_CLOSE");
      const empty: OrbResult = { sessionDate: date, variant, status: "INSUFFICIENT_DATA", direction: null, orHigh: null, orLow: null,
        orVolume: null, relVolume: null, entryAt: null, entryPrice: null, stopPrice: null, targetPrice: null, exitAt: null,
        exitPrice: null, exitReason: null, grossBps: null, costBps, netBps: null };
      if (!sim) { results.push(empty); continue; }
      const relVolume = prior.length >= ORB_RULES.inPlayLookbackSessions ? sim.orVolume / median(prior) : null;
      const r: OrbResult = { ...empty, orHigh: sim.orHigh, orLow: sim.orLow, orVolume: sim.orVolume, relVolume };
      if (variant === "ORB_IN_PLAY_1R") {
        if (relVolume == null) { results.push(r); continue; }
        if (relVolume < ORB_RULES.inPlayVolumeMultiple) { results.push({ ...r, status: "NOT_IN_PLAY" }); continue; }
      }
      if (!sim.entry) { results.push({ ...r, status: "NO_TRIGGER" }); continue; }
      const gross = sim.entry.dir * (sim.exit!.price / sim.entry.price - 1) * 10000;
      results.push({ ...r, status: "TRADE", direction: sim.entry.dir, entryAt: iso(sim.entry.t), entryPrice: sim.entry.price,
        stopPrice: sim.stop!, targetPrice: sim.target ?? null, exitAt: iso(sim.exit!.t), exitPrice: sim.exit!.price,
        exitReason: sim.exit!.reason, grossBps: gross, netBps: gross - costBps });
    }
    const s = simulate(day, true);
    if (s) orVolumes.push(s.orVolume);
  }
  return results;
}

/** Parse a Yahoo chart v8 payload into bars (rows with any missing field are dropped). */
export function parseYahooBars(payload: unknown): { bars: Bar[]; tz: string } {
  type Series = (number | null)[] | undefined;
  type Chart = { timestamp?: number[]; meta?: { exchangeTimezoneName?: string };
    indicators?: { quote?: { open?: Series; high?: Series; low?: Series; close?: Series; volume?: Series }[] } };
  const r = (payload as { chart?: { result?: Chart[] } })?.chart?.result?.[0];
  const ts: number[] = Array.isArray(r?.timestamp) ? r!.timestamp : [];
  const q = r?.indicators?.quote?.[0] ?? {};
  const bars: Bar[] = [];
  for (let i = 0; i < ts.length; i++) {
    const b = { t: Number(ts[i]), o: Number(q.open?.[i]), h: Number(q.high?.[i]), l: Number(q.low?.[i]), c: Number(q.close?.[i]), v: Number(q.volume?.[i]) };
    if (q.open?.[i] != null && q.close?.[i] != null && ok(b)) bars.push(b);
  }
  return { bars, tz: String(r?.meta?.exchangeTimezoneName ?? "America/New_York") };
}

export async function orbRulesFingerprint(): Promise<string> {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${ORB_RULES_VERSION}|${JSON.stringify(ORB_RULES)}`)));
  return [...d].map((b) => b.toString(16).padStart(2, "0")).join("");
}

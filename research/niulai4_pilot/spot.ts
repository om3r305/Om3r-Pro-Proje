// Spot reference (pure). The market resolves on the Chainlink BTC/USD 60s TWAP stream, which is
// not freely accessible in real time; the pilot therefore uses a SPOT PROXY and measures how often
// the proxy agrees with the actual resolution. The proxy is never presented as the oracle.

export interface SpotTick {
  tMs: number; // local receive time (clock-skew corrected by the caller)
  mid: number;
  source: "binance" | "coinbase";
}

export class SpotSeries {
  private ticks: SpotTick[] = [];
  constructor(private readonly keepMs = 15 * 60_000) {}

  add(t: SpotTick): void {
    if (!Number.isFinite(t.mid) || t.mid <= 0 || !Number.isFinite(t.tMs)) return;
    this.ticks.push(t);
    const cutoff = t.tMs - this.keepMs;
    while (this.ticks.length && this.ticks[0].tMs < cutoff) this.ticks.shift();
  }

  // Last known value at or before tMs (point-in-time), from one source.
  at(tMs: number, source: SpotTick["source"] = "binance"): number | null {
    for (let i = this.ticks.length - 1; i >= 0; i--) {
      const x = this.ticks[i];
      if (x.source === source && x.tMs <= tMs) return x.mid;
    }
    return null;
  }

  // Time-weighted average over (tMs - windowMs, tMs], each tick holding until the next one.
  // Returns null when the window has no coverage at its start (no tick at or before it).
  twap(tMs: number, windowMs = 60_000, source: SpotTick["source"] = "binance"): number | null {
    const from = tMs - windowMs;
    const xs = this.ticks.filter((x) => x.source === source && x.tMs <= tMs);
    const startIdx = xs.findLastIndex((x) => x.tMs <= from);
    if (startIdx < 0) return null;
    let area = 0;
    let prevT = from;
    let prevV = xs[startIdx].mid;
    for (let i = startIdx + 1; i < xs.length; i++) {
      area += prevV * (xs[i].tMs - prevT);
      prevT = xs[i].tMs;
      prevV = xs[i].mid;
    }
    area += prevV * (tMs - prevT);
    return area / windowMs;
  }
}

// B1 direction at a decision time: compare the spot TWAP proxy now with spot at the window start.
// null when either value is unknown -- the baseline then makes no trade for that observation.
export function spotDirection(series: SpotSeries, decisionMs: number, windowStartMs: number): "Up" | "Down" | null {
  const open = series.at(windowStartMs);
  const now = series.twap(decisionMs);
  if (open === null || now === null) return null;
  return now >= open ? "Up" : "Down";
}

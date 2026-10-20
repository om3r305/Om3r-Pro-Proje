// Public activity watcher (pure-ish: fetch and clock injected). Polls the wallet's public TRADE
// activity and emits each new trade exactly once with Brian's own first-seen time. Only BTC 5m
// Up/Down trades become observations; everything else is counted and ignored.

import { fetchPredictionJson, normalizeWalletActivityRow, type PredictionFetchOptions } from "../../supabase/functions/_shared/prediction_market.ts";
import { type Observation, type Outcome, scheduledCaptureMs } from "./evaluate.ts";
import { RULES } from "./rules.ts";
import { windowFromSlug } from "./windows.ts";

export interface WatchTick {
  httpStatus: number | null;
  newObservations: Observation[];
  ignoredOtherMarkets: number;
  rejectedRows: number;
}

export class ActivityWatcher {
  private seen = new Set<string>();
  private primed = false;
  constructor(
    private readonly wallet: string,
    private readonly clock: () => number, // skew-corrected
    private readonly options: PredictionFetchOptions = {},
  ) {}

  // The first poll only learns what already exists: trades that happened before the pilot started
  // have no measurable first-seen time and must not become observations.
  async tick(): Promise<WatchTick> {
    const url = `https://data-api.polymarket.com/activity?user=${this.wallet}&type=TRADE&limit=100`;
    const res = await fetchPredictionJson<unknown[]>(url, { maxAttempts: 2, ...this.options });
    const out: WatchTick = { httpStatus: res.httpStatus, newObservations: [], ignoredOtherMarkets: 0, rejectedRows: 0 };
    if (res.status !== "OK" || !Array.isArray(res.body)) return out;
    const seenAt = this.clock();
    for (const raw of res.body) {
      const n = normalizeWalletActivityRow(raw, new Date(seenAt).toISOString());
      if (!n.ok) {
        out.rejectedRows++;
        continue;
      }
      const t = n.value;
      if (this.seen.has(t.idempotencyKey)) continue;
      this.seen.add(t.idempotencyKey);
      if (!this.primed) continue;
      if (t.wallet !== this.wallet.toLowerCase() || t.side !== "BUY") {
        out.ignoredOtherMarkets++;
        continue;
      }
      const w = t.marketSlug ? windowFromSlug(t.marketSlug) : null;
      const outcome = t.outcome === "Up" || t.outcome === "Down" ? (t.outcome as Outcome) : null;
      if (!w || !outcome) {
        out.ignoredOtherMarkets++;
        continue;
      }
      out.newObservations.push({
        obsId: t.idempotencyKey,
        windowSlug: w.slug,
        windowStartSec: w.startSec,
        windowEndSec: w.endSec,
        outcome,
        tradeTsMs: Date.parse(t.tradedAt),
        firstSeenMs: seenAt,
      });
    }
    this.primed = true;
    return out;
  }
}

// Capture instants: decision at first_seen + d, book read when the order would arrive
// (+ orderLatencyMs). Scheduled from first_seen only -- never from later information.
export function captureTimes(firstSeenMs: number): { delaySeconds: number; atMs: number }[] {
  return RULES.delaysSeconds.map((d) => ({ delaySeconds: d, atMs: scheduledCaptureMs(firstSeenMs, d) }));
}

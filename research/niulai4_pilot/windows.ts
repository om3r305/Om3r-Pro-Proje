// 5-minute window arithmetic. A window is identified by its start epoch (seconds), exactly as in
// the Polymarket slug `btc-updown-5m-<start>`. The slug is only a lookup key: the authoritative
// start/end come from the market's eventStartTime/endDate, and the pilot cross-checks them.

import { RULES } from "./rules.ts";

export interface WindowRef {
  startSec: number;
  endSec: number;
  slug: string;
}

export function windowAt(epochMs: number, windowSeconds = RULES.windowSeconds, prefix = RULES.slugPrefix): WindowRef {
  const startSec = Math.floor(epochMs / 1000 / windowSeconds) * windowSeconds;
  return { startSec, endSec: startSec + windowSeconds, slug: `${prefix}${startSec}` };
}

export function activeAndNext(epochMs: number): [WindowRef, WindowRef] {
  const a = windowAt(epochMs);
  return [a, windowAt((a.endSec) * 1000)];
}

export function windowFromSlug(slug: string, prefix = RULES.slugPrefix, windowSeconds = RULES.windowSeconds): WindowRef | null {
  if (!slug.startsWith(prefix)) return null;
  const rest = slug.slice(prefix.length);
  if (!/^[0-9]{9,11}$/.test(rest)) return null;
  const startSec = Number(rest);
  if (startSec % windowSeconds !== 0) return null;
  return { startSec, endSec: startSec + windowSeconds, slug };
}

export function withinActiveHours(epochMs: number, hours = RULES.activeHoursUtc): boolean {
  const h = new Date(epochMs).getUTCHours();
  return h >= hours.start && h < hours.end;
}

export function inSamplingTail(epochMs: number, w: WindowRef, tailSeconds = RULES.sampleLastSecondsOfWindow): boolean {
  const t = epochMs / 1000;
  return t >= w.endSec - tailSeconds && t < w.endSec;
}

// Cross-check the slug-derived window against the market's own fields. A mismatch means the
// window cannot be trusted and the market is skipped (recorded, never guessed).
export function windowMatchesMarket(w: WindowRef, eventStartTime: unknown, endDate: unknown): boolean {
  const s = typeof eventStartTime === "string" ? Date.parse(eventStartTime) : NaN;
  const e = typeof endDate === "string" ? Date.parse(endDate) : NaN;
  return Number.isFinite(s) && Number.isFinite(e) && s / 1000 === w.startSec && e / 1000 === w.endSec;
}

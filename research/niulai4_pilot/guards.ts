// Stop rules and start preconditions (pure state machine + small helpers).

import { RULES } from "./rules.ts";

export type StopReason =
  | "DURATION_ELAPSED"
  | "DISK_CAP"
  | "ACCESS_DENIED"
  | "RATE_LIMITED"
  | "IDENTITY_UNVERIFIED"
  | "WS_UNSTABLE"
  | "OPERATOR";

export class StopGuard {
  private reason: StopReason | null = null;
  private consecutiveRateLimited = 0;
  private wsReconnects: number[] = [];
  constructor(private readonly startedMs: number, private readonly now: () => number = Date.now) {}

  get stopped(): StopReason | null {
    return this.reason;
  }

  stop(reason: StopReason): void {
    this.reason ??= reason; // the first reason wins and is never overwritten
  }

  checkDuration(): void {
    if (this.now() - this.startedMs >= RULES.pilotDays * 86_400_000) this.stop("DURATION_ELAPSED");
  }

  checkDisk(bytesWritten: number): void {
    if (bytesWritten >= RULES.diskCapBytes) this.stop("DISK_CAP");
  }

  // HTTP outcome of any pilot request. 401/403/451: access refused -> stop for good, never retry
  // around it. Repeated 429 after the client's own backoff -> stop.
  onHttpStatus(status: number | null): void {
    if (status === 401 || status === 403 || status === 451) {
      this.stop("ACCESS_DENIED");
      return;
    }
    if (status === 429) {
      this.consecutiveRateLimited++;
      if (this.consecutiveRateLimited >= RULES.maxConsecutiveRateLimited) this.stop("RATE_LIMITED");
      return;
    }
    if (status !== null && status >= 200 && status < 300) this.consecutiveRateLimited = 0;
  }

  onWsReconnect(): void {
    const t = this.now();
    this.wsReconnects = this.wsReconnects.filter((x) => t - x < 10 * 60_000);
    this.wsReconnects.push(t);
    if (this.wsReconnects.length > RULES.wsMaxReconnectsPer10Min) this.stop("WS_UNSTABLE");
  }

  onIdentity(status: string): void {
    if (status !== "VERIFIED" && status !== "UNREACHABLE") this.stop("IDENTITY_UNVERIFIED");
  }
}

// Exponential reconnect delay 1 s -> 60 s.
export function reconnectDelayMs(attempt: number): number {
  return Math.min(60_000, 1000 * 2 ** Math.max(0, attempt));
}

// Clock offset estimate against a millisecond server clock (Binance /api/v3/time):
// offset = serverTime - (localSend + rtt/2). |offset| + rtt/2 bounds the true skew.
export interface SkewSample {
  localSendMs: number;
  localRecvMs: number;
  serverMs: number;
}
export function estimateSkew(samples: SkewSample[]): { offsetMs: number; uncertaintyMs: number } | null {
  const ok = samples.filter((s) => s.localRecvMs >= s.localSendMs && Number.isFinite(s.serverMs));
  if (!ok.length) return null;
  // Use the lowest-RTT sample: it bounds the offset most tightly.
  const best = ok.reduce((a, b) => (b.localRecvMs - b.localSendMs < a.localRecvMs - a.localSendMs ? b : a));
  const rtt = best.localRecvMs - best.localSendMs;
  return { offsetMs: Math.round(best.serverMs - (best.localSendMs + rtt / 2)), uncertaintyMs: Math.round(rtt / 2) };
}

export interface Preflight {
  ok: boolean;
  checks: { name: string; ok: boolean; detail: string }[];
}

export function evaluatePreflight(input: {
  skew: { offsetMs: number; uncertaintyMs: number } | null;
  freeDiskBytes: number | null;
  identity: string;
  rulesFingerprint: string;
  approvedFingerprint: string | null;
}): Preflight {
  const checks: Preflight["checks"] = [];
  const skewBound = input.skew ? Math.abs(input.skew.offsetMs) + input.skew.uncertaintyMs : null;
  checks.push({
    name: "clock_skew",
    // The offset is applied as a correction, so what must be small is the UNCERTAINTY of it.
    ok: input.skew !== null && input.skew.uncertaintyMs <= RULES.maxClockSkewMs,
    detail: input.skew ? `offset ${input.skew.offsetMs} ms ± ${input.skew.uncertaintyMs} ms (bound ${skewBound} ms)` : "no time sample",
  });
  checks.push({
    name: "free_disk",
    ok: input.freeDiskBytes !== null && input.freeDiskBytes >= 1024 ** 3,
    detail: input.freeDiskBytes === null ? "unknown" : `${Math.round(input.freeDiskBytes / 1024 ** 2)} MB free`,
  });
  checks.push({ name: "identity", ok: input.identity === "VERIFIED", detail: input.identity });
  checks.push({
    name: "rules_approved",
    ok: input.approvedFingerprint !== null && input.approvedFingerprint === input.rulesFingerprint,
    detail: input.approvedFingerprint === null ? "no approval file" : input.approvedFingerprint === input.rulesFingerprint ? "matches" : "approval is for different rules",
  });
  return { ok: checks.every((c) => c.ok), checks };
}

// Parse `w32tm /stripchart /dataonly` output ("09:09:39, +00.2024137s") from one or more NTP
// servers. The sign follows w32tm: positive = the reference clock is ahead of the local clock,
// i.e. the correction to ADD to local time. Uncertainty = half the spread of all samples, plus
// 1 ms for the printed precision. At least 3 samples are required.
export function parseW32tmOffsets(outputs: string[]): { offsetMs: number; uncertaintyMs: number } | null {
  const xs: number[] = [];
  for (const out of outputs) {
    for (const m of out.matchAll(/,\s*([+-])(\d+)\.(\d+)s/g)) {
      const v = Number(`${m[2]}.${m[3]}`) * 1000 * (m[1] === "-" ? -1 : 1);
      if (Number.isFinite(v)) xs.push(v);
    }
  }
  if (xs.length < 3) return null;
  const sorted = [...xs].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  return { offsetMs: Math.round(median), uncertaintyMs: Math.ceil((sorted.at(-1)! - sorted[0]) / 2 + 1) };
}

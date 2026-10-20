// Optional CORE sink: pushes the pilot's own health and interim/final summaries to
// brian-niulai4-pilot-ingest so the UI can show real pilot state. OFF unless a key file is given
// (--core-sink-key-file). The sink is strictly secondary:
//   * it never stops, slows or alters the measurement (failures are counted, not thrown);
//   * its HTTP statuses never reach StopGuard (a refused sink is not a refused data source);
//   * the key is read from a local file, sent only in the x-brian-pilot-key header to the one
//     allow-listed CORE host, and never logged or written to the recorder.

export const CORE_SINK_URL = "https://qbcjuxhvhwagvqbjyemo.supabase.co/functions/v1/brian-niulai4-pilot-ingest";

export interface SinkStats {
  sent: number;
  stored: number;
  rejected: number;
  rateLimited: number;
  failed: number;
  lastStatus: string | null;
  lastError: string | null;
}

type FetchFn = (url: string, init: RequestInit) => Promise<Response>;

export class CoreSink {
  readonly stats: SinkStats = { sent: 0, stored: 0, rejected: 0, rateLimited: 0, failed: 0, lastStatus: null, lastError: null };
  private seq = 0;
  constructor(
    private readonly key: string,
    readonly runId: string,
    private readonly rulesVersion: string,
    private readonly rulesFingerprint: string,
    private readonly clock: () => number,
    private readonly fetchFn: FetchFn = (u, i) => fetch(u, i),
    private readonly url: string = CORE_SINK_URL,
  ) {
    // Locked to the one ingest URL; CORE is not on the market-data allow-list on purpose.
    if (url !== CORE_SINK_URL) throw new Error("core sink url is fixed");
    if (key.length < 32) throw new Error("core sink key must be at least 32 characters");
  }

  private base(kind: string): Record<string, unknown> {
    return {
      kind,
      run_id: this.runId,
      observed_at: new Date(this.clock()).toISOString(),
      rules_version: this.rulesVersion,
      rules_fingerprint: this.rulesFingerprint,
      shadow_only: true,
      trade_signal: false,
    };
  }

  private async send(payload: Record<string, unknown>): Promise<string> {
    this.stats.sent++;
    try {
      const res = await this.fetchFn(this.url, {
        method: "POST",
        headers: { "content-type": "application/json", "x-brian-pilot-key": this.key },
        body: JSON.stringify(payload),
        signal: AbortSignal.timeout(10_000),
      });
      let body: Record<string, unknown> = {};
      try { body = await res.json(); } catch { /* non-JSON */ }
      const status = String(body.status ?? body.error ?? `HTTP_${res.status}`);
      this.stats.lastStatus = status;
      if (status === "STORED" || status === "DUPLICATE") this.stats.stored++;
      else if (status === "RATE_LIMITED") this.stats.rateLimited++;
      else {
        this.stats.rejected++;
        this.stats.lastError = String(body.reason ?? body.error ?? status).slice(0, 80);
      }
      return status;
    } catch (e) {
      this.stats.failed++;
      this.stats.lastStatus = "NETWORK_ERROR";
      this.stats.lastError = ((e as Error)?.name ?? "Error").slice(0, 80);
      return "NETWORK_ERROR";
    }
  }

  start(hostLabel: string | null): Promise<string> {
    return this.send({ ...this.base("START"), host_label: hostLabel });
  }

  heartbeat(health: Record<string, unknown>): Promise<string> {
    return this.send({ ...this.base("HEARTBEAT"), seq: this.seq++, health: { ...health, sink: { ...this.stats } } });
  }

  report(summary: Record<string, unknown>, kind: "INTERIM" | "FINAL"): Promise<string> {
    return this.send({ ...this.base("REPORT"), seq: this.seq++, report_kind: kind, summary });
  }

  stop(reason: string): Promise<string> {
    return this.send({ ...this.base("STOP"), stop_reason: reason });
  }
}

// Key file: one line, at least 32 characters. Returns null (sink off) when no path is given.
export async function readSinkKey(path: string | null): Promise<string | null> {
  if (!path) return null;
  const key = (await Deno.readTextFile(path)).trim();
  if (key.length < 32 || /\s/.test(key)) throw new Error("core sink key file must contain one key of at least 32 characters");
  return key;
}

export function newRunId(nowMs: number, rand: () => number = Math.random): string {
  const ts = new Date(nowMs).toISOString().replace(/[-:]/g, "").replace(/\..*$/, "");
  return `niulai4-${ts}-${Math.floor(rand() * 0xffffff).toString(16).padStart(6, "0")}`;
}

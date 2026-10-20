// Local append-only JSONL recorder with a hard byte cap. One file per record kind per UTC day.
// Every file starts with a header line carrying the rules version and fingerprint.

export interface RecorderOptions {
  dir: string;
  capBytes: number;
  rulesVersion: string;
  rulesFingerprint: string;
  now?: () => number;
}

export type RecordKind = "observation" | "capture" | "tail_sample" | "window" | "health" | "resolution";

export class Recorder {
  private written = 0;
  private headers = new Set<string>();
  private readonly enc = new TextEncoder();
  constructor(private readonly o: RecorderOptions) {}

  get bytesWritten(): number {
    return this.written;
  }

  // Returns false (and writes nothing) once the cap would be exceeded.
  async write(kind: RecordKind, record: Record<string, unknown>): Promise<boolean> {
    const now = (this.o.now ?? Date.now)();
    const day = new Date(now).toISOString().slice(0, 10);
    const path = `${this.o.dir}/${day}.${kind}.jsonl`;
    let chunk = "";
    if (!this.headers.has(path)) {
      chunk += JSON.stringify({ header: true, kind, rulesVersion: this.o.rulesVersion, rulesFingerprint: this.o.rulesFingerprint, createdAt: new Date(now).toISOString() }) + "\n";
    }
    chunk += JSON.stringify({ ...record, recordedAt: new Date(now).toISOString() }) + "\n";
    const bytes = this.enc.encode(chunk);
    if (this.written + bytes.length > this.o.capBytes) return false;
    await Deno.mkdir(this.o.dir, { recursive: true });
    await Deno.writeFile(path, bytes, { append: true, create: true });
    this.headers.add(path);
    this.written += bytes.length;
    return true;
  }
}

export async function readJsonl(path: string): Promise<Record<string, unknown>[]> {
  const text = await Deno.readTextFile(path);
  return text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l)).filter((r) => !r.header);
}

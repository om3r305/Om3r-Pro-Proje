// Local append-only JSONL recorder with a hard byte cap: one file per kind per symbol per UTC day,
// each starting with a header line that carries the rules version and fingerprint.

export interface RecorderOptions {
  dir: string;
  capBytes: number;
  rulesVersion: string;
  rulesFingerprint: string;
}

export class Recorder {
  private written = 0;
  private headers = new Set<string>();
  private readonly enc = new TextEncoder();
  constructor(private readonly o: RecorderOptions) {}

  get bytesWritten(): number {
    return this.written;
  }

  // Returns false (and writes nothing) once the cap would be exceeded.
  async write(kind: "bucket" | "health" | "meta", key: string, records: Record<string, unknown>[], nowMs: number): Promise<boolean> {
    if (!records.length) return true;
    const day = new Date(nowMs).toISOString().slice(0, 10);
    const path = `${this.o.dir}/${day}.${key}.${kind}.jsonl`;
    let chunk = "";
    if (!this.headers.has(path)) {
      chunk += JSON.stringify({ header: true, kind, key, rulesVersion: this.o.rulesVersion, rulesFingerprint: this.o.rulesFingerprint, createdAt: new Date(nowMs).toISOString() }) + "\n";
    }
    for (const r of records) chunk += JSON.stringify(r) + "\n";
    const bytes = this.enc.encode(chunk);
    if (this.written + bytes.length > this.o.capBytes) return false;
    await Deno.mkdir(this.o.dir, { recursive: true });
    await Deno.writeFile(path, bytes, { append: true, create: true });
    this.headers.add(path);
    this.written += bytes.length;
    return true;
  }
}

// NIULAI4 pilot entry point. Three modes:
//   --selftest            preflight only (clock skew, disk, identity, rules fingerprint); records nothing
//   --start --out <dir> [--core-sink-key-file <path>] [--host-label <name>]
//                         run the pilot; REFUSES unless <dir>/APPROVED_RULES_FINGERPRINT equals the
//                         current rules fingerprint (written only after the operator approves).
//                         With a key file, health (60 s) and interim summaries (30 min) are also
//                         pushed to CORE (sink.ts); without it the pilot is local-only.
//   --report --out <dir> [--final --required-windows N]
//                         offline evaluation -> <dir>/summary.json; interim (no verdict) unless --final
//
// deno run --allow-net --allow-read --allow-write --allow-run=powershell,w32tm research/niulai4_pilot/main.ts --selftest

import { estimateSkew, evaluatePreflight, parseW32tmOffsets, type SkewSample, StopGuard } from "./guards.ts";
import { evaluate, type Capture, type Observation, type Outcome, summarize, type WindowInfo } from "./evaluate.ts";
import { readJsonl } from "./recorder.ts";
import { RULES, RULES_VERSION, rulesFingerprint } from "./rules.ts";
import { runPilot, verifyIdentityNow } from "./runtime.ts";
import { CoreSink, newRunId, readSinkKey } from "./sink.ts";

async function skewSamples(n = 5): Promise<SkewSample[]> {
  const out: SkewSample[] = [];
  for (let i = 0; i < n; i++) {
    const localSendMs = Date.now();
    try {
      const r = await fetch("https://api.binance.com/api/v3/time");
      const localRecvMs = Date.now();
      const body = await r.json();
      out.push({ localSendMs, localRecvMs, serverMs: Number(body.serverTime) });
    } catch { /* sample lost */ }
    await new Promise((r) => setTimeout(r, 200));
  }
  return out;
}

// Primary clock reference: Windows' built-in NTP client against two independent public servers.
// Fallback: Binance server time (coarser, bounded by half the HTTP round trip).
async function ntpSkew(): Promise<{ offsetMs: number; uncertaintyMs: number } | null> {
  const outs: string[] = [];
  for (const server of ["time.cloudflare.com", "ptbtime1.ptb.de"]) {
    try {
      const p = new Deno.Command("w32tm", { args: ["/stripchart", `/computer:${server}`, "/samples:3", "/dataonly"], stdout: "piped", stderr: "null" });
      outs.push(new TextDecoder().decode((await p.output()).stdout));
    } catch { /* not on Windows or blocked */ }
  }
  return parseW32tmOffsets(outs);
}

async function freeDiskBytes(dir: string): Promise<number | null> {
  try {
    const drive = (dir.match(/^([A-Za-z]):/)?.[1] ?? "C").toUpperCase();
    const p = new Deno.Command("powershell", { args: ["-NoProfile", "-Command", `(Get-PSDrive -Name ${drive}).Free`], stdout: "piped", stderr: "null" });
    const { stdout } = await p.output();
    const n = Number(new TextDecoder().decode(stdout).trim());
    return Number.isFinite(n) ? n : null;
  } catch {
    return null;
  }
}

async function preflight(outDir: string) {
  const fp = await rulesFingerprint();
  let approved: string | null = null;
  try { approved = (await Deno.readTextFile(`${outDir}/APPROVED_RULES_FINGERPRINT`)).trim(); } catch { approved = null; }
  const skew = (await ntpSkew()) ?? estimateSkew(await skewSamples());
  const result = evaluatePreflight({ skew, freeDiskBytes: await freeDiskBytes(outDir), identity: await verifyIdentityNow(), rulesFingerprint: fp, approvedFingerprint: approved });
  return { fp, skew, result };
}

async function report(outDir: string, finalAnalysis: boolean, frozenRequiredWindows: number | null) {
  const files = [...Deno.readDirSync(outDir)].map((e) => e.name);
  const load = async (kind: string) => (await Promise.all(files.filter((f) => f.endsWith(`.${kind}.jsonl`)).map((f) => readJsonl(`${outDir}/${f}`)))).flat();
  const observations = (await load("observation")) as unknown as Observation[];
  const captures = (await load("capture")) as unknown as Capture[];
  const windowsRaw = await load("window");
  const resolutions = await load("resolution");
  const winner = new Map(resolutions.map((r) => [String(r.slug), r.winner === "Up" || r.winner === "Down" ? (r.winner as Outcome) : null]));
  const windows: WindowInfo[] = windowsRaw.map((w) => ({ slug: String(w.slug), feeRate: typeof w.feeRate === "number" ? w.feeRate : null, winner: winner.get(String(w.slug)) ?? null }));
  const summary = summarize(observations, evaluate(observations, captures, windows), RULES_VERSION, { finalAnalysis, frozenRequiredWindows });
  await Deno.writeTextFile(`${outDir}/summary.json`, JSON.stringify(summary, null, 2));
  return summary;
}

if (import.meta.main) {
  const args = Deno.args;
  const outIdx = args.indexOf("--out");
  const outDir = outIdx >= 0 ? args[outIdx + 1] : "C:/Brian/brian-research/niulai4-pilot";
  if (args.includes("--selftest")) {
    const p = await preflight(outDir);
    console.log(JSON.stringify({ mode: "selftest", rulesVersion: RULES_VERSION, rulesFingerprint: p.fp, ...p.result }, null, 2));
  } else if (args.includes("--report")) {
    // Interim (descriptive) by default. A verdict needs BOTH --final and a --required-windows value
    // frozen from the pilot before the test period started.
    const reqIdx = args.indexOf("--required-windows");
    const req = reqIdx >= 0 ? Number(args[reqIdx + 1]) : NaN;
    console.log(JSON.stringify(await report(outDir, args.includes("--final"), Number.isFinite(req) && req > 0 ? req : null), null, 2));
  } else if (args.includes("--start")) {
    const p = await preflight(outDir);
    if (!p.result.ok) {
      console.error(JSON.stringify({ mode: "start", refused: true, ...p.result }, null, 2));
      Deno.exit(2);
    }
    const offsetMs = p.skew?.offsetMs ?? 0;
    const clock = () => Date.now() + offsetMs;
    const guard = new StopGuard(clock());
    const stop = () => guard.stop("OPERATOR");
    try { Deno.addSignalListener("SIGINT", stop); } catch { /* not available */ }
    const opt = (name: string) => (args.indexOf(name) >= 0 ? args[args.indexOf(name) + 1] ?? null : null);
    const sinkKey = await readSinkKey(opt("--core-sink-key-file"));
    const sink = sinkKey ? new CoreSink(sinkKey, newRunId(clock()), RULES_VERSION, p.fp, clock) : null;
    let interim: number | undefined;
    if (sink) {
      console.error(JSON.stringify({ mode: "start", coreSink: "ON", runId: sink.runId, start: await sink.start(opt("--host-label")) }));
      // Interim summaries carry INTERIM_NO_VERDICT; a verdict is only ever produced offline with --final.
      interim = setInterval(() => {
        report(outDir, false, null).then((s) => sink.report(s as unknown as Record<string, unknown>, "INTERIM")).catch(() => {});
      }, 30 * 60_000);
    }
    const res = await runPilot({
      outDir, rulesFingerprint: p.fp, clockOffsetMs: offsetMs, guard,
      onHealth: sink ? (h) => { void sink.heartbeat(h); } : undefined,
    });
    if (interim !== undefined) clearInterval(interim);
    if (sink) {
      try { await sink.report((await report(outDir, false, null)) as unknown as Record<string, unknown>, "INTERIM"); } catch { /* best effort */ }
      await sink.stop(String(res.reason ?? "UNKNOWN"));
    }
    console.log(JSON.stringify({ mode: "start", stopped: res.reason, stats: res.stats, coreSink: sink ? sink.stats : "OFF" }, null, 2));
  } else {
    console.log(`usage: --selftest | --start --out <dir> [--core-sink-key-file <path>] | --report --out <dir>  (rules ${RULES_VERSION}, wallet ${RULES.wallet})`);
  }
}

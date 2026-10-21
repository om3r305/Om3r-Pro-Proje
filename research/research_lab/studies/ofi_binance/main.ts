// OFI collector entry point.
//   --selftest --out <dir>   clock, disk, rules fingerprint, approval check; records nothing
//   --start --out <dir>      collect; REFUSES unless <dir>/APPROVED_RULES_FINGERPRINT matches
//
// deno run --allow-net --allow-read --allow-write --allow-run=powershell,w32tm research/research_lab/studies/ofi_binance/main.ts --selftest --out <dir>

import { parseW32tmOffsets } from "../../../niulai4_pilot/guards.ts";
import { runCollection, type StopReason } from "./runtime.ts";
import { RULES, RULES_VERSION, rulesFingerprint } from "./rules.ts";

async function ntp(): Promise<{ offsetMs: number; uncertaintyMs: number } | null> {
  const outs: string[] = [];
  for (const server of ["time.cloudflare.com", "ptbtime1.ptb.de"]) {
    try {
      const p = new Deno.Command("w32tm", { args: ["/stripchart", `/computer:${server}`, "/samples:3", "/dataonly"], stdout: "piped", stderr: "null" });
      outs.push(new TextDecoder().decode((await p.output()).stdout));
    } catch { /* not Windows */ }
  }
  return parseW32tmOffsets(outs);
}

async function freeDisk(dir: string): Promise<number | null> {
  try {
    const drive = (dir.match(/^([A-Za-z]):/)?.[1] ?? "C").toUpperCase();
    const p = new Deno.Command("powershell", { args: ["-NoProfile", "-Command", `(Get-PSDrive -Name ${drive}).Free`], stdout: "piped", stderr: "null" });
    const n = Number(new TextDecoder().decode((await p.output()).stdout).trim());
    return Number.isFinite(n) ? n : null;
  } catch { return null; }
}

export function preflight(input: { skew: { offsetMs: number; uncertaintyMs: number } | null; freeBytes: number | null; fp: string; approved: string | null }) {
  const checks = [
    { name: "clock", ok: !!input.skew && input.skew.uncertaintyMs <= RULES.maxClockUncertaintyMs, detail: input.skew ? `offset ${input.skew.offsetMs} ms ± ${input.skew.uncertaintyMs} ms` : "no NTP sample" },
    { name: "free_disk", ok: input.freeBytes !== null && input.freeBytes >= 2 * RULES.diskCapBytes, detail: input.freeBytes === null ? "unknown" : `${Math.round(input.freeBytes / 1024 ** 2)} MB free` },
    { name: "rules_approved", ok: input.approved === input.fp, detail: input.approved === null ? "no approval file" : input.approved === input.fp ? "matches" : "approval is for different rules" },
  ];
  return { ok: checks.every((c) => c.ok), checks };
}

if (import.meta.main) {
  const args = Deno.args;
  const outDir = args[args.indexOf("--out") + 1] ?? "";
  if (!outDir || args.indexOf("--out") < 0) {
    console.log(`usage: --selftest|--start --out <dir>  (rules ${RULES_VERSION})`);
    Deno.exit(1);
  }
  const fp = await rulesFingerprint();
  let approved: string | null = null;
  try { approved = (await Deno.readTextFile(`${outDir}/APPROVED_RULES_FINGERPRINT`)).trim(); } catch { approved = null; }
  const skew = await ntp();
  const pf = preflight({ skew, freeBytes: await freeDisk(outDir), fp, approved });
  if (args.includes("--selftest")) {
    console.log(JSON.stringify({ mode: "selftest", rulesVersion: RULES_VERSION, rulesFingerprint: fp, ...pf }, null, 2));
  } else if (args.includes("--start")) {
    if (!pf.ok) {
      console.error(JSON.stringify({ mode: "start", refused: true, ...pf }, null, 2));
      Deno.exit(2);
    }
    const stopSignal: { reason: StopReason | null } = { reason: null };
    try { Deno.addSignalListener("SIGINT", () => { stopSignal.reason ??= "OPERATOR"; }); } catch { /* n/a */ }
    const res = await runCollection({ outDir, rulesFingerprint: fp, clockOffsetMs: skew!.offsetMs, stopSignal });
    console.log(JSON.stringify({ mode: "start", ...res }, null, 2));
  }
}
